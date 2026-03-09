"""
F1 2026 TimeCopilot Prediction Runner
=======================================
Main CLI entry point for the F1 prediction pipeline.

Usage:
    # Full pre-weekend pipeline (Thursday before the race)
    python run_f1_forecast.py --race australia

    # Session-level update (after each session completes)
    python run_f1_forecast.py --race australia --session fp1
    python run_f1_forecast.py --race australia --session qualifying

    # Force rebuild all cached data
    python run_f1_forecast.py --race australia --refresh

    # Check if the hard-coded calendar matches live FastF1 schedule
    python run_f1_forecast.py --check-calendar

    # Print the full 2026 calendar
    python run_f1_forecast.py --print-calendar

    # Detect what race weekend is happening right now
    python run_f1_forecast.py --current

    # Verify a specific race year against live data
    python run_f1_forecast.py --check-calendar --year 2026
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


def _load_env() -> None:
    """Load .env file from the project root if present."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)  # don't override already-set env vars
    except ImportError:
        # Fallback: parse .env manually (no python-dotenv needed)
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            import os
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()


def main():
    parser = argparse.ArgumentParser(
        description="F1 2026 TimeCopilot Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--race", "-r",
        type=str,
        help="Race slug (e.g. australia, china, bahrain). Required unless using --current or --print-calendar.",
    )
    parser.add_argument(
        "--session", "-s",
        type=str,
        default=None,
        choices=["pre_weekend", "fp1", "fp2", "fp3", "qualifying",
                 "sprint_qualifying", "sprint"],
        help="Session stage to update. Omit to run full pre-weekend pipeline.",
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2026,
        help="Season year (default: 2026).",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="openai:gpt-4o-mini",
        help="LLM provider for TimeCopilot (default: openai:gpt-4o).",
    )
    parser.add_argument(
        "--last-race",
        type=str,
        default=None,
        help="Slug of the last completed race (for championship context).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force rebuild all cached intermediate data.",
    )
    parser.add_argument(
        "--check-calendar",
        action="store_true",
        help="Check hard-coded calendar against live FastF1 schedule and report differences.",
    )
    parser.add_argument(
        "--print-calendar",
        action="store_true",
        help="Print the full 2026 F1 calendar.",
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Detect the current race weekend and run the appropriate pipeline stage.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without actually running forecasts.",
    )

    args = parser.parse_args()

    from f1_pipeline.collectors.calendar_manager import CalendarManager
    cal = CalendarManager()

    # ── Calendar commands ─────────────────────────────────────────────────────

    if args.print_calendar:
        cal.print_calendar()
        return

    if args.check_calendar:
        print(f"\n🗓  Checking 2026 F1 calendar against live FastF1 schedule...")
        diffs = cal.check_for_updates(verbose=True)
        if diffs:
            print(f"\n⚠️  Update CALENDAR_2026 in calendar_manager.py if these are official changes.")
            sys.exit(1)
        return

    # ── Auto-detect current race weekend ─────────────────────────────────────

    if args.current:
        race = cal.current_race_weekend()
        if race is None:
            next_race = cal.next_race()
            if next_race:
                days_away = (next_race.race_date - date.today()).days
                print(f"No race weekend currently active.")
                print(f"Next race: {next_race.name} on {next_race.race_date} ({days_away} days away).")
            else:
                print("No upcoming races found in the 2026 calendar.")
            return

        print(f"Current race weekend: {race.name} ({race.race_date})")
        completed = cal.available_sessions(race.slug)
        print(f"Completed sessions: {completed}")

        if not completed:
            print("No sessions completed yet — running pre-weekend pipeline.")
            _run_pre_weekend(race.slug, args.year, args.llm, args.refresh, args.last_race, args.dry_run)
        elif completed[-1] == "Race":
            print("Race completed — running post-race championship update.")
            _run_post_race_championship(race.slug, args.year, args.llm, args.dry_run)
        else:
            latest_session = completed[-1]
            print(f"Running update for latest completed session: {latest_session}")
            _run_session_update(race.slug, latest_session, args.year, args.llm, args.dry_run)
        return

    # ── Explicit race + session ───────────────────────────────────────────────

    if args.race is None:
        print("Error: --race is required unless using --current, --print-calendar, or --check-calendar.")
        parser.print_help()
        sys.exit(1)

    # Validate race slug
    try:
        race = cal.get_race(args.race)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    session = args.session or "pre_weekend"

    if args.dry_run:
        print(f"\n[DRY RUN] Would run: race={race.name}, session={session}, year={args.year}, llm={args.llm}")
        print(f"[DRY RUN] Sprint weekend: {race.is_sprint}")
        print(f"[DRY RUN] Race date: {race.race_date}")
        available = cal.available_sessions(race.slug)
        print(f"[DRY RUN] Available sessions today: {available}")
        return

    if session == "pre_weekend":
        _run_pre_weekend(args.race, args.year, args.llm, args.refresh, args.last_race, False)
    elif session == "post_race":
        _run_post_race_championship(args.race, args.year, args.llm, False)
    else:
        # Map CLI session name to FastF1 session name
        session_map = {
            "fp1": "FP1",
            "fp2": "FP2",
            "fp3": "FP3",
            "qualifying": "Qualifying",
            "sprint_qualifying": "Sprint Qualifying",
            "sprint": "Sprint",
        }
        session_name = session_map.get(session, session)
        _run_session_update(args.race, session_name, args.year, args.llm, False)


def _run_pre_weekend(
    race_slug: str, year: int, llm: str, force_refresh: bool, last_race: str, dry_run: bool
) -> None:
    if dry_run:
        print(f"[DRY RUN] run_pre_weekend_pipeline({race_slug!r}, year={year})")
        return
    from f1_pipeline.forecasting.orchestrator import run_pre_weekend_pipeline
    run_pre_weekend_pipeline(
        race_slug=race_slug,
        year=year,
        llm=llm,
        force_refresh=force_refresh,
        last_race_slug=last_race,
    )


def _run_post_race_championship(
    race_slug: str, year: int, llm: str, dry_run: bool
) -> None:
    if dry_run:
        print(f"[DRY RUN] post_race_championship_update({race_slug!r}, year={year})")
        return
    from f1_pipeline.forecasting.orchestrator import post_race_championship_update
    post_race_championship_update(race_slug=race_slug, year=year, llm=llm)


def _run_session_update(
    race_slug: str, session_name: str, year: int, llm: str, dry_run: bool
) -> None:
    if dry_run:
        print(f"[DRY RUN] update_for_session({race_slug!r}, {session_name!r}, year={year})")
        return
    from f1_pipeline.forecasting.orchestrator import update_for_session
    update_for_session(
        race_slug=race_slug,
        session_type=session_name,
        year=year,
        llm=llm,
    )


if __name__ == "__main__":
    main()
