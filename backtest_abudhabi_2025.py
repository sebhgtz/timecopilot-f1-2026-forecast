#!/usr/bin/env python3
"""
Backtest: Abu Dhabi Grand Prix 2025 — Full Weekend Simulation
=============================================================
Runs every stage of the Abu Dhabi 2025 weekend using only data that
would have been available at that point in time.

Stages simulated:
  1. pre_weekend  — historical circuit data only
  2. fp1          — + FP1 pace data
  3. fp2          — + FP2 pace + long run data
  4. fp3          — + FP3 pace data
  5. qualifying   — + grid positions (most predictive)
  6. post_race    — championship update with final standings

Reports saved to: reports/abu_dhabi_2025/
Compare against actual results to validate prediction quality.

Usage:
    uv run python backtest_abudhabi_2025.py
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# Load .env so OPENAI_API_KEY is available without exporting it in the shell
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

# ── Constants ─────────────────────────────────────────────────────────────────

RACE_SLUG    = "abu_dhabi"
YEAR         = 2025
RACE_DATE    = pd.Timestamp("2025-12-07")
RACE_NAME    = "Abu Dhabi Grand Prix"
LLM          = "openai:gpt-4o-mini"

ROUND_NUMBER = 24          # Abu Dhabi is the season finale
RACES_BEFORE = 23
REMAINING_PRE  = 1         # Abu Dhabi is the last race
REMAINING_POST = 0         # season complete after Abu Dhabi

# ── 2025 active driver grid ───────────────────────────────────────────────────

DRIVERS_2025: dict[str, str] = {
    "VER": "Red Bull Racing",
    "TSU": "Red Bull Racing",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "GAS": "Alpine",
    "COL": "Alpine",
    "ALB": "Williams",
    "SAI": "Williams",
    "BEA": "Haas",
    "OCO": "Haas",
    "HUL": "Kick Sauber",
    "BOR": "Kick Sauber",
    "LAW": "Racing Bulls",
    "HAD": "Racing Bulls",
}

# Drivers with no Abu Dhabi Grand Prix history (2018–2024) get synthetic entries
# so TimeCopilot can still forecast for them with an estimated baseline position.
# COL/LAW/TSU excluded — they raced Abu Dhabi in 2024 and have circuit series data.
_NEW_DRIVERS_2025 = {
    "ANT": {"constructor": "Mercedes",     "driver_name": "Antonelli", "estimated_position": 8.0},
    "BOR": {"constructor": "Kick Sauber",  "driver_name": "Bortoleto", "estimated_position": 16.0},
    "HAD": {"constructor": "Racing Bulls", "driver_name": "Hadjar",    "estimated_position": 12.0},
    "BEA": {"constructor": "Haas",         "driver_name": "Bearman",   "estimated_position": 15.0},
}

# ── Step 1: Inject Abu Dhabi 2025 into CalendarManager ───────────────────────
# The CalendarManager only knows 2026 races. We inject a 2025 Abu Dhabi entry
# so that RaceWeekendCollector can find correct 2025 session dates.

from f1_pipeline.collectors.calendar_manager import (
    CalendarManager, Race, _make_standard_weekend,
)

_ABU_DHABI_2025 = Race(
    round=24,
    name=RACE_NAME,
    slug=RACE_SLUG,
    circuit="Yas Marina Circuit",
    location="Abu Dhabi, UAE",
    circuit_type="street",
    race_date=date(2025, 12, 7),
    sessions=_make_standard_weekend(date(2025, 12, 7)),
)


class _BacktestCalendar(CalendarManager):
    """CalendarManager subclass that resolves 'abu_dhabi' to the 2025 entry."""

    def get_race(self, slug: str) -> Race:
        if slug == RACE_SLUG:
            return _ABU_DHABI_2025
        return super().get_race(slug)

    def all_races(self, include_tests: bool = False) -> list[Race]:
        return [_ABU_DHABI_2025]

    def remaining_races(self, as_of=None) -> list[Race]:
        return []  # season over after Abu Dhabi 2025


# Patch before RaceWeekendCollector is imported (it creates CalendarManager on init)
import f1_pipeline.collectors.race_weekend_collector as _rw_mod
_rw_mod.CalendarManager = _BacktestCalendar

# ── Pipeline imports (after patch) ───────────────────────────────────────────

from f1_pipeline.collectors.jolpica_collector import JolpicaCollector
from f1_pipeline.collectors.openf1_collector import OpenF1Collector
from f1_pipeline.collectors.race_weekend_collector import RaceWeekendCollector
from f1_pipeline.features.championship_series import (
    build_driver_championship_series,
    build_constructor_championship_series,
    _CONSTRUCTOR_LINEAGE,
)
from f1_pipeline.features.circuit_series import (
    build_circuit_series, add_current_race_covariates, enrich_with_strategy,
)
from f1_pipeline.features.strategy_features import build_strategy_features, get_circuit_strategy
from f1_pipeline.forecasting.race_forecaster import RaceForecaster
from f1_pipeline.forecasting.championship_forecaster import ChampionshipForecaster
from f1_pipeline.reporting.report_generator import ReportGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_championship_data(
    jolpica: JolpicaCollector,
    cutoff: Optional[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build championship series for all years up to the given date cutoff.

    cutoff=RACE_DATE   → includes 2025 rounds 1-23 (pre-race simulation)
    cutoff=None        → includes all 2025 rounds including Abu Dhabi result
    """
    print("  → Historical standings 2015-2024...")
    d_hist = build_driver_championship_series(list(range(2015, 2025)), jolpica)
    c_hist = build_constructor_championship_series(list(range(2015, 2025)), jolpica)

    print("  → 2025 season standings...")
    d_2025 = build_driver_championship_series([2025], jolpica)
    c_2025 = build_constructor_championship_series([2025], jolpica)

    if cutoff is not None:
        d_2025 = d_2025[d_2025["ds"] < cutoff] if not d_2025.empty else d_2025
        c_2025 = c_2025[c_2025["ds"] < cutoff] if not c_2025.empty else c_2025

    driver_df = (
        pd.concat([d_hist, d_2025], ignore_index=True)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    constructor_df = (
        pd.concat([c_hist, c_2025], ignore_index=True)
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    return driver_df, constructor_df


_STAGE_LABELS = {
    "pre_weekend": "Pre-Weekend",
    "fp1": "After FP1",
    "fp2": "After FP2",
    "fp3": "After FP3",
    "sprint_qualifying": "After Sprint Quali",
    "sprint": "After Sprint",
    "qualifying": "After Qualifying",
    "post_race": "Post-Race",
}


def _run_stage(
    stage: str,
    circuit_df: pd.DataFrame,
    driver_champ_df: pd.DataFrame,
    constructor_champ_df: pd.DataFrame,
    strategy_context: dict,
    weekend_summary: Optional[dict],
    remaining_races: int,
    skip_race_forecast: bool = False,
    prediction_evolution: Optional[pd.DataFrame] = None,
    driver_filter: Optional[list] = None,
    constructor_filter: Optional[list] = None,
) -> Optional["RaceForecastResult"]:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  STAGE: {stage.upper()}")
    print(sep)

    rf = RaceForecaster(llm=LLM, post_call_delay_s=20)
    cf = ChampionshipForecaster(llm=LLM, post_call_delay_s=20)
    rg = ReportGenerator(race_slug=RACE_SLUG, year=YEAR, race_name=RACE_NAME)

    race_fc = None
    if not skip_race_forecast:
        race_fc = rf.forecast(
            circuit_df=circuit_df,
            circuit_slug=RACE_SLUG,
            year=YEAR,
            weekend_summary=weekend_summary,
            strategy_context=strategy_context,
            session_stage=stage,
        )

    # ChampionshipForecaster handles h >= 1 internally; pass the real remaining_races
    # so the report correctly shows "Season complete" when remaining_races=0.
    driver_champ_fc = cf.forecast_drivers(
        driver_champ_df,
        remaining_races=remaining_races,
        race_name=RACE_NAME,
        driver_filter=driver_filter,
        year=YEAR,
    )
    constructor_champ_fc = cf.forecast_constructors(
        constructor_champ_df,
        remaining_races=remaining_races,
        race_name=RACE_NAME,
        constructor_filter=constructor_filter,
        year=YEAR,
    )

    # Build full evolution including the current stage
    full_evolution = prediction_evolution
    if race_fc is not None:
        winner = race_fc.predicted_winner()
        new_row = pd.DataFrame([{
            "stage": _STAGE_LABELS.get(stage, stage.upper()),
            "predicted_winner": winner.get("driver_code", "?"),
            "win_probability": winner.get("win_probability", 0.0),
        }])
        if full_evolution is not None and not full_evolution.empty:
            full_evolution = pd.concat([full_evolution, new_row], ignore_index=True)
        else:
            full_evolution = new_row

    rg.generate_all(
        race_forecast=race_fc,
        driver_champ_forecast=driver_champ_fc,
        constructor_champ_forecast=constructor_champ_fc,
        prediction_evolution=full_evolution,
        strategy_context=(race_fc.strategy_context if race_fc is not None else strategy_context),
        session_stage=stage,
    )

    if race_fc:
        winner = race_fc.predicted_winner()
        print(f"\n  Predicted winner:    {winner.get('driver_code')} "
              f"({winner.get('constructor')}) — "
              f"{winner.get('win_probability', 0)*100:.1f}%")
    champion = driver_champ_fc.predicted_champion()
    print(f"  Predicted champion:  {champion.get('name')} "
          f"({champion.get('predicted_points', 0):.0f} pts predicted, "
          f"{champion.get('current_points', 0):.0f} pts now)")
    print(f"\n  ✅ Stage '{stage}' complete — reports/abu_dhabi_2025/")
    return race_fc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  BACKTEST: ABU DHABI GRAND PRIX 2025")
    print("  Simulating full race weekend with real historical data")
    print("  All 2025 sessions already happened — data fully available")
    print("=" * 60)

    jolpica = JolpicaCollector()
    openf1  = OpenF1Collector()

    # ── Build static datasets (shared across all stages) ─────────────────────

    print("\n📦 Building Abu Dhabi circuit series (2010–2024)...")
    circuit_df = build_circuit_series(
        years=list(range(2010, 2025)),
        jolpica=jolpica,
        circuits=[RACE_SLUG],
    )

    # Constructor standings at Round 23 (last round before Abu Dhabi, Round 24)
    _car_pace_rank_map: dict = {}
    try:
        _cons = jolpica.constructor_standings(YEAR, round=23)
        if not _cons.empty:
            _car_pace_rank_map = dict(zip(_cons["constructor"], _cons["position"].astype(float)))
    except Exception:
        pass

    # Add 2025 placeholder row:
    #   active_drivers    = 2025 grid (filters out retired drivers: BOT, PER, etc.)
    #   current_team_map  = 2025 assignments (HAM→Ferrari, SAI→Williams, etc.)
    #   car_pace_rank_map = per-constructor championship position at Round 23
    #   new_driver_entries = rookies/newcomers with no Abu Dhabi circuit history
    circuit_df = add_current_race_covariates(
        circuit_df,
        circuit_slug=RACE_SLUG,
        year=YEAR,
        active_drivers=set(DRIVERS_2025.keys()),
        current_team_map=DRIVERS_2025,
        car_pace_rank_map=_car_pace_rank_map,
        new_driver_entries=_NEW_DRIVERS_2025,
    )
    print(f"   {circuit_df['unique_id'].nunique()} driver series, {len(circuit_df)} rows")

    print("\n📦 Building strategy features (2023–2025)...")
    strategy_features = build_strategy_features(years=[2023, 2024, 2025], openf1=openf1, jolpica=jolpica)
    strategy_context  = get_circuit_strategy(strategy_features, RACE_SLUG)

    print("\n📦 Building pre-race championship series (2015–2024 + 2025 rounds 1–23)...")
    driver_champ_pre, constructor_champ_pre = _build_championship_data(
        jolpica, cutoff=RACE_DATE,
    )
    print(f"   Drivers: {driver_champ_pre['unique_id'].nunique()} series, "
          f"{len(driver_champ_pre):,} rows")

    # Weekend collector — uses the patched CalendarManager with 2025 session dates
    rw = RaceWeekendCollector(race_slug=RACE_SLUG, year=YEAR)

    # ── Stage 1: Pre-weekend ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RUNNING ALL 6 STAGES")
    print("=" * 60)

    # Only forecast championship for the 2025 grid.
    # Prevents retired drivers/defunct teams from appearing in standings.
    _driver_filter = list(DRIVERS_2025.keys())
    _constructor_filter = list(set(_CONSTRUCTOR_LINEAGE.get(v, v) for v in DRIVERS_2025.values()))

    evolution: Optional[pd.DataFrame] = None

    race_fc = _run_stage(
        "pre_weekend", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, None, remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        evolution = pd.DataFrame([{"stage": _STAGE_LABELS["pre_weekend"],
                                   "predicted_winner": w.get("driver_code", "?"),
                                   "win_probability": w.get("win_probability", 0.0)}])

    # ── Stage 2: FP1 ─────────────────────────────────────────────────────────
    print("\n  Collecting FP1 data from FastF1...")
    rw.collect_session("FP1")
    race_fc = _run_stage(
        "fp1", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(), remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{"stage": _STAGE_LABELS["fp1"],
                              "predicted_winner": w.get("driver_code", "?"),
                              "win_probability": w.get("win_probability", 0.0)}])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # ── Stage 3: FP2 ─────────────────────────────────────────────────────────
    print("\n  Collecting FP2 data from FastF1...")
    rw.collect_session("FP2")
    race_fc = _run_stage(
        "fp2", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(), remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{"stage": _STAGE_LABELS["fp2"],
                              "predicted_winner": w.get("driver_code", "?"),
                              "win_probability": w.get("win_probability", 0.0)}])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # ── Stage 4: FP3 ─────────────────────────────────────────────────────────
    print("\n  Collecting FP3 data from FastF1...")
    rw.collect_session("FP3")
    race_fc = _run_stage(
        "fp3", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(), remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{"stage": _STAGE_LABELS["fp3"],
                              "predicted_winner": w.get("driver_code", "?"),
                              "win_probability": w.get("win_probability", 0.0)}])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # ── Stage 5: Qualifying ───────────────────────────────────────────────────
    print("\n  Collecting Qualifying data from FastF1...")
    rw.collect_session("Qualifying")
    race_fc = _run_stage(
        "qualifying", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(), remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{"stage": _STAGE_LABELS["qualifying"],
                              "predicted_winner": w.get("driver_code", "?"),
                              "win_probability": w.get("win_probability", 0.0)}])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # ── Stage 6: Post-race ────────────────────────────────────────────────────
    print("\n📦 Building post-race championship series (2015–2025 complete)...")
    driver_champ_full, constructor_champ_full = _build_championship_data(
        jolpica, cutoff=None,
    )
    print(f"   Drivers: {driver_champ_full['unique_id'].nunique()} series, "
          f"{len(driver_champ_full):,} rows")

    _run_stage(
        "post_race", circuit_df,
        driver_champ_full, constructor_champ_full,
        strategy_context, None, remaining_races=REMAINING_POST,
        skip_race_forecast=True,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ✅ BACKTEST COMPLETE")
    print(f"  Reports → reports/abu_dhabi_2025/")
    print()
    print("  To compare predictions vs actuals, check:")
    print("    social_card_qualifying.txt    ← predicted race winner")
    print("    social_card_post_race.txt     ← predicted 2025 champion")
    print("    linkedin_post_qualifying.md   ← full pre-race analysis")
    print("    linkedin_post_post_race.md    ← full championship analysis")
    print("    charts/                       ← probability charts (PNG)")
    print("=" * 60)


if __name__ == "__main__":
    main()
