#!/usr/bin/env python3
"""
Backtest: FORMULA 1 QATAR AIRWAYS AZERBAIJAN GRAND PRIX 2025
=============================================================
Runs every stage of the Azerbaijan 2025 weekend using only data that
would have been available at that point in time.

Circuit: Baku City Circuit, Baku, Azerbaijan
Date:    19–21 September 2025 (STANDARD WEEKEND)
           FP1 + FP2:        Friday 19 Sep
           FP3 + Qualifying: Saturday 20 Sep
           Race:             Sunday 21 Sep
Round:   17 of 24, 7 races remaining after Azerbaijan

This backtest exercises all implemented features:
  ✓ Per-session weather from Open-Meteo archive (Baku = hot, usually dry)
  ✓ Wet-driver-stats injected into LLM query when wet conditions detected
  ✓ 2025 team assignments (HAM→Ferrari, SAI→Williams, ANT→Mercedes, ...)
  ✓ Active-driver filter using 2025 grid (BOT / PER retired after 2024)
  ✓ Standard weekend format (FP1 → FP2 → FP3 → Qualifying → Race)
  ✓ New-driver synthetic entries (ANT/BOR/HAD/BEA/LAW — no prior Baku history)
  ✓ Prediction evolution chart across pre_weekend → fp1 → fp2 → fp3 → qualifying
  ✓ Post-race accuracy log (predicted vs actual winner)
  ✓ Post-race weather log (qualifying + race archive weather for learning dataset)
  ✓ Constructor name normalisation via _CONSTRUCTOR_LINEAGE (Kick Sauber → Audi)

Reports saved to: reports/azerbaijan_2025/

Usage:
    uv run python backtest_azerbaijan_2025.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta
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

RACE_SLUG  = "azerbaijan"
YEAR       = 2025
RACE_DATE  = pd.Timestamp("2025-09-21")
RACE_NAME  = "Azerbaijan Grand Prix"
LLM        = "openai:gpt-4o-mini"

ROUND_NUMBER    = 17          # Azerbaijan was Round 17 in 2025
RACES_BEFORE    = 16          # rounds completed before Azerbaijan (rounds 1–16)
REMAINING_PRE   = 24 - RACES_BEFORE   # = 8 (Azerbaijan + 7 after, for pre-race forecast)
REMAINING_POST  = 24 - ROUND_NUMBER   # = 7 (races after Azerbaijan, for post-race forecast)

# ── 2025 active driver grid ───────────────────────────────────────────────────
# Used as current_team_map so circuit-series placeholder rows show 2025 teams,
# not the historical team from the last entry in the series.
# Key transfers vs 2024: HAM→Ferrari, SAI→Williams, ANT→Mercedes (rookie),
#                        TSU→Red Bull, LAW+HAD→Racing Bulls, OCO→Haas,
#                        BOR→Kick Sauber (rookie), COL→Alpine (mid-season).

DRIVERS_2025: dict[str, str] = {
    # Red Bull Racing
    "VER": "Red Bull Racing",
    "TSU": "Red Bull Racing",       # Tsunoda promoted from VCARB
    # Ferrari
    "LEC": "Ferrari",
    "HAM": "Ferrari",               # Hamilton joined from Mercedes
    # McLaren
    "NOR": "McLaren",
    "PIA": "McLaren",
    # Mercedes
    "RUS": "Mercedes",
    "ANT": "Mercedes",              # Antonelli (rookie, replaced Hamilton)
    # Aston Martin
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    # Alpine — Colapinto replaced Doohan mid-season; by Round 17 COL was racing
    "GAS": "Alpine",
    "COL": "Alpine",
    # Williams
    "ALB": "Williams",
    "SAI": "Williams",              # Sainz joined from Ferrari
    # Haas
    "BEA": "Haas",                  # Bearman
    "OCO": "Haas",                  # Ocon moved from Alpine
    # Kick Sauber
    "HUL": "Kick Sauber",
    "BOR": "Kick Sauber",           # Bortoleto (rookie)
    # Racing Bulls (VCARB)
    "LAW": "Racing Bulls",
    "HAD": "Racing Bulls",          # Hadjar (rookie)
}

# ── Step 1: Inject Azerbaijan 2025 into CalendarManager ──────────────────────
# CalendarManager only knows 2026 races.  Inject a 2025 Azerbaijan entry so that
# RaceWeekendCollector finds correct 2025 session dates and coordinates.

from f1_pipeline.collectors.calendar_manager import (
    CalendarManager, Race, _make_standard_weekend,
)

_AZERBAIJAN_2025 = Race(
    round=ROUND_NUMBER,
    name=RACE_NAME,
    slug=RACE_SLUG,
    circuit="Baku City Circuit",
    location="Baku, Azerbaijan",
    circuit_type="street",
    race_date=date(2025, 9, 21),
    # Standard weekend: FP1+FP2 Fri 19 Sep, FP3+Qualifying Sat 20 Sep, Race Sun 21 Sep
    sessions=_make_standard_weekend(date(2025, 9, 21)),
)


class _BacktestCalendar(CalendarManager):
    """CalendarManager subclass that resolves 'azerbaijan' to the 2025 entry."""

    def get_race(self, slug: str) -> Race:
        if slug == RACE_SLUG:
            return _AZERBAIJAN_2025
        return super().get_race(slug)

    def all_races(self, include_tests: bool = False) -> list[Race]:
        return [_AZERBAIJAN_2025]

    def remaining_races(self, as_of=None) -> list[Race]:
        return []  # season isn't over; handled via REMAINING_POST constant


# Patch before RaceWeekendCollector is imported (it creates CalendarManager on init)
import f1_pipeline.collectors.race_weekend_collector as _rw_mod
_rw_mod.CalendarManager = _BacktestCalendar

# ── Pipeline imports (after patch) ───────────────────────────────────────────

from f1_pipeline.collectors.jolpica_collector import JolpicaCollector
from f1_pipeline.collectors.openf1_collector import OpenF1Collector
from f1_pipeline.collectors.race_weekend_collector import RaceWeekendCollector
from f1_pipeline.collectors.weather_fetcher import fetch_weekend_weather
from f1_pipeline.collectors.weather_log import log_race_weather, wet_driver_stats
from f1_pipeline.features.championship_series import (
    build_driver_championship_series,
    build_constructor_championship_series,
    _CONSTRUCTOR_LINEAGE,
)
from f1_pipeline.features.circuit_series import (
    build_circuit_series, add_current_race_covariates, enrich_with_strategy,
)
from f1_pipeline.features.strategy_features import build_strategy_features, get_circuit_strategy
from f1_pipeline.forecasting.race_forecaster import RaceForecaster, RaceForecastResult
from f1_pipeline.forecasting.championship_forecaster import ChampionshipForecaster
from f1_pipeline.reporting.report_generator import ReportGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_championship_data(
    jolpica: JolpicaCollector,
    cutoff: Optional[pd.Timestamp],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build championship series for all years up to the given date cutoff.

    cutoff=RACE_DATE  → includes 2025 rounds 1–16 only (pre-Azerbaijan simulation)
    cutoff=None       → includes all 2025 rounds through Azerbaijan (round 17)
    """
    print("  → Historical standings 2015–2024...")
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
    "fp1":         "After FP1",
    "fp2":         "After FP2",
    "fp3":         "After FP3",
    "qualifying":  "After Qualifying",
    "post_race":   "Post-Race",
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
) -> Optional[RaceForecastResult]:
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
            "stage":            _STAGE_LABELS.get(stage, stage.upper()),
            "predicted_winner": winner.get("driver_code", "?"),
            "win_probability":  winner.get("win_probability", 0.0),
        }])
        full_evolution = (
            pd.concat([full_evolution, new_row], ignore_index=True)
            if full_evolution is not None and not full_evolution.empty
            else new_row
        )

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
    print(f"\n  ✅ Stage '{stage}' complete — reports/azerbaijan_2025/")
    return race_fc


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  BACKTEST: AZERBAIJAN GRAND PRIX 2025 (STANDARD WEEKEND)")
    print("  Baku City Circuit, 19–21 September 2025")
    print("  Round 17/24 — 7 races remaining after Azerbaijan")
    print("  Stages: Pre-Weekend → FP1 → FP2 → FP3 → Qualifying → Post-Race")
    print("=" * 60)

    jolpica = JolpicaCollector()
    openf1  = OpenF1Collector()

    # ── Per-session weather from Open-Meteo archive ───────────────────────────
    # Baku is typically hot and dry in September, but street circuits can be
    # affected by unpredictable local conditions.  Fetch archive data for each
    # actual session date so the LLM query gets session-level weather context.
    print("\n🌦  Fetching per-session weather from Open-Meteo archive (Baku 2025)...")
    weekend_weather = fetch_weekend_weather(
        circuit_slug=RACE_SLUG,
        race_date=date(2025, 9, 21),
        session_dates={
            "FP1":        date(2025, 9, 19),  # Friday
            "FP2":        date(2025, 9, 19),  # Friday
            "Qualifying": date(2025, 9, 20),  # Saturday
            "Race":       date(2025, 9, 21),  # Sunday
        },
    )
    for sess, w in weekend_weather.items():
        enc = w.get("enc")
        precip = w.get("precip_mm")
        prob = w.get("precip_prob")
        cond = "wet" if enc == 1 else ("dry" if enc == 0 else "?")
        prob_str = f", prob={prob:.0f}%" if prob is not None else ""
        precip_str = f", precip={precip:.1f}mm" if precip is not None else ""
        print(f"   {sess:12s}: {cond.upper()}{prob_str}{precip_str}")

    # Race-day enc for circuit series placeholder row
    race_weather_enc = (weekend_weather.get("Race") or {}).get("enc")
    quali_weather_enc = (weekend_weather.get("Qualifying") or {}).get("enc")
    # If qualifying OR race is wet → mark circuit as wet (conservative)
    circuit_weather_enc = 1 if (race_weather_enc == 1 or quali_weather_enc == 1) else (
        race_weather_enc if race_weather_enc is not None else None
    )

    # ── Build static datasets ─────────────────────────────────────────────────

    print("\n📦 Building Azerbaijan circuit series (2010–2024)...")
    circuit_df = build_circuit_series(
        years=list(range(2010, 2025)),
        jolpica=jolpica,
        circuits=[RACE_SLUG],
    )

    # Drivers with no Azerbaijan Grand Prix history (2010–2024) get synthetic entries
    # so TimeCopilot can still forecast for them with an estimated baseline position.
    #
    # Included (no Baku race history):
    #   ANT — 2025 rookie, never raced Baku
    #   BOR — 2025 rookie, never raced Baku
    #   HAD — 2025 rookie, never raced Baku
    #   BEA — joined F1 mid-2024 (Monza onward), Baku is before Monza in 2024 calendar
    #   LAW — joined F1 mid-2023 (Singapore R15) and mid-2024 (Singapore R18);
    #          Azerbaijan has always been earlier in the calendar than Singapore
    #
    # Excluded (have prior Baku circuit history):
    #   COL — raced Baku in 2024 with Williams (Azerbaijan was Round 17 of 2024)
    #   All other 2025 drivers — raced Baku in prior years with their previous teams
    _NEW_DRIVERS_2025 = {
        "ANT": {"constructor": "Mercedes",     "driver_name": "Antonelli", "estimated_position": 12.0},
        "BOR": {"constructor": "Kick Sauber",  "driver_name": "Bortoleto", "estimated_position": 17.0},
        "HAD": {"constructor": "Racing Bulls", "driver_name": "Hadjar",    "estimated_position": 12.0},
        "BEA": {"constructor": "Haas",         "driver_name": "Bearman",   "estimated_position": 15.0},
        "LAW": {"constructor": "Racing Bulls", "driver_name": "Lawson",    "estimated_position": 13.0},
    }

    # Constructor championship standings at Round 16 (last round before Azerbaijan)
    # → gives per-constructor car_pace_rank reflecting 2025 mid-season competitiveness
    print("\n📦 Fetching constructor standings at Round 16 (pre-Azerbaijan)...")
    _car_pace_rank_map: dict = {}
    try:
        _cons_standings = jolpica.constructor_standings(YEAR, round=ROUND_NUMBER - 1)
        if not _cons_standings.empty:
            _car_pace_rank_map = dict(
                zip(_cons_standings["constructor"], _cons_standings["position"].astype(float))
            )
            print(f"   Car pace map: { {k: int(v) for k, v in _car_pace_rank_map.items()} }")
    except Exception as _e:
        print(f"   ⚠️  Could not fetch constructor standings: {_e} — using default 5.0")

    # Add 2025 placeholder row:
    #   active_drivers    = 2025 grid (filters out retired drivers: BOT, PER, etc.)
    #   current_team_map  = 2025 assignments (HAM→Ferrari, SAI→Williams, etc.)
    #   car_pace_rank_map = per-constructor championship position at Round 16
    #   new_driver_entries = rookies/newcomers with no Baku circuit history
    #   weather_enc       = derived from actual archive weather above
    circuit_df = add_current_race_covariates(
        circuit_df,
        circuit_slug=RACE_SLUG,
        year=YEAR,
        active_drivers=set(DRIVERS_2025.keys()),
        current_team_map=DRIVERS_2025,
        car_pace_rank_map=_car_pace_rank_map,
        new_driver_entries=_NEW_DRIVERS_2025,
        weather_enc=circuit_weather_enc,
    )
    print(f"   {circuit_df['unique_id'].nunique()} driver series, {len(circuit_df)} rows")

    # Show wet driver stats if conditions were wet
    if circuit_weather_enc == 1:
        from f1_pipeline.collectors.weather_log import wet_driver_stats, format_wet_stats_for_query
        stats = wet_driver_stats(circuit_df)
        if not stats.empty:
            print("\n🌧  Wet driver advantage at Baku (from historical circuit data):")
            print(stats.head(8).to_string(index=False))
            print(f"\n   LLM snippet: {format_wet_stats_for_query(stats)[:200]}...")

    print("\n📦 Building strategy features (2023–2025)...")
    strategy_features = build_strategy_features(
        years=[2023, 2024, 2025], openf1=openf1, jolpica=jolpica
    )
    strategy_context = get_circuit_strategy(strategy_features, RACE_SLUG)

    print("\n📦 Building pre-race championship series "
          "(2015–2024 + 2025 rounds 1–16)...")
    driver_champ_pre, constructor_champ_pre = _build_championship_data(
        jolpica, cutoff=RACE_DATE,
    )
    print(f"   Drivers: {driver_champ_pre['unique_id'].nunique()} series, "
          f"{len(driver_champ_pre):,} rows")

    # Weekend collector — uses patched CalendarManager → correct 2025 session dates
    rw = RaceWeekendCollector(race_slug=RACE_SLUG, year=YEAR)

    # ── Run all stages ────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  RUNNING ALL 6 STAGES (STANDARD WEEKEND FORMAT)")
    print("=" * 60)

    # Only forecast championship for the 2025 grid.
    # Prevents retired drivers/defunct teams from appearing in standings.
    # _CONSTRUCTOR_LINEAGE normalizes "Kick Sauber" → "Audi" so the filter
    # matches the canonical name used in the championship series unique_ids.
    _driver_filter = list(DRIVERS_2025.keys())
    _constructor_filter = list(set(_CONSTRUCTOR_LINEAGE.get(v, v) for v in DRIVERS_2025.values()))

    evolution: Optional[pd.DataFrame] = None

    # Stage 1: Pre-weekend — historical data only, weather_enc set on circuit_df
    race_fc = _run_stage(
        "pre_weekend", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, None,
        remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        evolution = pd.DataFrame([{
            "stage": _STAGE_LABELS["pre_weekend"],
            "predicted_winner": w.get("driver_code", "?"),
            "win_probability": w.get("win_probability", 0.0),
        }])

    # Stage 2: FP1
    print("\n  Collecting FP1 data from FastF1...")
    rw.collect_session("FP1")
    race_fc = _run_stage(
        "fp1", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(),
        remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{
            "stage": _STAGE_LABELS["fp1"],
            "predicted_winner": w.get("driver_code", "?"),
            "win_probability": w.get("win_probability", 0.0),
        }])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # Stage 3: FP2
    print("\n  Collecting FP2 data from FastF1...")
    rw.collect_session("FP2")
    race_fc = _run_stage(
        "fp2", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(),
        remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{
            "stage": _STAGE_LABELS["fp2"],
            "predicted_winner": w.get("driver_code", "?"),
            "win_probability": w.get("win_probability", 0.0),
        }])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # Stage 4: FP3
    print("\n  Collecting FP3 data from FastF1...")
    rw.collect_session("FP3")
    race_fc = _run_stage(
        "fp3", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(),
        remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{
            "stage": _STAGE_LABELS["fp3"],
            "predicted_winner": w.get("driver_code", "?"),
            "win_probability": w.get("win_probability", 0.0),
        }])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # Stage 5: Qualifying — strongest predictor; weather snapshot includes Sat conditions
    print("\n  Collecting Qualifying data from FastF1...")
    rw.collect_session("Qualifying")
    race_fc = _run_stage(
        "qualifying", circuit_df,
        driver_champ_pre, constructor_champ_pre,
        strategy_context, rw.weekend_summary(),
        remaining_races=REMAINING_PRE,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )
    if race_fc is not None:
        w = race_fc.predicted_winner()
        row = pd.DataFrame([{
            "stage": _STAGE_LABELS["qualifying"],
            "predicted_winner": w.get("driver_code", "?"),
            "win_probability": w.get("win_probability", 0.0),
        }])
        evolution = pd.concat([evolution, row], ignore_index=True) if evolution is not None else row

    # Stage 6: Post-race championship — uses full 2025 standings through Azerbaijan
    print("\n📦 Building post-race championship series (2015–2025 through Azerbaijan)...")
    driver_champ_full, constructor_champ_full = _build_championship_data(
        jolpica, cutoff=None,
    )
    print(f"   Drivers: {driver_champ_full['unique_id'].nunique()} series, "
          f"{len(driver_champ_full):,} rows")

    _run_stage(
        "post_race", circuit_df,
        driver_champ_full, constructor_champ_full,
        strategy_context, None,
        remaining_races=REMAINING_POST,
        skip_race_forecast=True,
        prediction_evolution=evolution,
        driver_filter=_driver_filter,
        constructor_filter=_constructor_filter,
    )

    # ── Post-race logging ─────────────────────────────────────────────────────

    print("\n📊 Post-race logging...")

    # 1. Accuracy log — compare qualifying prediction vs actual winner
    from f1_pipeline.forecasting.orchestrator import log_race_accuracy
    log_race_accuracy(RACE_SLUG, YEAR, ROUND_NUMBER, jolpica)

    # 2. Weather log — record qualifying + race archive weather for learning dataset
    #    This feeds wet_driver_stats() in future race weekends at similar circuits.
    log_race_weather(
        race_slug=RACE_SLUG,
        year=YEAR,
        round_number=ROUND_NUMBER,
        race_name=RACE_NAME,
        circuit_type="street",
        race_date=date(2025, 9, 21),
        qualifying_date=date(2025, 9, 20),
    )

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("  ✅ BACKTEST COMPLETE")
    print(f"  Reports → reports/azerbaijan_2025/")
    print()
    print("  Compare predictions vs actuals:")
    print("    social_card_qualifying.txt         ← predicted race winner")
    print("    social_card_post_race.txt          ← predicted 2025 champion")
    print("    linkedin_post_qualifying.md        ← full pre-race analysis")
    print("    linkedin_post_post_race.md         ← full championship analysis")
    print("    charts/                            ← probability + evolution charts")
    print("    reports/accuracy_log.csv           ← predicted vs actual winner")
    print("    reports/weather_log.csv            ← per-session archive weather")
    print("=" * 60)


if __name__ == "__main__":
    main()
