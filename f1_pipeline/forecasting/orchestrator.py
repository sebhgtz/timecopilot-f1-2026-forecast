"""
F1 Prediction Pipeline Orchestrator
======================================
Main entry point that chains all pipeline phases together:

  1. Load historical data (cached after first run)
  2. Build feature time series (championship, circuit, session pace, strategy)
  3. Run TimeCopilot forecasts (championship + race winner)
  4. Generate reports (social cards, LinkedIn post, Plotly charts)

Two public functions:
  - run_pre_weekend_pipeline(race_slug): Full pipeline before race weekend
  - update_for_session(race_slug, session_type): Incremental update after a session

Both functions cache intermediate datasets so subsequent runs are fast.

Usage:
    from f1_pipeline.forecasting.orchestrator import run_pre_weekend_pipeline, update_for_session

    # Full pipeline before the Australian GP:
    run_pre_weekend_pipeline("australia", year=2026)

    # After qualifying completes:
    update_for_session("australia", "Qualifying", year=2026)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from ..collectors.calendar_manager import CalendarManager, DRIVERS_2026, ROOKIES_2026
from ..collectors.historical_collector import (
    build_historical_dataset, build_testing_dataset,
    get_testing_pace_summary, testing_pace_to_position_estimate,
)
from ..collectors.jolpica_collector import JolpicaCollector
from ..collectors.openf1_collector import OpenF1Collector
from ..features.championship_series import build_championship_series, append_current_season
from ..features.circuit_series import (
    build_circuit_series, enrich_with_weather as enrich_circuit_weather,
    enrich_with_strategy, add_current_race_covariates,
)
from ..features.strategy_features import build_strategy_features, get_circuit_strategy
from .race_weekend_updater import RaceWeekendUpdater
from ..reporting.report_generator import ReportGenerator
from ..collectors.weather_fetcher import fetch_circuit_weather_enc
from ..collectors.weather_log import log_race_weather


# ── Configuration ─────────────────────────────────────────────────────────────

HISTORICAL_YEARS = list(range(2015, 2026))       # Championship series
CIRCUIT_YEARS = list(range(2010, 2026))           # Circuit series (Jolpica data from 2010)
OPENF1_YEARS = list(range(2023, 2026))            # OpenF1 strategy data

DATA_CACHE_DIR = Path(".pipeline_cache")
REPORTS_DIR = Path("reports")
FASTF1_CACHE = ".fastf1_cache"
LLM = "openai:gpt-4o-mini"


def run_pre_weekend_pipeline(
    race_slug: str,
    year: int = 2026,
    llm: str = LLM,
    force_refresh: bool = False,
    last_race_slug: Optional[str] = None,
) -> None:
    """
    Run the full pre-weekend prediction pipeline.

    Steps:
      1. Build (or load cached) historical datasets
      2. Build championship series + circuit series
      3. Run championship forecast
      4. Run initial race forecast (historical only)
      5. Generate social card + LinkedIn post + charts

    Args:
        race_slug:      e.g. "australia"
        year:           e.g. 2026
        llm:            LLM provider string (e.g. "openai:gpt-4o-mini")
        force_refresh:  Ignore cached intermediate data and rebuild everything
        last_race_slug: Slug of the most recent completed race (for championship context)
    """
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    cal = CalendarManager()
    race = cal.get_race(race_slug)
    remaining_races = len(cal.remaining_races())

    print(f"\n{'='*60}")
    print(f"  F1 PREDICTION PIPELINE — {race.name} {year}")
    print(f"  Remaining in season: {remaining_races} races")
    print(f"{'='*60}\n")

    # ── Step 1: Build/load historical datasets ────────────────────────────────
    print("📦 Step 1: Loading historical data...")

    jolpica = JolpicaCollector()
    openf1 = OpenF1Collector()

    driver_champ_df, constructor_champ_df = _load_or_build_championship_series(
        jolpica, force_refresh
    )
    circuit_df = _load_or_build_circuit_series(jolpica, openf1, force_refresh)
    strategy_features = _load_or_build_strategy_features(openf1, jolpica, force_refresh)

    # ── Step 2: Append current-season standings ───────────────────────────────
    print("\n📊 Step 2: Appending current season standings...")
    # Use a no-cache collector for the current year so stale Jolpica cache entries
    # (which may contain 2025 data retagged as 2026) don't pollute the series.
    jolpica_fresh = JolpicaCollector(use_cache=False)
    driver_champ_df = append_current_season(driver_champ_df, year, jolpica_fresh, entity="driver")
    constructor_champ_df = append_current_season(constructor_champ_df, year, jolpica_fresh, entity="constructor")
    # Defensive: restrict to active 2026 grid drivers only (guards against stale Jolpica data)
    _allowed_driver_uids = {f"driver_{d}" for d in DRIVERS_2026.keys()}
    driver_champ_df = driver_champ_df[driver_champ_df["unique_id"].isin(_allowed_driver_uids)]

    # Add placeholder row for current race in circuit series.
    # Pass the verified 2026 driver list so retired drivers are excluded,
    # and seed rookie/new-team entries from pre-season testing pace.
    testing_pace = _load_or_build_testing_pace(year)
    new_driver_entries = _build_new_driver_entries(testing_pace)

    # Fetch race-day weather forecast from Open-Meteo (no API key needed)
    print("\n🌦  Fetching race-weekend weather forecast...")
    weather_enc = fetch_circuit_weather_enc(race_slug, race.race_date)

    # Per-constructor car pace rank from mid-season standings (round N-1)
    _car_pace_rank_map: dict = {}
    if race.round > 1:
        try:
            _cons = jolpica.constructor_standings(year, round=race.round - 1)
            if not _cons.empty:
                _car_pace_rank_map = dict(zip(_cons["constructor"], _cons["position"].astype(float)))
        except Exception:
            pass

    circuit_df = add_current_race_covariates(
        circuit_df,
        circuit_slug=race_slug,
        year=year,
        active_drivers=set(DRIVERS_2026.keys()),
        new_driver_entries=new_driver_entries,
        current_team_map=DRIVERS_2026,
        car_pace_rank_map=_car_pace_rank_map,
        weather_enc=weather_enc,
    )

    # ── Step 3: Run forecasts ─────────────────────────────────────────────────
    print("\n🔮 Step 3: Running TimeCopilot forecasts...")

    updater = RaceWeekendUpdater(race_slug=race_slug, year=year, llm=llm, post_call_delay_s=20,
                                 driver_filter=list(DRIVERS_2026.keys()))
    updater.load_data(circuit_df, driver_champ_df, constructor_champ_df, strategy_features)

    last_race_name = (
        cal.get_race(last_race_slug).name if last_race_slug else "Pre-Season"
    )
    race_fc, champ_fc = updater.run_pre_weekend(
        remaining_races=remaining_races,
        last_race_name=last_race_name,
    )
    updater.save_championship_result()

    # ── Step 4: Also run constructor championship ─────────────────────────────
    from .championship_forecaster import ChampionshipForecaster
    cf = ChampionshipForecaster(llm=llm, post_call_delay_s=20)
    cons_fc = cf.forecast_constructors(
        constructor_df=constructor_champ_df,
        remaining_races=remaining_races,
        race_name=last_race_name,
        constructor_filter=list(set(DRIVERS_2026.values())),
        year=year,
    )
    # driver championship also filtered to active grid (avoids retired-era extrapolation)

    # ── Step 5: Generate reports ──────────────────────────────────────────────
    print("\n📄 Step 5: Generating reports...")

    rg = ReportGenerator(race_slug=race_slug, year=year, race_name=race.name)
    rg.generate_all(
        race_forecast=race_fc,
        driver_champ_forecast=champ_fc,
        constructor_champ_forecast=cons_fc,
        prediction_evolution=updater.prediction_evolution(),
        strategy_context=strategy_features,
        session_stage="pre_weekend",
    )

    print(f"\n✅ Pre-weekend pipeline complete for {race.name} {year}")
    print(f"   Reports saved to: {rg.race_dir}/")
    print(f"   Predicted winner: {race_fc.predicted_winner()}")
    print(f"   Predicted champion: {champ_fc.predicted_champion()}")


def update_for_session(
    race_slug: str,
    session_type: str,
    year: int = 2026,
    llm: str = LLM,
) -> None:
    """
    Incremental pipeline update after a session completes.

    Loads the existing updater state, collects the new session,
    re-runs the race forecast with updated covariates, and generates
    an updated report.

    Args:
        race_slug:    e.g. "australia"
        session_type: "FP1" | "FP2" | "FP3" | "Qualifying" | "Sprint Qualifying" | "Sprint"
        year:         e.g. 2026
        llm:          LLM provider string
    """
    print(f"\n{'='*60}")
    print(f"  SESSION UPDATE — {session_type.upper()} | {race_slug} {year}")
    print(f"{'='*60}\n")

    DATA_CACHE_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    cal = CalendarManager()
    race = cal.get_race(race_slug)
    jolpica = JolpicaCollector()
    openf1 = OpenF1Collector()

    # Load cached datasets
    driver_champ_df, constructor_champ_df = _load_or_build_championship_series(
        jolpica, force_refresh=False
    )
    circuit_df = _load_or_build_circuit_series(jolpica, openf1, force_refresh=False)
    strategy_features = _load_or_build_strategy_features(openf1, jolpica, force_refresh=False)

    jolpica_fresh = JolpicaCollector(use_cache=False)
    driver_champ_df = append_current_season(driver_champ_df, year, jolpica_fresh, entity="driver")
    constructor_champ_df = append_current_season(constructor_champ_df, year, jolpica_fresh, entity="constructor")
    _allowed_driver_uids = {f"driver_{d}" for d in DRIVERS_2026.keys()}
    driver_champ_df = driver_champ_df[driver_champ_df["unique_id"].isin(_allowed_driver_uids)]
    testing_pace = _load_or_build_testing_pace(year)
    new_driver_entries = _build_new_driver_entries(testing_pace)

    # For session updates, race_weekend_collector fetches per-session weather
    # (via forecast_weekend_weather inside weekend_summary). We also pre-populate
    # the circuit series placeholder with the race-day forecast so TimeCopilot
    # sees the correct weather_enc covariate.
    weather_enc = fetch_circuit_weather_enc(race_slug, race.race_date)

    _car_pace_rank_map: dict = {}
    if race.round > 1:
        try:
            _cons = jolpica.constructor_standings(year, round=race.round - 1)
            if not _cons.empty:
                _car_pace_rank_map = dict(zip(_cons["constructor"], _cons["position"].astype(float)))
        except Exception:
            pass

    circuit_df = add_current_race_covariates(
        circuit_df,
        circuit_slug=race_slug,
        year=year,
        active_drivers=set(DRIVERS_2026.keys()),
        new_driver_entries=new_driver_entries,
        current_team_map=DRIVERS_2026,
        car_pace_rank_map=_car_pace_rank_map,
        weather_enc=weather_enc,
    )

    updater = RaceWeekendUpdater(race_slug=race_slug, year=year, llm=llm, post_call_delay_s=20,
                                 driver_filter=list(DRIVERS_2026.keys()))
    updater.load_data(circuit_df, driver_champ_df, constructor_champ_df, strategy_features)

    # Restore previously saved stage forecasts if any
    _restore_stage_forecasts(updater, race_slug, year)

    # Run the session update
    race_fc = updater.update(session_type)
    if race_fc is None:
        print(f"  ⏳  No update made — {session_type} data not yet available.")
        return

    # Load the most recent championship forecast
    champ_fc = _load_saved_championship_forecast(race_slug, year)

    # Generate updated report
    rg = ReportGenerator(race_slug=race_slug, year=year, race_name=race.name)
    stage_key = session_type.lower().replace(" ", "_")
    rg.generate_all(
        race_forecast=race_fc,
        driver_champ_forecast=champ_fc,
        constructor_champ_forecast=None,
        prediction_evolution=updater.prediction_evolution(),
        strategy_context=strategy_features,
        session_stage=stage_key,
    )

    print(f"\n✅ {session_type} update complete for {race.name} {year}")
    print(f"   Updated prediction: {race_fc.predicted_winner()}")
    print(f"   Key insight: {race_fc.key_insight}")


def post_race_championship_update(
    race_slug: str,
    year: int = 2026,
    llm: str = LLM,
) -> None:
    """
    Re-run the championship forecast the day after each race.

    Called automatically on Monday morning (race_date + 1) by the daily cron.
    Incorporates the just-completed race's points into both driver and
    constructor championship forecasts and generates a 'post_race' report.
    The race prediction is omitted (race already happened).
    """
    print(f"\n{'='*60}")
    print(f"  POST-RACE CHAMPIONSHIP UPDATE — {race_slug.upper()} {year}")
    print(f"{'='*60}\n")

    DATA_CACHE_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    cal = CalendarManager()
    race = cal.get_race(race_slug)
    # Race is done — remaining_races excludes the completed one
    remaining_races = len(cal.remaining_races())

    jolpica = JolpicaCollector()
    openf1 = OpenF1Collector()

    print("📦 Loading championship data (with latest race result)...")
    driver_champ_df, constructor_champ_df = _load_or_build_championship_series(
        jolpica, force_refresh=False
    )
    jolpica_fresh = JolpicaCollector(use_cache=False)
    driver_champ_df = append_current_season(driver_champ_df, year, jolpica_fresh, entity="driver")
    constructor_champ_df = append_current_season(constructor_champ_df, year, jolpica_fresh, entity="constructor")
    # Defensive filter: drop retired/unknown drivers not in the 2026 grid
    _allowed_driver_uids = {f"driver_{d}" for d in DRIVERS_2026.keys()}
    driver_champ_df = driver_champ_df[driver_champ_df["unique_id"].isin(_allowed_driver_uids)]

    print(f"\n🔮 Running post-race championship forecasts ({remaining_races} races remaining)...")
    from .championship_forecaster import ChampionshipForecaster
    cf = ChampionshipForecaster(llm=llm, post_call_delay_s=20)
    driver_champ_fc = cf.forecast_drivers(
        driver_champ_df,
        remaining_races=remaining_races,
        race_name=race.name,
        driver_filter=list(DRIVERS_2026.keys()),
        year=year,
    )
    constructor_champ_fc = cf.forecast_constructors(
        constructor_champ_df,
        remaining_races=remaining_races,
        race_name=race.name,
        constructor_filter=list(set(DRIVERS_2026.values())),
        year=year,
    )

    # Fetch actual race results for the result section and accuracy logging
    actual_race_results = None
    try:
        actual_race_results = jolpica_fresh.race_results(year, round=race.round)
        if actual_race_results is not None and not actual_race_results.empty:
            print(f"   ✓ Fetched actual race results ({len(actual_race_results)} finishers)")
    except Exception as exc:
        print(f"  ⚠️  Could not fetch actual race results: {exc}")

    # Log accuracy BEFORE generate_all so the result badge reads the correct actual_winner
    if race.round > 0:
        log_race_accuracy(race_slug, year, race.round, jolpica_fresh)

    print("\n📄 Generating post-race report...")
    rg = ReportGenerator(race_slug=race_slug, year=year, race_name=race.name)
    rg.generate_all(
        race_forecast=None,
        driver_champ_forecast=driver_champ_fc,
        constructor_champ_forecast=constructor_champ_fc,
        prediction_evolution=None,
        strategy_context=None,
        session_stage="post_race",
        actual_race_results=actual_race_results,
    )

    print(f"\n✅ Post-race championship update complete for {race.name} {year}")
    print(f"   Predicted champion: {driver_champ_fc.predicted_champion()}")

    # Log actual session weather (qualifying + race) from Open-Meteo archive
    # Accumulates over the season so wet_driver_stats can learn from results
    if race.round > 0:
        from datetime import timedelta
        qualifying_date = race.race_date - timedelta(days=1)
        log_race_weather(
            race_slug=race_slug,
            year=year,
            round_number=race.round,
            race_name=race.name,
            circuit_type=race.circuit_type,
            race_date=race.race_date,
            qualifying_date=qualifying_date,
        )


# ── Accuracy tracker ──────────────────────────────────────────────────────────

# Stage priority for selecting best pre-race prediction to evaluate
_ACCURACY_STAGE_PRIORITY = ["qualifying", "sprint", "fp3", "fp2", "fp1", "pre_weekend"]

ACCURACY_LOG = REPORTS_DIR / "accuracy_log.csv"


def log_race_accuracy(
    race_slug: str,
    year: int,
    round_number: int,
    jolpica: Optional[JolpicaCollector] = None,
) -> None:
    """
    Compare pre-race predicted winner to actual race winner and append to accuracy log.

    Uses the best available stage prediction (qualifying > sprint > fp3 > fp2 > fp1 > pre_weekend).
    Writes one row per race to reports/accuracy_log.csv.

    Columns: race_slug, year, round, race_name, session_stage,
             predicted_winner, actual_winner, correct,
             actual_position_of_predicted, win_probability, timestamp
    """
    race_dir = REPORTS_DIR / f"{race_slug}_{year}"
    if not race_dir.exists():
        print(f"  ⚠️  No reports found for {race_slug}_{year} — skipping accuracy log.")
        return

    # Find the best pre-race forecast stage
    stage_key = None
    top10 = pd.DataFrame()
    for stage in _ACCURACY_STAGE_PRIORITY:
        top10_file = race_dir / f"top10_{stage}.csv"
        if top10_file.exists():
            try:
                top10 = pd.read_csv(top10_file)
                stage_key = stage
                break
            except Exception:
                continue

    if top10.empty or stage_key is None:
        print(f"  ⚠️  No pre-race prediction found for {race_slug}_{year} — skipping accuracy log.")
        return

    # Predicted winner from the top10 CSV
    top10_sorted = top10.sort_values("predicted_position")
    predicted_code = str(top10_sorted.iloc[0].get("driver_code", "?"))
    win_prob = float(top10_sorted.iloc[0].get("win_probability", 0.0))

    # Actual race winner from Jolpica
    actual_winner = "?"
    actual_position_of_predicted = None
    race_name = race_slug.replace("_", " ").title() + " Grand Prix"
    try:
        jol = jolpica or JolpicaCollector()
        results = jol.race_results(year, round_number)
        if not results.empty:
            race_name = results["race_name"].iloc[0] if "race_name" in results.columns else race_name
            winner_row = results[results["finish_position"] == 1]
            if not winner_row.empty:
                actual_winner = str(winner_row.iloc[0]["driver_code"])
            predicted_row = results[results["driver_code"] == predicted_code]
            if not predicted_row.empty:
                actual_position_of_predicted = int(predicted_row.iloc[0]["finish_position"])
    except Exception as exc:
        print(f"  ⚠️  Could not fetch race result for accuracy log: {exc}")

    correct = (predicted_code == actual_winner)

    # Append to CSV
    REPORTS_DIR.mkdir(exist_ok=True)
    row = {
        "race_slug": race_slug,
        "year": year,
        "round": round_number,
        "race_name": race_name,
        "session_stage": stage_key,
        "predicted_winner": predicted_code,
        "actual_winner": actual_winner,
        "correct": correct,
        "actual_position_of_predicted": actual_position_of_predicted,
        "win_probability": round(win_prob, 3),
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if ACCURACY_LOG.exists():
        existing = pd.read_csv(ACCURACY_LOG)
        # Avoid duplicate entries for the same race
        existing = existing[~((existing["race_slug"] == race_slug) & (existing["year"] == year))]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(ACCURACY_LOG, index=False)

    status = "✅ CORRECT" if correct else "❌ WRONG"
    pos_str = f" (actual P{actual_position_of_predicted})" if actual_position_of_predicted else ""
    print(f"\n📊 Accuracy log: {race_name} — predicted {predicted_code}, actual {actual_winner} {status}{pos_str}")
    print(f"   Stage used: {stage_key} | Win probability: {win_prob:.1%}")

    # Print running accuracy
    try:
        total = len(updated)
        correct_count = int(updated["correct"].sum())
        print(f"   Season accuracy: {correct_count}/{total} ({correct_count/total:.0%})")
    except Exception:
        pass


# ── Testing pace helpers ──────────────────────────────────────────────────────

def _load_or_build_testing_pace(year: int) -> dict[str, float]:
    """Load pre-season testing pace summary (cached as JSON)."""
    cache_file = DATA_CACHE_DIR / f"testing_pace_{year}.json"
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    if cache_file.exists():
        import json
        return json.loads(cache_file.read_text())
    try:
        pace = get_testing_pace_summary(year=year)
        if pace:
            import json
            cache_file.write_text(json.dumps(pace))
        return pace
    except Exception as exc:
        print(f"   ⚠️  Could not load testing pace: {exc}")
        return {}


def _build_new_driver_entries(testing_pace: dict[str, float]) -> dict:
    """
    Build new_driver_entries for ROOKIES_2026 using testing pace estimates.
    Only includes drivers with NO historical F1 race data.
    """
    entries = {}
    for code in ROOKIES_2026:
        team = DRIVERS_2026.get(code, "")
        est_pos = testing_pace_to_position_estimate(testing_pace, code)
        entries[code] = {
            "constructor": team,
            "estimated_position": est_pos,
            "car_pace_rank": 10.0,
        }
    return entries


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_or_build_championship_series(
    jolpica: JolpicaCollector,
    force_refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver_cache = DATA_CACHE_DIR / "driver_championship_series.parquet"
    cons_cache = DATA_CACHE_DIR / "constructor_championship_series.parquet"

    if not force_refresh and driver_cache.exists() and cons_cache.exists():
        print("   ✓ Loading cached championship series...")
        return pd.read_parquet(driver_cache), pd.read_parquet(cons_cache)

    print("   → Building championship series from Jolpica API (this takes a while)...")
    driver_df, cons_df = build_championship_series(
        years=HISTORICAL_YEARS, jolpica_collector=jolpica
    )
    driver_df.to_parquet(driver_cache, index=False)
    cons_df.to_parquet(cons_cache, index=False)
    return driver_df, cons_df


def _load_or_build_circuit_series(
    jolpica: JolpicaCollector,
    openf1: OpenF1Collector,
    force_refresh: bool,
) -> pd.DataFrame:
    circuit_cache = DATA_CACHE_DIR / "circuit_series.parquet"

    if not force_refresh and circuit_cache.exists():
        print("   ✓ Loading cached circuit series...")
        return pd.read_parquet(circuit_cache)

    print("   → Building circuit series from Jolpica API...")
    circuit_df = build_circuit_series(years=CIRCUIT_YEARS, jolpica=jolpica)

    # Enrich with OpenF1 strategy data
    try:
        rc_data = openf1.bulk_race_control(OPENF1_YEARS, [])
        circuit_df = enrich_circuit_weather(circuit_df, rc_data)

        stints_data = openf1.bulk_stints(OPENF1_YEARS, [])
        circuit_df = enrich_with_strategy(circuit_df, stints_data)
    except Exception as exc:
        print(f"   ⚠️  OpenF1 enrichment failed: {exc}")

    circuit_df.to_parquet(circuit_cache, index=False)
    return circuit_df


def _load_or_build_strategy_features(
    openf1: OpenF1Collector,
    jolpica: JolpicaCollector,
    force_refresh: bool,
) -> dict[str, pd.DataFrame]:
    strategy_cache = DATA_CACHE_DIR / "strategy_features.json"

    if not force_refresh and strategy_cache.exists():
        print("   ✓ Loading cached strategy features...")
        cached = json.loads(strategy_cache.read_text())
        return {k: pd.DataFrame(v) for k, v in cached.items()}

    print("   → Building strategy features from OpenF1...")
    features = build_strategy_features(
        years=OPENF1_YEARS,
        openf1=openf1,
        jolpica=jolpica,
    )
    # Save to JSON
    strategy_cache.write_text(
        json.dumps(
            {k: v.to_dict(orient="records") for k, v in features.items()},
            default=str,
        )
    )
    return features


def _restore_stage_forecasts(updater: RaceWeekendUpdater, race_slug: str, year: int) -> None:
    """Restore previously saved stage forecasts from disk."""
    race_dir = REPORTS_DIR / f"{race_slug}_{year}"
    if not race_dir.exists():
        return
    for json_file in race_dir.glob("race_forecast_*.json"):
        stage_key = json_file.stem.replace("race_forecast_", "")
        try:
            meta = json.loads(json_file.read_text())
            # Reconstruct a minimal RaceForecastResult from saved metadata
            from .race_forecaster import RaceForecastResult
            top10_file = race_dir / f"top10_{stage_key}.csv"
            top10 = pd.read_csv(top10_file) if top10_file.exists() else pd.DataFrame()
            fc = RaceForecastResult(
                race_name=meta["race_name"],
                circuit_slug=meta["circuit_slug"],
                year=meta["year"],
                session_stage=stage_key,
                predicted_top10=top10,
                forecast_df=pd.DataFrame(),
                narrative=meta.get("narrative", ""),
                key_insight=meta.get("key_insight", ""),
                timestamp=meta.get("timestamp"),
            )
            updater._stage_forecasts[stage_key] = fc
        except Exception:
            pass


def _load_saved_championship_forecast(race_slug: str, year: int):
    """Load the most recently saved championship forecast."""
    fc_file = REPORTS_DIR / f"{race_slug}_{year}" / "championship_forecast.json"
    if not fc_file.exists():
        return None
    try:
        meta = json.loads(fc_file.read_text())
        from .championship_forecaster import ChampionshipForecastResult
        fc = ChampionshipForecastResult(
            entity=meta.get("entity", "driver"),
            race_name=meta.get("race_name", ""),
            remaining_races=meta.get("remaining_races", 0),
            forecast_df=pd.DataFrame(),
            current_standings=pd.DataFrame(),
            predicted_final=pd.DataFrame(meta.get("top5", [])),
            narrative=meta.get("narrative", ""),
            timestamp=meta.get("timestamp"),
        )
        return fc
    except Exception:
        return None
