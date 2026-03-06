"""
Race Weekend Updater
=====================
Updates the race prediction incrementally as each session of the
race weekend completes.

Sequence:
  1. pre_weekend  — historical data only (Thursday before race)
  2. fp1          — add FP1 relative pace (Friday AM)
  3. fp2          — add FP2 pace + long-run pace (Friday PM)
  4. fp3          — add FP3 pace (Saturday AM)
  5. qualifying   — add grid positions (Saturday PM) ← KEY UPDATE
  6. sprint_quali — add sprint grid (sprint weekends only)
  7. sprint       — add sprint result (sprint weekends only)

Each update re-runs the RaceForecaster with progressively better covariates.
Comparison of prediction confidence across stages is stored for reporting.

Usage:
    updater = RaceWeekendUpdater(race_slug="australia", year=2026)
    updater.run_pre_weekend(circuit_df, championship_df)
    # ... after FP1 completes ...
    updater.update("fp1")
    # ... after qualifying ...
    updater.update("qualifying")
    report = updater.generate_report()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import json

from ..collectors.race_weekend_collector import RaceWeekendCollector
from ..collectors.calendar_manager import CalendarManager
from ..features.strategy_features import get_circuit_strategy
from .race_forecaster import RaceForecaster, RaceForecastResult
from .championship_forecaster import ChampionshipForecaster, ChampionshipForecastResult


SESSION_SEQUENCE = [
    "pre_weekend",
    "FP1",
    "FP2",
    "FP3",
    "Qualifying",
    "Sprint Qualifying",
    "Sprint",
]


class RaceWeekendUpdater:
    """
    Manages the progressive prediction update flow for a race weekend.
    Stores each stage's forecast so we can show the prediction evolution in reports.
    """

    def __init__(
        self,
        race_slug: str,
        year: int = 2026,
        llm: str = "openai:gpt-4o-mini",
        output_dir: str = "reports",
        post_call_delay_s: float = 20.0,
        driver_filter: Optional[list] = None,
    ):
        self.race_slug = race_slug
        self.year = year
        self.llm = llm
        self.output_dir = Path(output_dir) / f"{race_slug}_{year}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._driver_filter = driver_filter  # e.g. list(DRIVERS_2026.keys())

        self.calendar = CalendarManager()
        self.race = self.calendar.get_race(race_slug)
        self.rw_collector = RaceWeekendCollector(race_slug=race_slug, year=year)
        self.rf = RaceForecaster(llm=llm, post_call_delay_s=post_call_delay_s)
        self.cf = ChampionshipForecaster(llm=llm, post_call_delay_s=post_call_delay_s)

        # Storage for each stage's forecast
        self._stage_forecasts: dict[str, RaceForecastResult] = {}
        self._championship_result: Optional[ChampionshipForecastResult] = None
        self._strategy_context: Optional[dict] = None

        # Pre-loaded data
        self._circuit_df: Optional[pd.DataFrame] = None
        self._driver_champ_df: Optional[pd.DataFrame] = None
        self._constructor_champ_df: Optional[pd.DataFrame] = None
        self._strategy_features: Optional[dict] = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(
        self,
        circuit_df: pd.DataFrame,
        driver_champ_df: pd.DataFrame,
        constructor_champ_df: pd.DataFrame,
        strategy_features: Optional[dict] = None,
    ) -> None:
        """Load pre-computed feature datasets."""
        self._circuit_df = circuit_df
        self._driver_champ_df = driver_champ_df
        self._constructor_champ_df = constructor_champ_df
        self._strategy_features = strategy_features

        if strategy_features:
            self._strategy_context = get_circuit_strategy(strategy_features, self.race_slug)

    # ── Pre-weekend (Thursday) ────────────────────────────────────────────────

    def run_pre_weekend(
        self,
        remaining_races: Optional[int] = None,
        last_race_name: str = "Pre-Season",
    ) -> tuple[RaceForecastResult, ChampionshipForecastResult]:
        """
        Run the initial pre-weekend prediction using only historical data.
        This is the first public post of the race week.

        Returns:
            (race_forecast, championship_forecast)
        """
        print(f"\n🏁 Pre-weekend pipeline — {self.race.name} {self.year}")
        self._validate_data_loaded()

        if remaining_races is None:
            remaining_races = len(self.calendar.remaining_races())

        # Race forecast (historical only, no covariates yet)
        race_result = self.rf.forecast(
            circuit_df=self._circuit_df,
            circuit_slug=self.race_slug,
            year=self.year,
            weekend_summary=None,
            strategy_context=self._strategy_context,
            session_stage="pre_weekend",
        )
        self._stage_forecasts["pre_weekend"] = race_result

        # Championship forecast — filter to active grid drivers only
        champ_result = self.cf.forecast_drivers(
            driver_df=self._driver_champ_df,
            remaining_races=remaining_races,
            race_name=last_race_name,
            driver_filter=self._driver_filter,
            year=self.year,
        )
        self._championship_result = champ_result

        self._save_stage("pre_weekend")
        return race_result, champ_result

    # ── Session-by-session updates ────────────────────────────────────────────

    def update(self, session_name: str) -> Optional[RaceForecastResult]:
        """
        Collect session data and update the race prediction.

        session_name: "FP1" | "FP2" | "FP3" | "Qualifying" | "Sprint Qualifying" | "Sprint"

        Returns updated RaceForecastResult, or None if session not available yet.
        """
        print(f"\n🔄 Updating prediction after {session_name}...")

        # Collect the session
        session_data = self.rw_collector.collect_session(session_name)
        if session_data is None:
            print(f"  ⏳  {session_name} not yet available.")
            return None

        # Build weekend summary with all completed sessions so far
        weekend_summary = self.rw_collector.weekend_summary()

        stage_key = session_name.lower().replace(" ", "_")
        result = self.rf.forecast(
            circuit_df=self._circuit_df,
            circuit_slug=self.race_slug,
            year=self.year,
            weekend_summary=weekend_summary,
            strategy_context=self._strategy_context,
            session_stage=stage_key,
        )
        self._stage_forecasts[stage_key] = result
        self._save_stage(stage_key)
        return result

    def update_all_available(self) -> dict[str, RaceForecastResult]:
        """
        Collect and update all sessions that have completed since last run.
        Returns dict of stage_key → RaceForecastResult for completed stages.
        """
        available_sessions = self.calendar.available_sessions(self.race_slug)
        for session_name in available_sessions:
            stage_key = session_name.lower().replace(" ", "_")
            if stage_key not in self._stage_forecasts:
                self.update(session_name)
        return self._stage_forecasts

    # ── Race weekend summary ──────────────────────────────────────────────────

    def latest_race_forecast(self) -> Optional[RaceForecastResult]:
        """Return the most recent race forecast (latest stage completed)."""
        for stage in reversed(SESSION_SEQUENCE):
            key = stage.lower().replace(" ", "_")
            if key in self._stage_forecasts:
                return self._stage_forecasts[key]
        return None

    def prediction_evolution(self) -> pd.DataFrame:
        """
        Return how the predicted winner has changed across stages.
        Useful for showing "prediction journey" in the report.

        Columns: stage, predicted_winner, win_probability, updated_at
        """
        rows = []
        for stage in SESSION_SEQUENCE:
            key = stage.lower().replace(" ", "_")
            if key in self._stage_forecasts:
                fc = self._stage_forecasts[key]
                winner = fc.predicted_winner()
                rows.append({
                    "stage": stage,
                    "predicted_winner": winner.get("driver_code", "?"),
                    "win_probability": winner.get("win_probability", 0.0),
                    "key_insight": fc.key_insight,
                    "updated_at": fc.timestamp,
                })
        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_stage(self, stage_key: str) -> None:
        """Save stage forecast metadata to disk for report generation."""
        fc = self._stage_forecasts.get(stage_key)
        if fc is None:
            return

        meta = {
            "stage": stage_key,
            "race_name": fc.race_name,
            "circuit_slug": fc.circuit_slug,
            "year": fc.year,
            "timestamp": fc.timestamp,
            "predicted_winner": fc.predicted_winner(),
            "podium": fc.predicted_podium(),
            "key_insight": fc.key_insight,
            "narrative": fc.narrative[:2000],  # truncate for storage
        }
        out_file = self.output_dir / f"race_forecast_{stage_key}.json"
        out_file.write_text(json.dumps(meta, indent=2, default=str))

        # Also save predicted top 10 as CSV
        if not fc.predicted_top10.empty:
            fc.predicted_top10.to_csv(
                self.output_dir / f"top10_{stage_key}.csv", index=False
            )

        print(f"  💾  Saved {stage_key} forecast → {out_file}")

    def save_championship_result(self) -> None:
        """Save the championship forecast to disk."""
        if self._championship_result is None:
            return

        cr = self._championship_result
        meta = {
            "entity": cr.entity,
            "race_name": cr.race_name,
            "remaining_races": cr.remaining_races,
            "timestamp": cr.timestamp,
            "predicted_champion": cr.predicted_champion(),
            "top5": cr.top_n().to_dict(orient="records"),
            "narrative": cr.narrative[:2000],
        }
        out_file = self.output_dir / "championship_forecast.json"
        out_file.write_text(json.dumps(meta, indent=2, default=str))

        if not cr.predicted_final.empty:
            cr.predicted_final.to_csv(
                self.output_dir / "championship_final_standings.csv", index=False
            )
        print(f"  💾  Saved championship forecast → {out_file}")

    def _validate_data_loaded(self) -> None:
        if self._circuit_df is None or self._driver_champ_df is None:
            raise RuntimeError(
                "Data not loaded. Call load_data() before running forecasts."
            )
