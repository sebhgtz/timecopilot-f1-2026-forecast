"""
Race Weekend Collector
======================
Collects live / near-live data for the current or most-recent race weekend.
Sources:
  - FastF1 (available 30–120 min after session ends for most data)
  - OpenF1 REST API (available ~3 seconds after events; historical with no auth)

Typical usage flow:
    rw = RaceWeekendCollector(race_slug="australia")
    # After FP1 ends:
    fp1_data = rw.collect_session("FP1")
    # After Qualifying ends:
    quali_data = rw.collect_session("Qualifying")
    # Compile all collected sessions:
    summary = rw.weekend_summary()
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional
import pandas as pd

from .calendar_manager import CalendarManager, Race
from .historical_collector import extract_session_timeseries, enrich_with_weather, setup_cache
from .openf1_collector import OpenF1Collector
from .weather_fetcher import fetch_weekend_weather


class RaceWeekendCollector:
    """
    Collects session-by-session data for a 2026 race weekend.

    Stores each session's DataFrame in memory (and optionally to disk)
    so the orchestrator can incorporate new sessions incrementally.
    """

    def __init__(
        self,
        race_slug: str,
        year: int = 2026,
        cache_dir: str = ".fastf1_cache",
        output_dir: str = "output/race_weekends",
    ):
        self.calendar = CalendarManager()
        self.race = self.calendar.get_race(race_slug)
        self.year = year
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir) / race_slug
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.openf1 = OpenF1Collector()
        self._session_data: dict[str, pd.DataFrame] = {}
        setup_cache(cache_dir)

    # ── Session collection ────────────────────────────────────────────────────

    def collect_session(
        self, session_type: str, force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Collect lap-level data for a single session (e.g. "FP1", "Qualifying").
        Returns None if session data is not yet available.

        Data is cached in memory and on disk for subsequent calls.
        """
        cache_key = session_type.replace(" ", "_")
        cache_file = self.output_dir / f"{cache_key}_laps.parquet"

        # Return cached version if available
        if not force_refresh and cache_key in self._session_data:
            return self._session_data[cache_key]
        if not force_refresh and cache_file.exists():
            df = pd.read_parquet(cache_file)
            self._session_data[cache_key] = df
            return df

        # Check if this session has completed based on calendar
        session_obj = self.race.get_session(session_type)
        if session_obj is None:
            print(f"  ⚠️  Session '{session_type}' not found in {self.race.name} schedule.")
            return None
        if session_obj.date > date.today():
            print(f"  ⏳  {session_type} hasn't happened yet (scheduled {session_obj.date}).")
            return None

        print(f"  → Collecting {self.year} {self.race.name} [{session_type}]...")

        # Pull from FastF1
        df = extract_session_timeseries(
            year=self.year,
            gp=self.race.name,
            session_type=session_type,
            cache_dir=self.cache_dir,
        )

        if df is None or df.empty:
            print(f"     ⚠️  No FastF1 data for {session_type} yet. Trying OpenF1...")
            df = self._collect_via_openf1(session_type)

        if df is None or df.empty:
            print(f"     ⚠️  No data available for {session_type}.")
            return None

        # Enrich with OpenF1 strategy data for race sessions
        if session_type in ("Race", "Sprint"):
            df = self._enrich_with_strategy(df, session_type)

        # Cache to disk
        df.to_parquet(cache_file, index=False)
        self._session_data[cache_key] = df
        print(f"     ✓ {len(df):,} laps collected for {session_type}")
        return df

    def collect_all_completed_sessions(self) -> dict[str, pd.DataFrame]:
        """Collect data for all sessions that have already completed."""
        available = self.calendar.available_sessions(self.race.slug)
        for session_name in available:
            self.collect_session(session_name)
        return self._session_data

    # ── Session feature extraction ────────────────────────────────────────────

    def relative_pace_ranking(self, session_type: str) -> Optional[pd.DataFrame]:
        """
        Return a ranking of drivers by relative pace in a session.

        Columns: driver_code, team, best_lap_s, relative_pace, pace_rank
        """
        df = self._session_data.get(session_type.replace(" ", "_"))
        if df is None:
            df = self.collect_session(session_type)
        if df is None or df.empty:
            return None

        # Best lap per driver
        summary = (
            df.groupby("Driver")
            .agg(
                team=("Team", "first"),
                best_lap_s=("LapTime", "min"),
            )
            .dropna(subset=["best_lap_s"])
            .reset_index()
        )
        summary = summary.rename(columns={"Driver": "driver_code"})
        session_best = summary["best_lap_s"].min()
        summary["relative_pace"] = summary["best_lap_s"] / session_best
        summary = summary.sort_values("relative_pace").reset_index(drop=True)
        summary["pace_rank"] = range(1, len(summary) + 1)
        return summary

    def long_run_pace(self, session_type: str = "FP2", min_stint_laps: int = 8) -> Optional[pd.DataFrame]:
        """
        Extract long-run pace from FP2 (or any session).
        Long runs (stints of ≥min_stint_laps consecutive laps on same compound)
        are a proxy for race pace.

        Columns: driver_code, team, compound, avg_race_pace_s, stint_laps, pace_rank
        """
        df = self._session_data.get(session_type.replace(" ", "_"))
        if df is None:
            df = self.collect_session(session_type)
        if df is None or df.empty:
            return None

        # Filter accurate laps without pits
        laps = df[df.get("IsAccurate", pd.Series(True, index=df.index))].copy()
        if "PitThisLap" in laps.columns:
            laps = laps[laps["PitThisLap"] == 0]

        # Compute stint-level avg pace
        rows = []
        for driver, grp in laps.groupby("Driver"):
            grp = grp.sort_values("LapNumber")
            # Detect stints by compound changes
            if "Compound" in grp.columns:
                grp["stint"] = (grp["Compound"] != grp["Compound"].shift()).cumsum()
            else:
                grp["stint"] = 1
            for _, stint_grp in grp.groupby("stint"):
                if len(stint_grp) >= min_stint_laps and "LapTime" in stint_grp.columns:
                    rows.append({
                        "driver_code": driver,
                        "team": grp["Team"].iloc[0] if "Team" in grp.columns else "",
                        "compound": stint_grp["Compound"].iloc[0] if "Compound" in stint_grp.columns else "",
                        "avg_race_pace_s": stint_grp["LapTime"].median(),
                        "stint_laps": len(stint_grp),
                    })

        if not rows:
            return None

        result = pd.DataFrame(rows)
        # Best long-run pace per driver (minimum avg pace)
        result = (
            result.sort_values("avg_race_pace_s")
            .groupby("driver_code")
            .first()
            .reset_index()
        )
        session_best = result["avg_race_pace_s"].min()
        result["relative_race_pace"] = result["avg_race_pace_s"] / session_best
        result = result.sort_values("relative_race_pace").reset_index(drop=True)
        result["race_pace_rank"] = range(1, len(result) + 1)
        return result

    def qualifying_grid(self) -> Optional[pd.DataFrame]:
        """
        Extract qualifying grid positions from the Qualifying session.
        Returns: driver_code, team, grid_position, q3_time_s (or best available)
        """
        df = self._session_data.get("Qualifying")
        if df is None:
            df = self.collect_session("Qualifying")
        if df is None or df.empty:
            return None

        summary = (
            df.groupby("Driver")
            .agg(team=("Team", "first"), best_lap_s=("LapTime", "min"))
            .dropna(subset=["best_lap_s"])
            .reset_index()
            .rename(columns={"Driver": "driver_code"})
        )
        summary = summary.sort_values("best_lap_s").reset_index(drop=True)
        summary["grid_position"] = range(1, len(summary) + 1)
        return summary

    def sprint_grid(self) -> Optional[pd.DataFrame]:
        """Extract sprint qualifying grid (only for sprint weekends)."""
        if not self.race.is_sprint:
            return None
        df = self._session_data.get("Sprint_Qualifying")
        if df is None:
            df = self.collect_session("Sprint Qualifying")
        if df is None or df.empty:
            return None

        summary = (
            df.groupby("Driver")
            .agg(team=("Team", "first"), best_lap_s=("LapTime", "min"))
            .dropna(subset=["best_lap_s"])
            .reset_index()
            .rename(columns={"Driver": "driver_code"})
        )
        summary = summary.sort_values("best_lap_s").reset_index(drop=True)
        summary["sprint_grid_position"] = range(1, len(summary) + 1)
        return summary

    def sprint_result(self) -> Optional[pd.DataFrame]:
        """
        Extract sprint race finishing positions (only for sprint weekends).
        Columns: driver_code, team, sprint_finish_position, sprint_finish_gap_s

        Sprint results are a strong predictor of main race performance — they
        reveal actual car pace on race setup at the same circuit.
        """
        if not self.race.is_sprint:
            return None
        df = self._session_data.get("Sprint")
        if df is None:
            df = self.collect_session("Sprint")
        if df is None or df.empty:
            return None

        # Get the final lap for each driver (their last lap = their finishing lap)
        # Use FinalPosition if available, otherwise infer from last Position reading
        if "FinalPosition" in df.columns and df["FinalPosition"].notna().any():
            result = (
                df.dropna(subset=["FinalPosition"])
                .groupby("Driver")
                .agg(
                    team=("Team", "first"),
                    sprint_finish_position=("FinalPosition", "last"),
                )
                .reset_index()
                .rename(columns={"Driver": "driver_code"})
            )
        elif "Position" in df.columns and df["Position"].notna().any():
            result = (
                df.dropna(subset=["Position"])
                .groupby("Driver")
                .agg(
                    team=("Team", "first"),
                    sprint_finish_position=("Position", "last"),
                )
                .reset_index()
                .rename(columns={"Driver": "driver_code"})
            )
        else:
            return None

        result["sprint_finish_position"] = pd.to_numeric(
            result["sprint_finish_position"], errors="coerce"
        )
        result = result.dropna(subset=["sprint_finish_position"])
        result = result.sort_values("sprint_finish_position").reset_index(drop=True)
        return result

    # ── Weather ───────────────────────────────────────────────────────────────

    def race_day_weather(self) -> Optional[pd.DataFrame]:
        """
        Fetch race-day weather from OpenF1 (available in real-time + historical).
        Returns: air_temperature, track_temperature, humidity, rainfall, wind_speed
        """
        try:
            return self.openf1.weather(self.year, self.race.name, "Race")
        except Exception as exc:
            print(f"  ⚠️  Could not fetch race day weather: {exc}")
            return None

    def session_weather(self, session_type: str) -> Optional[pd.DataFrame]:
        """Fetch weather for any session in this race weekend."""
        try:
            return self.openf1.weather(self.year, self.race.name, session_type)
        except Exception as exc:
            print(f"  ⚠️  Could not fetch {session_type} weather: {exc}")
            return None

    def forecast_weekend_weather(self) -> dict[str, dict]:
        """
        Fetch weather forecasts (or archive actuals) for all race weekend sessions
        using Open-Meteo. Returns a dict of {session_name: weather_dict}.

        Each weather_dict contains:
            enc         — 0=dry, 1=wet, None=unavailable
            precip_mm   — precipitation amount in mm
            precip_prob — precipitation probability % (forecast only)

        Session dates are derived from the calendar. For standard weekends:
            FP1, FP2 → Friday; FP3, Qualifying → Saturday; Race → Sunday
        For sprint weekends:
            FP1, Sprint Qualifying → Friday; Sprint, Qualifying → Saturday; Race → Sunday
        """
        race_date = self.race.race_date
        session_dates = {}
        for session in self.race.sessions:
            session_dates[session.name] = session.date

        return fetch_weekend_weather(
            circuit_slug=self.race.slug,
            race_date=race_date,
            session_dates=session_dates,
        )

    # ── Strategy ─────────────────────────────────────────────────────────────

    def race_stints(self) -> Optional[pd.DataFrame]:
        """Fetch tire stint data from OpenF1 for the race."""
        try:
            return self.openf1.stints(self.year, self.race.name, "Race")
        except Exception as exc:
            print(f"  ⚠️  {exc}")
            return None

    def safety_car_laps(self) -> list[int]:
        """Return laps where a safety car was deployed (from OpenF1)."""
        try:
            return self.openf1.safety_car_laps(self.year, self.race.name)
        except Exception:
            return []

    # ── Weekend summary ───────────────────────────────────────────────────────

    def weekend_summary(self) -> dict:
        """
        Compile a summary dict of everything collected so far, ready for
        the race forecaster to consume.

        Keys:
            race_name, race_slug, race_date, is_sprint,
            fp1_pace, fp2_pace, fp2_long_run_pace, fp3_pace,
            quali_grid, sprint_grid (if sprint),
            weather_summary, sessions_available
        """
        self.collect_all_completed_sessions()

        summary: dict = {
            "race_name": self.race.name,
            "race_slug": self.race.slug,
            "race_date": self.race.race_date.isoformat(),
            "is_sprint": self.race.is_sprint,
            "sessions_available": list(self._session_data.keys()),
        }

        for sess in ("FP1", "FP2", "FP3"):
            key = f"{sess.lower()}_pace"
            summary[key] = self.relative_pace_ranking(sess)

        summary["fp2_long_run_pace"] = self.long_run_pace("FP2")
        summary["quali_grid"] = self.qualifying_grid()

        if self.race.is_sprint:
            summary["sprint_grid"] = self.sprint_grid()
            # Sprint Qualifying pace (replaces FP2/FP3 pace on sprint weekends)
            summary["sprint_qualifying_pace"] = self.relative_pace_ranking("Sprint Qualifying")
            # Sprint race result — strong predictor of main race car order
            summary["sprint_result"] = self.sprint_result()

        # ── Weather ── (Open-Meteo: forecast for future, archive for past)
        # Fetch per-session weather so we know if qualifying or the race is wet.
        # weather_enc reflects race-day conditions (the primary forecasting target).
        # qualifying_weather_enc reflects Saturday conditions (affects tire choice).
        try:
            session_weather = self.forecast_weekend_weather()
            summary["session_weather"] = session_weather   # full per-session breakdown

            race_w = session_weather.get("Race", {})
            quali_w = session_weather.get("Qualifying", {})

            race_enc = race_w.get("enc")
            quali_enc = quali_w.get("enc")

            # weather_enc = 1 (wet) if race OR qualifying is wet
            if race_enc == 1 or quali_enc == 1:
                weather_enc = 1
            elif race_enc == 0 and quali_enc == 0:
                weather_enc = 0
            else:
                weather_enc = race_enc  # one of them may be None

            summary["weather_enc"] = weather_enc
            summary["weather_summary"] = {
                "rainfall": weather_enc == 1,
                "race_enc": race_enc,
                "qualifying_enc": quali_enc,
                "race_precip_mm": race_w.get("precip_mm"),
                "race_precip_prob": race_w.get("precip_prob"),
                "qualifying_precip_mm": quali_w.get("precip_mm"),
                "qualifying_precip_prob": quali_w.get("precip_prob"),
            }
        except Exception as exc:
            print(f"  ⚠️  Could not fetch weekend weather forecast: {exc}")
            summary["weather_enc"] = None
            summary["weather_summary"] = {}
            summary["session_weather"] = {}

        return summary

    # ── OpenF1 fallback ───────────────────────────────────────────────────────

    def _collect_via_openf1(self, session_type: str) -> Optional[pd.DataFrame]:
        """
        Fallback: pull lap data directly from OpenF1 if FastF1 is not yet available.
        Returns a simplified DataFrame with the same key columns.
        """
        try:
            df = self.openf1.laps(self.year, self.race.name, session_type)
            if df.empty:
                return None
            # Rename to match FastF1 column conventions
            rename = {
                "driver_number": "DriverNumber",
                "lap_number": "LapNumber",
                "lap_duration": "LapTime",
                "duration_sector_1": "Sector1Time",
                "duration_sector_2": "Sector2Time",
                "duration_sector_3": "Sector3Time",
                "i1_speed": "SpeedI1",
                "i2_speed": "SpeedI2",
                "finish_line_speed": "SpeedFL",
                "st_speed": "SpeedST",
                "date_start": "timestamp",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            df["Year"] = self.year
            df["GrandPrix"] = self.race.name
            df["SessionType"] = session_type
            df["series_id"] = f"{self.year}_{self.race.name}_{session_type}__" + df.get("DriverNumber", "").astype(str)
            return df
        except Exception as exc:
            print(f"     ⚠️  OpenF1 fallback failed: {exc}")
            return None

    def _enrich_with_strategy(self, df: pd.DataFrame, session_type: str) -> pd.DataFrame:
        """Add OpenF1 stint/strategy data to race session laps."""
        try:
            stints = self.openf1.stints(self.year, self.race.name, session_type)
            if stints.empty:
                return df
            # Merge compound info from OpenF1 stints onto FastF1 laps by driver + lap range
            # OpenF1 uses driver_number; FastF1 uses DriverNumber — normalise
            stints["DriverNumber"] = stints.get("driver_number", stints.get("DriverNumber", "")).astype(str)
            df["DriverNumber"] = df["DriverNumber"].astype(str)

            stints_sub = stints[["DriverNumber", "stint_number", "compound", "lap_start", "lap_end", "tyre_age_at_start"]].copy()
            # Merge on driver + lap range
            df = df.merge(
                stints_sub.rename(columns={
                    "compound": "OF1_Compound",
                    "tyre_age_at_start": "TyreAgeAtStintStart",
                }),
                on="DriverNumber",
                how="left",
            )
            # Keep only matching stint (lap within [lap_start, lap_end])
            if "LapNumber" in df.columns and "lap_start" in df.columns:
                mask = (df["LapNumber"] >= df["lap_start"]) & (df["LapNumber"] <= df["lap_end"])
                df = df[mask].drop_duplicates(subset=["series_id", "LapNumber"])
        except Exception:
            pass
        return df
