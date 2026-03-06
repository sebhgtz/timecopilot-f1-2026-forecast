"""
Championship Forecaster
========================
Uses TimeCopilot to predict the Driver and Constructor World Championship
winners for the 2026 F1 season.

Input: Championship points time series (one row per driver per race, 2015–present).
Output: Predicted final standings with confidence intervals + LLM narrative.

Best-performing models for this use case:
  - Prophet: monotonic cumulative growth + annual seasonality
  - N-HiTS / N-BEATS: non-linear form patterns
  - Chronos / TimesFM: zero-shot pattern matching across seasons
  - Ensemble: combines all three for robustness

Usage:
    cf = ChampionshipForecaster(llm="openai:gpt-4o-mini")
    result = cf.forecast_drivers(driver_df, remaining_races=20)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class ChampionshipForecastResult:
    """Holds the output of a championship forecast."""
    entity: str                          # "driver" or "constructor"
    race_name: str                       # Race after which forecast was made
    remaining_races: int
    forecast_df: pd.DataFrame            # TimeCopilot fcst_df
    current_standings: pd.DataFrame      # Current standings snapshot
    predicted_final: pd.DataFrame        # Aggregated predicted final points
    narrative: str = ""                  # LLM explanation
    model_used: str = ""
    timestamp: Optional[str] = None

    def predicted_champion(self) -> dict:
        """Return the predicted champion with probability estimate."""
        if self.predicted_final.empty or "predicted_points" not in self.predicted_final.columns:
            return {}
        top = self.predicted_final.sort_values("predicted_points", ascending=False).iloc[0]
        return {
            "name": top.get("unique_id", "").replace("driver_", "").replace("constructor_", ""),
            "predicted_points": float(top.get("predicted_points", 0)),
            "current_points": float(top.get("current_points", 0)),
        }

    def top_n(self, n: int = 5) -> pd.DataFrame:
        if self.predicted_final.empty or "predicted_points" not in self.predicted_final.columns:
            return self.predicted_final.head(n)
        return self.predicted_final.sort_values("predicted_points", ascending=False).head(n)

    def summary(self) -> str:
        champion = self.predicted_champion()
        lines = [
            f"🏆 {self.entity.title()} Championship Forecast — After {self.race_name}",
            f"   Remaining races: {self.remaining_races}",
            f"   Predicted champion: {champion.get('name', 'Unknown')} "
            f"({champion.get('predicted_points', 0):.0f} pts predicted)",
            "",
            "   Top 5 predicted final standings:",
        ]
        for i, row in self.top_n().iterrows():
            name = str(row.get("unique_id", "")).replace("driver_", "").replace("constructor_", "")
            pts = float(row.get("predicted_points", 0))
            curr = float(row.get("current_points", 0))
            lines.append(f"   {row.get('predicted_position', '?')}. {name}: {pts:.0f} pts (now: {curr:.0f})")
        if self.narrative:
            lines += ["", "   Analysis:", f"   {self.narrative[:500]}"]
        return "\n".join(lines)


class ChampionshipForecaster:
    """
    Wraps TimeCopilot for championship points forecasting.

    Handles both driver and constructor series, computes the horizon
    dynamically from the calendar, and injects the LLM query with
    relevant F1 context.
    """

    def __init__(
        self,
        llm: str = "openai:gpt-4o-mini",
        retries: int = 2,
        post_call_delay_s: float = 5.0,
    ):
        self.llm = llm
        self.retries = retries
        self.post_call_delay_s = post_call_delay_s

    def _get_timecopliot(self):
        from timecopilot import TimeCopilot
        return TimeCopilot(llm=self.llm, retries=self.retries)

    # ── Driver championship ────────────────────────────────────────────────────

    def forecast_drivers(
        self,
        driver_df: pd.DataFrame,
        remaining_races: int,
        race_name: str = "Current Race",
        max_points_per_race: float = 26.0,
        driver_filter: Optional[list[str]] = None,
        year: int = 2026,
    ) -> ChampionshipForecastResult:
        """
        Forecast driver championship final standings.

        Args:
            driver_df:            Championship points time series (from championship_series.py)
            remaining_races:      Number of races left in the season
            race_name:            Name of the most recent race (for report)
            max_points_per_race:  Maximum points available per race (25 + 1 fastest lap)
            driver_filter:        Only forecast these driver codes (e.g. top 8 contenders)
            year:                 Season year for the LLM query context

        Returns:
            ChampionshipForecastResult
        """
        return self._forecast(
            series_df=driver_df,
            entity="driver",
            remaining_races=remaining_races,
            race_name=race_name,
            entity_filter=driver_filter,
            query=self._driver_query(remaining_races, max_points_per_race, year=year),
            year=year,
        )

    def forecast_constructors(
        self,
        constructor_df: pd.DataFrame,
        remaining_races: int,
        race_name: str = "Current Race",
        constructor_filter: Optional[list[str]] = None,
        year: int = 2026,
    ) -> ChampionshipForecastResult:
        """
        Forecast constructor championship final standings.

        Args:
            year: Season year for the LLM query context
        """
        return self._forecast(
            series_df=constructor_df,
            entity="constructor",
            remaining_races=remaining_races,
            race_name=race_name,
            entity_filter=constructor_filter,
            query=self._constructor_query(remaining_races, year=year),
            year=year,
        )

    # ── Core forecasting ──────────────────────────────────────────────────────

    def _forecast(
        self,
        series_df: pd.DataFrame,
        entity: str,
        remaining_races: int,
        race_name: str,
        entity_filter: Optional[list[str]],
        query: str,
        year: int = 2026,
    ) -> ChampionshipForecastResult:
        """Run TimeCopilot on the championship series."""
        # Filter to entity of interest
        df = series_df.copy()
        if entity_filter:
            # Build the exact unique_ids that correspond to each filter entry.
            # Driver unique_ids: championship_series uses f"driver_{driver_code}" where
            # driver_code comes from the Jolpica API in uppercase (e.g. "VER", "HAM").
            # Constructor unique_ids: championship_series lowercases the canonical name
            # slug (e.g. "constructor_red_bull_racing"), so the filter must also lowercase.
            if entity == "driver":
                allowed = {f"driver_{c.upper()}" for c in entity_filter}
            else:
                allowed = {
                    "constructor_" + c.lower().replace(" ", "_").replace("-", "_")
                    for c in entity_filter
                }
            df = df[df["unique_id"].isin(allowed)]

        if df.empty:
            return ChampionshipForecastResult(
                entity=entity, race_name=race_name,
                remaining_races=remaining_races,
                forecast_df=pd.DataFrame(),
                current_standings=pd.DataFrame(),
                predicted_final=pd.DataFrame(),
                narrative="No data available for forecasting.",
            )

        # Prepare TimeCopilot input
        tc_df = df[["unique_id", "ds", "y"]].copy()
        tc_df["ds"] = pd.to_datetime(tc_df["ds"])
        tc_df = tc_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Drop any rows with NaN y values (can occur from 429-dropped rounds)
        tc_df = tc_df.dropna(subset=["y"])

        # TimeCopilot requires h >= 1. Use max(remaining_races, 1) as the
        # actual forecast horizon; the result still stores the real remaining_races
        # (which may be 0 for post-race) for display purposes.
        h = max(remaining_races, 1)
        min_series_len = h + 2
        counts = tc_df.groupby("unique_id")["ds"].count()
        valid_ids = counts[counts >= min_series_len].index
        tc_df = tc_df[tc_df["unique_id"].isin(valid_ids)]

        # Normalize irregular race dates to a regular biweekly grid.
        # TimeCopilot requires a consistent frequency; actual race dates are ~2 weeks apart
        # but not perfectly regular. We preserve ordering and trajectory while making
        # the series compatible with freq="2W".
        tc_df = _normalize_to_biweekly(tc_df)

        # Current standings snapshot — filter to current year only so Round 1
        # doesn't inherit 2025 final standings when no 2026 races have run yet.
        current_year_df = df[pd.to_datetime(df["ds"]).dt.year == year]
        if current_year_df.empty:
            # Season hasn't started — everyone is at 0 pts
            current_standings = df.groupby("unique_id").last()[[]].reset_index()
            current_standings["current_points"] = 0.0
            current_standings["championship_position"] = 0
        else:
            current_standings = (
                current_year_df.groupby("unique_id")
                .agg(current_points=("y", "last"), championship_position=("championship_position", "last"))
                .reset_index()
                .sort_values("current_points", ascending=False)
                .reset_index(drop=True)
            )
        current_standings["current_position"] = range(1, len(current_standings) + 1)

        print(f"\n🔮 Running TimeCopilot championship forecast ({entity})...")
        print(f"   Series: {tc_df['unique_id'].nunique()} | Horizon: {h} races | LLM: {self.llm}")

        forecast_df = pd.DataFrame()
        narrative = ""
        model_used = "unknown"
        tc = None

        try:
            tc = self._get_timecopliot()
            result = tc.forecast(
                df=tc_df,
                freq="2W",
                h=h,
                query=query,
            )
            # Safely extract fcst_df (can't use `or` on DataFrames — truth-value ambiguity)
            raw = getattr(result, "fcst_df", None)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                forecast_df = raw
            # result.output is ForecastAgentOutput (Pydantic model), not a string
            result_output = result.output
            narrative = (
                getattr(result_output, "user_query_response", None)
                or getattr(result_output, "forecast_analysis", "")
                or str(result_output)
            ) or ""
            model_used = getattr(result_output, "selected_model", "unknown")
        except Exception as exc:
            print(f"  ⚠️  TimeCopilot error: {exc}")
            print(f"      Falling back to current-standings projection.")
            narrative = ""
            model_used = "failed"
            # Even on LLM output validation failure, try to retrieve the model's fcst_df
            if tc is not None:
                raw = getattr(tc, "fcst_df", None)
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    forecast_df = raw
        finally:
            if self.post_call_delay_s > 0:
                import time
                print(f"   ⏳ Waiting {self.post_call_delay_s:.0f}s before next LLM call...")
                time.sleep(self.post_call_delay_s)

        # Aggregate predictions to final standings
        predicted_final = self._aggregate_final_standings(
            forecast_df, current_standings, h
        )

        return ChampionshipForecastResult(
            entity=entity,
            race_name=race_name,
            remaining_races=remaining_races,
            forecast_df=forecast_df,
            current_standings=current_standings,
            predicted_final=predicted_final,
            narrative=narrative,
            model_used=model_used,
            timestamp=pd.Timestamp.now().isoformat(),
        )

    def _aggregate_final_standings(
        self,
        forecast_df: pd.DataFrame,
        current_standings: pd.DataFrame,
        remaining_races: int,
    ) -> pd.DataFrame:
        """
        Convert TimeCopilot's forecast into predicted final points.

        TimeCopilot forecasts cumulative points h steps ahead.
        The last forecasted value = predicted final points total.
        """
        # Always build a safe fallback that has predicted_points
        fallback = current_standings.copy()
        if "current_points" in fallback.columns:
            fallback["predicted_points"] = fallback["current_points"]
        else:
            fallback["predicted_points"] = 0.0
        fallback["predicted_position"] = range(1, len(fallback) + 1)

        if forecast_df.empty:
            return fallback

        try:
            # Sort by ds first so .last() reliably picks the latest forecast step
            last_forecast = (
                forecast_df.sort_values("ds")
                .groupby("unique_id", as_index=False)
                .last()
            )

            # Ensure unique_id is a column (reset_index already handles this when
            # as_index=False, but guard against unexpected index structures)
            if "unique_id" not in last_forecast.columns:
                last_forecast = last_forecast.reset_index()

            # Identify the forecast value column — TimeCopilot uses model name(s)
            # Exclude metadata columns; treat the first remaining column as the forecast
            non_meta = [c for c in last_forecast.columns if c not in ("unique_id", "ds", "cutoff")]
            if non_meta:
                last_forecast = last_forecast.rename(columns={non_meta[0]: "predicted_points"})
            else:
                last_forecast["predicted_points"] = 0.0

            result = current_standings.merge(
                last_forecast[["unique_id", "predicted_points"]],
                on="unique_id",
                how="left",
            )

        except Exception as exc:
            print(f"  ⚠️  _aggregate_final_standings error ({exc}) — using current standings")
            return fallback

        # Fill any merge misses with current points (no regression assumed)
        if "current_points" in result.columns:
            result["predicted_points"] = result["predicted_points"].fillna(result["current_points"])
        else:
            result["predicted_points"] = result["predicted_points"].fillna(0.0)

        # Final safety net: ensure column always exists
        if "predicted_points" not in result.columns:
            result["predicted_points"] = result.get("current_points", 0.0)

        result = result.sort_values("predicted_points", ascending=False).reset_index(drop=True)
        result["predicted_position"] = range(1, len(result) + 1)
        return result

    # ── LLM query builders ────────────────────────────────────────────────────

    def _driver_query(self, remaining_races: int, max_pts: float, year: int = 2026) -> str:
        return (
            f"This is the {year} F1 Driver World Championship. "
            f"Each series represents a driver's cumulative points after each race. "
            f"There are {remaining_races} races remaining, with a maximum of {max_pts:.0f} "
            f"points available per race (25 pts for win + 1 fastest lap). "
            f"Consider: (1) current championship gaps, (2) recent form trajectory, "
            f"(3) whether any driver is on a dominant winning streak vs. inconsistent, "
            f"(4) whether the championship leader has enough cushion to clinch. "
            f"Which driver will win the {year} F1 World Championship, and what is the "
            f"likely margin of victory? Identify any realistic challengers."
        )

    def _constructor_query(self, remaining_races: int, year: int = 2026) -> str:
        return (
            f"This is the {year} F1 Constructor World Championship. "
            f"Each series represents a constructor's cumulative points after each race. "
            f"Constructors score points from both their drivers (max 44 per race). "
            f"There are {remaining_races} races remaining. "
            f"Consider: (1) which team has the most consistent double-points finishes, "
            f"(2) recent car development trends, (3) whether any team has shown "
            f"an accelerating or decelerating scoring rate. "
            f"Which constructor will win the {year} World Championship?"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_to_biweekly(tc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map irregular race dates to a regular biweekly grid per series.

    TimeCopilot requires a consistent, named frequency. Championship race dates
    are ~2 weeks apart but not perfectly regular. We preserve the ordering and
    trajectory of each series while making it compatible with freq="2W".

    Each unique_id's series is re-indexed starting from 2015-03-01 with 2-week steps,
    so a series with N data points occupies N consecutive biweekly slots.
    """
    if tc_df.empty:
        return tc_df
    normalized = []
    start = pd.Timestamp("2015-03-01")
    for uid, group in tc_df.groupby("unique_id", sort=False):
        group = group.sort_values("ds").copy().reset_index(drop=True)
        n = len(group)
        group["ds"] = pd.date_range(start=start, periods=n, freq="2W")
        normalized.append(group)
    if not normalized:
        return tc_df
    return pd.concat(normalized, ignore_index=True)
