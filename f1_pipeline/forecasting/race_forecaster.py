"""
Race Winner Forecaster
=======================
Uses TimeCopilot to predict the winner and top finishers of a Grand Prix,
combining:
  1. Historical circuit-specific performance (h=1 on per-circuit annual series)
  2. Current race weekend covariates (FP pace, qualifying grid, weather)

Two-stage approach:
  Stage 1: TimeCopilot on circuit series — historical finishing position trend
  Stage 2: Covariate injection — override with qualifying position + FP pace
            when available (most predictive signals)

Usage:
    rf = RaceForecaster(llm="openai:gpt-4o-mini")
    result = rf.forecast(
        circuit_df=circuit_df,
        circuit_slug="australia",
        year=2026,
        weekend_summary=rw_collector.weekend_summary(),
    )
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RaceForecastResult:
    """Holds the output of a race winner forecast."""
    race_name: str
    circuit_slug: str
    year: int
    session_stage: str                   # "pre_weekend" | "fp1" | "fp2" | "fp3" | "qualifying"
    predicted_top10: pd.DataFrame        # driver_code, predicted_position, probability
    forecast_df: pd.DataFrame            # raw TimeCopilot fcst_df
    narrative: str = ""
    model_used: str = ""
    key_insight: str = ""               # one-liner for social card
    timestamp: Optional[str] = None
    strategy_context: Optional[dict] = None  # wet-boosted context (may differ from caller's copy)

    def predicted_winner(self) -> dict:
        if self.predicted_top10.empty:
            return {}
        winner = self.predicted_top10.sort_values("predicted_position").iloc[0]
        return {
            "driver_code": str(winner.get("driver_code", "")),
            "constructor": str(winner.get("constructor", "")),
            "win_probability": float(winner.get("win_probability", 0.0)),
            "predicted_position": float(winner.get("predicted_position", 1.0)),
        }

    def predicted_podium(self) -> list[dict]:
        top3 = self.predicted_top10.sort_values("predicted_position").head(3)
        return [
            {
                "position": i + 1,
                "driver_code": str(row.get("driver_code", "")),
                "constructor": str(row.get("constructor", "")),
                "probability": float(row.get("win_probability", 0.0)),
            }
            for i, (_, row) in enumerate(top3.iterrows())
        ]

    def summary(self) -> str:
        winner = self.predicted_winner()
        podium = self.predicted_podium()
        lines = [
            f"🏎️ {self.race_name} {self.year} — Race Prediction ({self.session_stage.upper()})",
            f"   Predicted winner: {winner.get('driver_code', '?')} "
            f"({winner.get('constructor', '')}) — {winner.get('win_probability', 0)*100:.0f}%",
            "",
            "   Predicted podium:",
        ]
        for p in podium:
            lines.append(
                f"   P{p['position']}: {p['driver_code']} ({p['constructor']}) "
                f"— {p['probability']*100:.0f}%"
            )
        if self.key_insight:
            lines += ["", f"   Key insight: {self.key_insight}"]
        return "\n".join(lines)


class RaceForecaster:
    """
    Forecasts race winner using TimeCopilot on circuit-specific time series
    enriched with current race weekend covariates.
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

    # ── Main forecast ─────────────────────────────────────────────────────────

    def forecast(
        self,
        circuit_df: pd.DataFrame,
        circuit_slug: str,
        year: int = 2026,
        weekend_summary: Optional[dict] = None,
        strategy_context: Optional[dict] = None,
        session_stage: str = "pre_weekend",
        top_n_drivers: int = 20,
    ) -> RaceForecastResult:
        """
        Predict the race winner for a given circuit.

        Args:
            circuit_df:        Circuit series from circuit_series.py (historical + placeholder row)
            circuit_slug:      e.g. "australia"
            year:              e.g. 2026
            weekend_summary:   Dict from RaceWeekendCollector.weekend_summary()
            strategy_context:  Dict from strategy_features.get_circuit_strategy()
            session_stage:     Stage of prediction ("pre_weekend", "fp1", ..., "qualifying")
            top_n_drivers:     Focus on top N drivers by historical pace

        Returns:
            RaceForecastResult
        """
        # Filter to this circuit only
        circ_df = circuit_df[circuit_df["circuit_slug"] == circuit_slug].copy()
        if circ_df.empty:
            return self._empty_result(circuit_slug, year, session_stage)

        # Build covariates dict from weekend summary
        covariates = self._extract_covariates(weekend_summary, circuit_slug, year)

        # If wet conditions are forecast, compute wet driver stats from circuit history.
        # Store both the formatted string (for LLM query) and the raw DataFrame
        # (for numeric position adjustment applied after _build_predicted_standings).
        if covariates.get("weather_enc") == 1:
            try:
                from f1_pipeline.collectors.weather_log import (
                    wet_driver_stats, format_wet_stats_for_query,
                )
                stats_df = wet_driver_stats(circ_df)
                covariates["wet_stats_query"] = format_wet_stats_for_query(stats_df)
                covariates["wet_stats_df"] = stats_df
            except Exception:
                covariates["wet_stats_query"] = ""
                covariates["wet_stats_df"] = None
        else:
            covariates["wet_stats_query"] = ""
            covariates["wet_stats_df"] = None

        # Boost SC/red-flag probability when wet conditions are forecast.
        # Historical SC frequency is calculated from 2023-2024 dry races and severely
        # underestimates incident risk on wet tracks (e.g. Belgium 2025 = red-flagged).
        # Wet conditions → minimum 60% SC probability, 45% red-flag probability.
        if strategy_context and covariates.get("weather_enc") == 1:
            strategy_context = dict(strategy_context)  # copy — don't mutate caller's dict
            strategy_context["sc_probability"] = min(
                max(strategy_context.get("sc_probability", 0) * 2.5, 0.60), 1.0
            )
            strategy_context["vsc_probability"] = min(
                max(strategy_context.get("vsc_probability", 0) * 2.0, 0.50), 1.0
            )
            strategy_context["red_flag_probability"] = min(
                max(strategy_context.get("red_flag_probability", 0) * 3.0, 0.45), 1.0
            )

        # Inject current-year covariates into circuit series placeholder row
        circ_df = self._inject_covariates(circ_df, year, covariates)

        # Prophet's cross-validation needs training_size >= 2 per window.
        # With freq="YS" and h=1, test_size=1 so we need at least h+2=3 data points.
        min_len = 1 + 2  # h + 2 (1 for test, 2 for minimum Prophet training)
        counts = circ_df.groupby("unique_id")["ds"].count()
        valid = counts[counts >= min_len].index
        circ_df = circ_df[circ_df["unique_id"].isin(valid)]

        # Build TimeCopilot input — only unique_id, ds, y (no exogenous columns;
        # TimeCopilot does not yet support cross-validation with exogenous variables).
        # Covariate context (grid position, weather, strategy) is injected into the query.
        tc_df = circ_df[["unique_id", "ds", "y"]].copy()
        tc_df["ds"] = pd.to_datetime(tc_df["ds"])
        tc_df = tc_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        # Normalize to regular annual dates (YS = year-start) so TimeCopilot's
        # cross-validation aligns correctly. Race dates vary by a few weeks each
        # year which confuses freq="Y" validation. We preserve ordering & trajectory.
        tc_df = _normalize_to_annual(tc_df)

        race_name = f"{year} {circuit_slug.replace('_', ' ').title()} Grand Prix"
        query = self._build_query(
            circuit_slug, year, covariates, strategy_context, session_stage
        )

        print(f"\n🏁 Running TimeCopilot race forecast — {race_name} ({session_stage})...")
        print(f"   Drivers: {tc_df['unique_id'].nunique()} | Stage: {session_stage}")

        forecast_df = pd.DataFrame()
        narrative = ""
        model_used = "unknown"
        tc = None

        try:
            tc = self._get_timecopliot()
            result = tc.forecast(
                df=tc_df,
                freq="YS",
                h=1,
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
            print(f"      Falling back to statistical ranking (historical avg).")
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

        predicted_top10 = self._build_predicted_standings(
            forecast_df, circ_df, covariates, year, session_stage
        )

        # Numeric wet weather adjustment: directly shift predicted positions based on
        # each driver's historical wet vs dry advantage. This works even when the LLM
        # narrative fails (rate limit, etc.) — the positions themselves reflect wet pace.
        wet_stats_df = covariates.get("wet_stats_df")
        if (
            covariates.get("weather_enc") == 1
            and wet_stats_df is not None
            and not wet_stats_df.empty
            and not predicted_top10.empty
        ):
            advantage_map = dict(zip(wet_stats_df["driver_code"], wet_stats_df["advantage"]))
            adj = predicted_top10["driver_code"].map(advantage_map).fillna(0)
            predicted_top10["predicted_position"] = (
                predicted_top10["predicted_position"] - adj
            ).clip(lower=1, upper=20)
            predicted_top10 = predicted_top10.sort_values(
                "predicted_position"
            ).reset_index(drop=True)

        key_insight = self._extract_key_insight(
            predicted_top10, covariates, session_stage, strategy_context
        )

        return RaceForecastResult(
            race_name=race_name,
            circuit_slug=circuit_slug,
            year=year,
            session_stage=session_stage,
            predicted_top10=predicted_top10,
            forecast_df=forecast_df,
            narrative=narrative,
            model_used=model_used,
            key_insight=key_insight,
            timestamp=pd.Timestamp.now().isoformat(),
            strategy_context=strategy_context,  # may be wet-boosted copy
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_covariates(
        self, weekend_summary: Optional[dict], circuit_slug: str, year: int
    ) -> dict:
        """Extract relevant covariates from the weekend summary dict."""
        covariates: dict = {
            "circuit_slug": circuit_slug,
            "year": year,
            "is_sprint": False,
            "grid_positions": {},            # driver_code → main quali grid_position
            "fp1_pace_ranks": {},            # driver_code → pace_rank
            "fp2_pace_ranks": {},
            "fp2_long_run_ranks": {},
            "fp3_pace_ranks": {},
            "sprint_qualifying_ranks": {},   # sprint weekend: SQ pace ranks
            "sprint_results": {},            # sprint weekend: {driver → finish pos}
            "weather_desc": "dry",
            "weather_enc": 0,
        }
        if weekend_summary is None:
            return covariates

        covariates["is_sprint"] = bool(weekend_summary.get("is_sprint", False))

        # Qualifying grid (main race — always the most important signal)
        quali_grid = weekend_summary.get("quali_grid")
        if quali_grid is not None and not quali_grid.empty:
            covariates["grid_positions"] = dict(
                zip(quali_grid["driver_code"], quali_grid["grid_position"])
            )

        # FP pace ranks (standard weekends)
        for fp in ("fp1", "fp2", "fp3"):
            pace = weekend_summary.get(f"{fp}_pace")
            if pace is not None and not pace.empty and "driver_code" in pace.columns:
                covariates[f"{fp}_pace_ranks"] = dict(
                    zip(pace["driver_code"], pace["pace_rank"])
                )

        # Long run pace (race pace proxy from FP2; not available on sprint weekends)
        lr = weekend_summary.get("fp2_long_run_pace")
        if lr is not None and not lr.empty and "driver_code" in lr.columns:
            covariates["fp2_long_run_ranks"] = dict(
                zip(lr["driver_code"], lr.get("race_pace_rank", lr.get("pace_rank", [])))
            )

        # Sprint weekend: Sprint Qualifying pace (replaces FP2/FP3 as the key pace signal)
        sq_pace = weekend_summary.get("sprint_qualifying_pace")
        if sq_pace is not None and not sq_pace.empty and "driver_code" in sq_pace.columns:
            covariates["sprint_qualifying_ranks"] = dict(
                zip(sq_pace["driver_code"], sq_pace["pace_rank"])
            )

        # Sprint weekend: Sprint race result (direct evidence of car pace on this track)
        sprint_res = weekend_summary.get("sprint_result")
        if sprint_res is not None and not sprint_res.empty and "driver_code" in sprint_res.columns:
            covariates["sprint_results"] = dict(
                zip(
                    sprint_res["driver_code"],
                    sprint_res["sprint_finish_position"],
                )
            )

        # Weather — prefer top-level weather_enc from Open-Meteo forecast (set by
        # RaceWeekendCollector.forecast_weekend_weather); fall back to OpenF1 rainfall flag.
        weather_enc = weekend_summary.get("weather_enc")
        if weather_enc is None:
            # Legacy fallback: OpenF1 race-day rainfall
            weather_summary = weekend_summary.get("weather_summary", {})
            if isinstance(weather_summary, dict) and weather_summary.get("rainfall"):
                weather_enc = 1
            else:
                weather_enc = 0

        covariates["weather_enc"] = int(weather_enc) if weather_enc is not None else 0
        covariates["weather_desc"] = "wet" if covariates["weather_enc"] == 1 else "dry"

        # Detailed per-session weather (for richer LLM context)
        session_weather = weekend_summary.get("session_weather", {})
        covariates["session_weather"] = session_weather

        return covariates

    def _inject_covariates(
        self, circ_df: pd.DataFrame, year: int, covariates: dict
    ) -> pd.DataFrame:
        """
        Update the placeholder (NaN) row for the current year with
        actual covariates (grid position, FP pace ranks, weather).
        """
        circ_df = circ_df.copy()
        current_year_mask = circ_df["year"] == year

        grid_positions = covariates.get("grid_positions", {})
        fp1_ranks = covariates.get("fp1_pace_ranks", {})
        fp2_ranks = covariates.get("fp2_pace_ranks", {})
        fp2_lr_ranks = covariates.get("fp2_long_run_ranks", {})
        weather_enc = covariates.get("weather_enc", 0)

        for idx in circ_df[current_year_mask].index:
            driver_code = circ_df.at[idx, "driver_code"]
            if grid_positions.get(driver_code):
                circ_df.at[idx, "grid_position"] = float(grid_positions[driver_code])
            if weather_enc:
                circ_df.at[idx, "weather_enc"] = float(weather_enc)
            # Add FP pace as extra covariates if columns exist
            if "fp1_pace_rank" in circ_df.columns and fp1_ranks.get(driver_code):
                circ_df.at[idx, "fp1_pace_rank"] = float(fp1_ranks[driver_code])
            if "fp2_pace_rank" in circ_df.columns and fp2_ranks.get(driver_code):
                circ_df.at[idx, "fp2_pace_rank"] = float(fp2_ranks[driver_code])

        return circ_df

    def _build_predicted_standings(
        self,
        forecast_df: pd.DataFrame,
        circ_df: pd.DataFrame,
        covariates: dict,
        year: int,
        session_stage: str = "",
    ) -> pd.DataFrame:
        """
        Convert TimeCopilot's 1-step-ahead forecasts to predicted finishing positions.
        Lower forecasted value = better position.
        """
        # Get driver metadata from circuit series.
        # Prefer the current-year placeholder row (which reflects current team
        # assignments from current_team_map) over historical rows.
        driver_meta = (
            circ_df.groupby("unique_id")
            .last()
            .reset_index()[["unique_id", "driver_code", "constructor"]]
        )
        # Override constructor with current 2026 team assignments so historical
        # team names (e.g. PER as Red Bull) don't leak into the race prediction.
        from ..collectors.calendar_manager import DRIVERS_2026 as _CURRENT_TEAMS
        driver_meta["constructor"] = (
            driver_meta["driver_code"].map(_CURRENT_TEAMS)
            .fillna(driver_meta["constructor"])
        )

        if not forecast_df.empty:
            # Extract predicted finishing position from forecast
            fc_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds", "cutoff")]
            if fc_cols:
                last_fc = forecast_df.groupby("unique_id").last().reset_index()
                last_fc["predicted_position"] = last_fc[fc_cols[0]].clip(lower=1, upper=20).fillna(10.0)
            else:
                last_fc = pd.DataFrame({"unique_id": circ_df["unique_id"].unique()})
                last_fc["predicted_position"] = 5.0
        else:
            # Fallback: rank by historical avg finish at this circuit
            hist = circ_df[circ_df["year"] < year].copy()
            hist_avg = (
                hist.groupby("unique_id")["y"]
                .mean()
                .reset_index()
                .rename(columns={"y": "predicted_position"})
            )
            last_fc = hist_avg

        result = last_fc.merge(driver_meta, on="unique_id", how="left")

        # Blend with available positional signals (in priority order):
        # 1. Qualifying grid position (most predictive — ~65% of race result variance)
        # 2. Sprint result (strong race-pace signal on sprint weekends, before qualifying)
        # 3. Sprint qualifying rank (weak positional signal, FP2 equivalent for pace)
        grid_positions = covariates.get("grid_positions", {})
        sprint_results = covariates.get("sprint_results", {})
        sprint_quali_ranks = covariates.get("sprint_qualifying_ranks", {})

        if grid_positions:
            result["grid_position"] = result["driver_code"].map(grid_positions)
            # At qualifying stage, grid position is the dominant signal (~65% of race variance).
            # Earlier stages rely more on historical trend.
            base_grid_weight = 0.65 if session_stage == "qualifying" else 0.30
            hist_weight = 1.0 - base_grid_weight
            result["predicted_position"] = (
                result["predicted_position"].fillna(10) * hist_weight +
                result["grid_position"].fillna(result["predicted_position"].fillna(10)) * base_grid_weight
            )
            # For qualifying: drivers predicted far back (>12) despite a good grid position
            # are likely rookies with only synthetic circuit history. Trust the grid position more.
            if session_stage == "qualifying":
                grid_val = result["driver_code"].map(grid_positions)
                back_of_grid = result["predicted_position"] > 12
                has_good_grid = grid_val < 8  # qualified in top 7
                correct_rookies = back_of_grid & has_good_grid
                if correct_rookies.any():
                    result.loc[correct_rookies, "predicted_position"] = (
                        result.loc[correct_rookies, "predicted_position"] * 0.15 +
                        grid_val[correct_rookies] * 0.85
                    )
        elif sprint_results:
            # No qualifying yet, but sprint result available — use as strong positional signal
            result["sprint_position"] = result["driver_code"].map(sprint_results)
            # Blend: 60% TimeCopilot history, 40% sprint result (higher weight than SQ)
            result["predicted_position"] = (
                result["predicted_position"].fillna(10) * 0.60 +
                result["sprint_position"].fillna(result["predicted_position"].fillna(10)) * 0.40
            )
        elif sprint_quali_ranks:
            # Sprint qualifying pace only — treat like FP2 (lighter blend)
            result["sq_rank"] = result["driver_code"].map(sprint_quali_ranks)
            result["predicted_position"] = (
                result["predicted_position"].fillna(10) * 0.80 +
                result["sq_rank"].fillna(result["predicted_position"].fillna(10)) * 0.20
            )
        else:
            # No qualifying/sprint data yet — blend FP pace as a light signal.
            # Prefer FP2 (more representative of race pace) over FP1.
            fp2_ranks = covariates.get("fp2_pace_ranks", {})
            fp1_ranks = covariates.get("fp1_pace_ranks", {})
            fp_ranks = fp2_ranks or fp1_ranks
            if fp_ranks:
                result["fp_pace_rank"] = result["driver_code"].map(fp_ranks)
                result["predicted_position"] = (
                    result["predicted_position"].fillna(10) * 0.80 +
                    result["fp_pace_rank"].fillna(result["predicted_position"].fillna(10)) * 0.20
                )

        result = result.sort_values("predicted_position").reset_index(drop=True)
        result["predicted_rank"] = range(1, len(result) + 1)

        # Compute win probability using steeper inverse-square curve.
        # 1/pos² amplifies differences: P1 is 25× more likely than P5 (vs 5× with linear).
        pos = result["predicted_position"].fillna(10.0).replace(0, 0.1)
        scores = 1.0 / pos ** 2
        result["win_probability"] = scores / scores.sum()

        return result.head(10)

    def _empty_result(self, circuit_slug: str, year: int, session_stage: str) -> RaceForecastResult:
        return RaceForecastResult(
            race_name=f"{year} {circuit_slug} Grand Prix",
            circuit_slug=circuit_slug,
            year=year,
            session_stage=session_stage,
            predicted_top10=pd.DataFrame(),
            forecast_df=pd.DataFrame(),
            narrative="No historical data available for this circuit.",
            key_insight="Insufficient data.",
        )

    def _build_query(
        self,
        circuit_slug: str,
        year: int,
        covariates: dict,
        strategy_context: Optional[dict],
        session_stage: str,
    ) -> str:
        is_sprint = covariates.get("is_sprint", False)

        grid_info = ""
        if covariates.get("grid_positions"):
            top3_grid = sorted(covariates["grid_positions"].items(), key=lambda x: x[1])[:3]
            grid_info = f" Qualifying: {', '.join(f'{d} (P{p:.0f})' for d, p in top3_grid)}."

        weather_enc = covariates.get("weather_enc", 0)
        weather_desc = covariates.get("weather_desc", "dry")
        wet_stats = covariates.get("wet_stats_query", "")

        # Build weather context — include per-session breakdown if available
        session_weather = covariates.get("session_weather", {})
        quali_enc = (session_weather.get("Qualifying") or {}).get("enc")
        race_enc = (session_weather.get("Race") or {}).get("enc")
        if quali_enc is not None and race_enc is not None and quali_enc != race_enc:
            q_str = "wet" if quali_enc == 1 else "dry"
            r_str = "wet" if race_enc == 1 else "dry"
            weather_info = f" Track conditions: qualifying={q_str}, race={r_str}."
        else:
            weather_info = f" Track conditions: {weather_desc}."

        if wet_stats:
            weather_info += f" {wet_stats}"

        # For standard weekends: FP2 long-run pace. For sprint weekends: sprint result.
        fp_info = ""
        sprint_info = ""
        if is_sprint:
            if covariates.get("sprint_results"):
                top3_sr = sorted(covariates["sprint_results"].items(), key=lambda x: x[1])[:3]
                sprint_info = (
                    f" Sprint race top 3: {', '.join(f'{d} (P{int(p)})' for d, p in top3_sr)}."
                    f" Sprint results indicate car/driver pace on race setup at this circuit."
                )
            elif covariates.get("sprint_qualifying_ranks"):
                top3_sq = sorted(covariates["sprint_qualifying_ranks"].items(), key=lambda x: x[1])[:3]
                sprint_info = (
                    f" Sprint Qualifying pace leaders: {', '.join(f'{d} (rank {int(r)})' for d, r in top3_sq)}."
                )
        else:
            if covariates.get("fp2_long_run_ranks"):
                top3_lr = sorted(covariates["fp2_long_run_ranks"].items(), key=lambda x: x[1])[:3]
                fp_info = f" FP2 long-run leaders: {', '.join(f'{d} (rank {r:.0f})' for d, r in top3_lr)}."

        strategy_info = ""
        if strategy_context:
            modal = strategy_context.get("modal_stops", 2)
            sc_prob = strategy_context.get("sc_probability", 0)
            red_flag_prob = strategy_context.get("red_flag_probability", 0)
            vsc_prob = strategy_context.get("vsc_probability", 0)
            incident_detail = f"Safety car probability: {sc_prob*100:.0f}%"
            if red_flag_prob >= 0.30:
                incident_detail += f", red flag probability: {red_flag_prob*100:.0f}%"
            if vsc_prob >= 0.40:
                incident_detail += f", VSC probability: {vsc_prob*100:.0f}%"
            strategy_info = (
                f" Historical strategy: {modal:.0f}-stop typically used here. "
                f"{incident_detail}."
            )

        sprint_context = " This is a sprint weekend." if is_sprint else ""

        return (
            f"This is historical finishing position data for Formula 1 drivers at the "
            f"{circuit_slug.replace('_', ' ').title()} circuit (lower = better result; "
            f"20 = DNF).{sprint_context} Predict each driver's likely finishing position in the "
            f"{year} race (h=1).{grid_info}{sprint_info}{weather_info}{fp_info}{strategy_info} "
            f"Consider: (1) driver's historical trend at this circuit, "
            f"(2) qualifying grid position as the strongest predictor, "
            f"{'(3) sprint race result as direct evidence of car race pace, ' if is_sprint else ''}"
            f"(3) car performance trajectory this season, "
            f"(4) likelihood of incidents (street circuits have higher DNF rates). "
            f"Who will win, and who is the best podium bet?"
        )

    def _extract_key_insight(
        self,
        predicted_top10: pd.DataFrame,
        covariates: dict,
        session_stage: str,
        strategy_context: Optional[dict],
    ) -> str:
        """Generate a one-liner key insight for the social post."""
        if predicted_top10.empty:
            return "Insufficient data for insight."

        winner = predicted_top10.iloc[0]
        driver = str(winner.get("driver_code", "?"))
        win_pct = float(winner.get("win_probability", 0)) * 100

        grid_positions = covariates.get("grid_positions", {})
        sprint_results = covariates.get("sprint_results", {})
        grid_pos = grid_positions.get(driver)

        if session_stage == "qualifying" and grid_pos:
            grid_pos = int(grid_pos)
            if grid_pos == 1:
                return f"{driver} starts from pole — strong favourite at {win_pct:.0f}% win probability."
            elif grid_pos <= 3:
                return f"{driver} (P{grid_pos} on grid) leads predictions at {win_pct:.0f}% win probability."
            else:
                return f"{driver} starts P{grid_pos} but historical pace at this circuit suggests victory is likely."

        if session_stage == "sprint" and sprint_results.get(driver):
            sprint_pos = int(sprint_results[driver])
            if sprint_pos <= 3:
                return f"{driver} finished P{sprint_pos} in the sprint — leads race prediction at {win_pct:.0f}%."
            else:
                return f"{driver} leads race prediction at {win_pct:.0f}% despite sprint result of P{sprint_pos}."

        sc_prob = (strategy_context or {}).get("sc_probability", 0)
        if sc_prob > 0.6:
            return f"High safety car probability ({sc_prob*100:.0f}%) makes strategy crucial — {driver} leads predictions."

        return f"{driver} predicted winner at {win_pct:.0f}% based on historical circuit performance."


# ── Module-level helpers ───────────────────────────────────────────────────────

def _normalize_to_annual(tc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map irregular per-circuit race dates to a regular year-start (YS) grid.

    Circuit series have one data point per year per driver, but the actual
    race date varies by a few weeks each year (e.g. Australian GP can be
    March 15 or March 24). freq="YS" requires dates to be exactly Jan 1.
    We re-index each series to consecutive YS dates so TimeCopilot's
    cross-validation aligns correctly without gaps or misaligned steps.
    """
    if tc_df.empty:
        return tc_df
    normalized = []
    start = pd.Timestamp("2018-01-01")
    for uid, group in tc_df.groupby("unique_id", sort=False):
        group = group.sort_values("ds").copy().reset_index(drop=True)
        n = len(group)
        group["ds"] = pd.date_range(start=start, periods=n, freq="YS")
        normalized.append(group)
    if not normalized:
        return tc_df
    return pd.concat(normalized, ignore_index=True)
