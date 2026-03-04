"""
Session Pace Series Builder
============================
Tracks how a driver's relative pace evolves through a race weekend
and historically across seasons. This is the "FP1 → FP2 → FP3 → Qualifying"
progression that reveals a driver/car's competitiveness before the race.

Two outputs:
1. Historical session pace series (for TimeCopilot training, 2018–2025)
   unique_id = "VER_FP1_australia" — one data point per year
   ds = session date
   y = relative pace (driver_best / session_best; 1.000 = fastest)

2. Current race weekend pace (for live covariate injection)
   Computed from RaceWeekendCollector and added to circuit series.

Data source: FastF1 historical_collector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from ..collectors.historical_collector import build_historical_dataset, extract_session_timeseries
from ..collectors.calendar_manager import CalendarManager


SESSION_ORDER = ["FP1", "FP2", "FP3", "Q", "SQ", "S", "R"]
SESSION_LABEL = {
    "FP1": "FP1", "FP2": "FP2", "FP3": "FP3",
    "Q": "Qualifying", "SQ": "Sprint Qualifying",
    "S": "Sprint", "R": "Race",
}


def build_session_pace_series(
    years: list[int],
    session_types: list[str] = ("FP1", "FP2", "FP3", "Q"),
    circuit_slugs: Optional[list[str]] = None,
    cache_dir: str = ".fastf1_cache",
) -> pd.DataFrame:
    """
    Build a cross-session pace evolution time series for each driver at each circuit.

    Each row = one driver's best relative pace in one session of one race weekend.
    The progression across sessions (FP1→FP2→FP3→Q) forms a mini time series
    within the race weekend, and the year-over-year series shows development trend.

    Returns TimeCopilot long-format DataFrame:
        unique_id                  | ds                  | y (relative pace)
        VER_australia_FP1          | 2024-03-15 10:30:00 | 1.003
        VER_australia_FP2          | 2024-03-15 14:00:00 | 1.000
        VER_australia_Qualifying   | 2024-03-16 15:00:00 | 1.000
    """
    cal = CalendarManager()
    all_rows = []

    for year in years:
        for session_type in session_types:
            print(f"  → {year} [{session_type}] pace series...")

            try:
                import fastf1
                schedule = fastf1.get_event_schedule(year, include_testing=False)
            except Exception as exc:
                print(f"     ⚠️  Schedule error: {exc}")
                continue

            for _, event in schedule.iterrows():
                gp_name = event["EventName"]
                circuit_slug = _circuit_slug_from_name(gp_name)
                if circuit_slugs and circuit_slug not in circuit_slugs:
                    continue

                df = extract_session_timeseries(year, gp_name, session_type, cache_dir)
                if df is None or df.empty or "LapTime" not in df.columns:
                    continue

                # Compute per-driver best lap relative to session best
                session_rows = _compute_relative_pace(df, year, gp_name, circuit_slug, session_type)
                all_rows.extend(session_rows)

    result = pd.DataFrame(all_rows)
    if result.empty:
        return result

    result["ds"] = pd.to_datetime(result["ds"])
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    print(f"   ✓ {result['unique_id'].nunique()} pace series, {len(result):,} rows")
    return result


def compute_weekend_pace_covariates(
    race_slug: str,
    year: int,
    cache_dir: str = ".fastf1_cache",
) -> pd.DataFrame:
    """
    Compute relative pace covariates for a specific race weekend's completed sessions.
    Returns a DataFrame with one row per driver per session.

    Columns: driver_code, team, session_type, best_lap_s, relative_pace,
             pace_rank, long_run_pace, long_run_rank
    """
    cal = CalendarManager()
    race = cal.get_race(race_slug)
    all_rows = []

    for session_name in ["FP1", "FP2", "FP3", "Q", "SQ"]:
        df = extract_session_timeseries(year, race.name, session_name, cache_dir)
        if df is None or df.empty or "LapTime" not in df.columns:
            continue

        session_rows = _compute_relative_pace(df, year, race.name, race_slug, session_name)
        all_rows.extend(session_rows)

        # FP2 long-run pace
        if session_name == "FP2":
            long_run = _compute_long_run_pace(df, year, race.name, race_slug)
            all_rows.extend(long_run)

    return pd.DataFrame(all_rows)


def build_quali_vs_race_delta(
    years: list[int],
    jolpica_collector=None,
    cache_dir: str = ".fastf1_cache",
) -> pd.DataFrame:
    """
    Build a historical table of qualifying position vs race finish position
    for each driver at each circuit. Used to compute:
      - P_gain = (grid_position - finish_position)  — positive = gained places
      - Correlation between qualifying pace and race pace

    Returns DataFrame with: year, circuit_slug, driver_code, grid_pos,
                             finish_pos, position_gain, quali_pace, race_pace
    """
    if jolpica_collector is None:
        from ..collectors.jolpica_collector import JolpicaCollector
        jolpica_collector = JolpicaCollector()

    all_rows = []

    for year in years:
        print(f"  → {year} qualifying vs race delta...")
        try:
            race_results = jolpica_collector.race_results(year)
            quali_results = jolpica_collector.qualifying_results(year)
        except Exception as exc:
            print(f"     ⚠️  {exc}")
            continue

        for _, race_row in race_results.iterrows():
            circuit_slug = _circuit_slug_from_name(race_row.get("race_name", ""))
            if not circuit_slug:
                continue

            driver_code = str(race_row.get("driver_code", ""))
            finish_pos = race_row.get("finish_position")

            # Find qualifying position
            round_num = int(race_row.get("round", 0))
            driver_quali = quali_results[
                (quali_results["round"] == round_num) &
                (quali_results["driver_code"] == driver_code)
            ]
            grid_pos = driver_quali["grid_position"].iloc[0] if not driver_quali.empty else None

            if grid_pos is None or finish_pos is None:
                continue

            all_rows.append({
                "year": year,
                "circuit_slug": circuit_slug,
                "driver_code": driver_code,
                "constructor": race_row.get("constructor", ""),
                "grid_position": int(grid_pos),
                "finish_position": int(finish_pos) if not np.isnan(float(finish_pos)) else 20,
                "position_gain": int(grid_pos) - (int(finish_pos) if not np.isnan(float(finish_pos)) else 20),
            })

    return pd.DataFrame(all_rows)


# ── Driver ranking for a given session ───────────────────────────────────────

def session_driver_rankings(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a session's lap-level DataFrame, return a driver ranking table.

    Columns: driver_code, team, best_lap_s, relative_pace, pace_rank,
             sector_1_avg, sector_2_avg, sector_3_avg
    """
    if session_df.empty or "LapTime" not in session_df.columns:
        return pd.DataFrame()

    grp_cols = ["Driver", "Team"] if "Team" in session_df.columns else ["Driver"]
    agg = {
        "LapTime": ["min", "mean"],
    }
    if "Sector1Time" in session_df.columns:
        agg["Sector1Time"] = "mean"
    if "Sector2Time" in session_df.columns:
        agg["Sector2Time"] = "mean"
    if "Sector3Time" in session_df.columns:
        agg["Sector3Time"] = "mean"

    summary = session_df[session_df["LapTime"].notna()].groupby(grp_cols).agg(agg)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    rename = {
        "Driver": "driver_code",
        "LapTime_min": "best_lap_s",
        "LapTime_mean": "avg_lap_s",
        "Sector1Time_mean": "sector_1_avg",
        "Sector2Time_mean": "sector_2_avg",
        "Sector3Time_mean": "sector_3_avg",
    }
    summary = summary.rename(columns={k: v for k, v in rename.items() if k in summary.columns})

    session_best = summary["best_lap_s"].min()
    summary["relative_pace"] = summary["best_lap_s"] / session_best
    summary = summary.sort_values("relative_pace").reset_index(drop=True)
    summary["pace_rank"] = range(1, len(summary) + 1)
    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_relative_pace(
    df: pd.DataFrame, year: int, gp_name: str, circuit_slug: str, session_type: str
) -> list[dict]:
    """Compute per-driver best relative pace for a session."""
    rows = []
    accurate = df[df.get("IsAccurate", pd.Series(True, index=df.index))].copy() if "IsAccurate" in df.columns else df.copy()
    accurate = accurate[accurate["LapTime"].notna()]
    if accurate.empty:
        return rows

    session_best = accurate["LapTime"].min()
    if session_best <= 0:
        return rows

    session_date = df["timestamp"].min() if "timestamp" in df.columns else pd.Timestamp(f"{year}-01-01")

    for driver_code, driver_laps in accurate.groupby("Driver"):
        best_lap = driver_laps["LapTime"].min()
        rel_pace = best_lap / session_best
        rows.append({
            "unique_id": f"{driver_code}_{circuit_slug}_{SESSION_LABEL.get(session_type, session_type)}",
            "ds": session_date,
            "y": float(rel_pace),
            "driver_code": str(driver_code),
            "team": str(driver_laps["Team"].iloc[0]) if "Team" in driver_laps.columns else "",
            "circuit_slug": circuit_slug,
            "session_type": session_type,
            "year": year,
            "best_lap_s": float(best_lap),
            "session_best_s": float(session_best),
            "gap_to_best_s": float(best_lap - session_best),
        })
    return rows


def _compute_long_run_pace(
    df: pd.DataFrame, year: int, gp_name: str, circuit_slug: str, min_stint_laps: int = 8
) -> list[dict]:
    """Extract long-run race pace from FP2 stints."""
    rows = []
    if "LapTime" not in df.columns or "Compound" not in df.columns:
        return rows

    accurate = df[df.get("IsAccurate", pd.Series(True, index=df.index))].copy() if "IsAccurate" in df.columns else df.copy()
    accurate = accurate[accurate["LapTime"].notna()]
    if "PitThisLap" in accurate.columns:
        accurate = accurate[accurate["PitThisLap"] == 0]

    # Compute best long run per driver
    best_long_run_by_driver: dict[str, float] = {}
    for driver_code, driver_laps in accurate.groupby("Driver"):
        driver_laps = driver_laps.sort_values("LapNumber")
        driver_laps["stint"] = (driver_laps["Compound"] != driver_laps["Compound"].shift()).cumsum()
        for _, stint_grp in driver_laps.groupby("stint"):
            if len(stint_grp) >= min_stint_laps:
                avg_pace = float(stint_grp["LapTime"].median())
                if str(driver_code) not in best_long_run_by_driver or avg_pace < best_long_run_by_driver[str(driver_code)]:
                    best_long_run_by_driver[str(driver_code)] = avg_pace

    if not best_long_run_by_driver:
        return rows

    field_best = min(best_long_run_by_driver.values())
    session_date = df["timestamp"].min() if "timestamp" in df.columns else pd.Timestamp(f"{year}-01-01")

    for driver_code, long_run_pace in best_long_run_by_driver.items():
        rows.append({
            "unique_id": f"{driver_code}_{circuit_slug}_LongRun",
            "ds": session_date,
            "y": float(long_run_pace / field_best),
            "driver_code": str(driver_code),
            "team": str(df[df["Driver"] == driver_code]["Team"].iloc[0]) if "Team" in df.columns and len(df[df["Driver"] == driver_code]) > 0 else "",
            "circuit_slug": circuit_slug,
            "session_type": "FP2_LongRun",
            "year": year,
            "best_lap_s": float(long_run_pace),
            "session_best_s": float(field_best),
            "gap_to_best_s": float(long_run_pace - field_best),
        })
    return rows


def _circuit_slug_from_name(gp_name: str) -> Optional[str]:
    """Import circuit slug mapper from circuit_series to avoid duplication."""
    from .circuit_series import _circuit_slug
    return _circuit_slug(gp_name)
