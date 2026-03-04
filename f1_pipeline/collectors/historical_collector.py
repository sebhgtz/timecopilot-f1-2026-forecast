"""
Historical FastF1 Collector
============================
Extends the original F1 Timeseries Pipeline.py to cover:
  - All seasons 2018–current (not just 2022–2024)
  - All session types: FP1, FP2, FP3, Qualifying, Sprint Qualifying, Sprint, Race
  - Pre-season testing sessions (Barcelona, Bahrain 2026)

Each row = one lap for one driver in one session.
Output is suitable for TimeCopilot's long-format time series.

Key function: extract_session_timeseries(year, gp, session_type)
              — generalises the original extract_race_timeseries().
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE_DIR = ".fastf1_cache"
OUTPUT_DIR = "output"

# Session type string mapping (FastF1 accepts these)
SESSION_ALIASES = {
    "FP1": "FP1",
    "FP2": "FP2",
    "FP3": "FP3",
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Q": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "SQ": "SQ",
    "Sprint": "S",
    "S": "S",
    "Race": "R",
    "R": "R",
}

# Tire compound integer encoding
COMPOUND_MAP = {
    "SOFT": 1, "MEDIUM": 2, "HARD": 3,
    "INTERMEDIATE": 4, "WET": 5,
    "TEST-UNKNOWN": 6, "UNKNOWN": 0,
}


def setup_cache(cache_dir: str = CACHE_DIR) -> None:
    Path(cache_dir).mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_session_timeseries(
    year: int,
    gp: str,
    session_type: str = "R",
    cache_dir: str = CACHE_DIR,
) -> Optional[pd.DataFrame]:
    """
    Load a FastF1 session and return a lap-level time series DataFrame.

    Each row = one lap for one driver.
    Works for any session type: FP1/FP2/FP3/Q/SQ/S/R.

    Position data is NaN in FastF1 for non-race sessions — we derive
    relative pace (lap_time / session_best_lap) instead.
    """
    setup_cache(cache_dir)
    ftype = SESSION_ALIASES.get(session_type, session_type)

    try:
        session = fastf1.get_session(year, gp, ftype)
        session.load(telemetry=False, weather=True, messages=False)
    except Exception as exc:
        print(f"  ⚠️  Could not load {year} {gp} {ftype}: {exc}")
        return None

    laps = session.laps.copy()
    if laps.empty:
        return None

    # ── Core lap columns ──────────────────────────────────────────────────────
    core_cols = [
        "Driver", "DriverNumber", "Team",
        "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Compound", "TyreLife", "FreshTyre",
        "PitInTime", "PitOutTime",
        "TrackStatus", "IsAccurate",
        "Time",          # cumulative session time at lap end
        "LapStartTime",  # cumulative session time at lap start
        "Position",      # NaN for FP/Q sessions
    ]
    available = [c for c in core_cols if c in laps.columns]
    df = laps[available].copy()

    # ── Convert timedeltas to seconds ─────────────────────────────────────────
    td_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
               "PitInTime", "PitOutTime", "Time", "LapStartTime"]
    for col in td_cols:
        if col in df.columns:
            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

    # ── Wall-clock timestamp ──────────────────────────────────────────────────
    race_start = getattr(session, "session_start_time", None)
    if race_start is not None and "Time" in df.columns:
        df["timestamp"] = race_start + pd.to_timedelta(df["Time"].fillna(0), unit="s")
    else:
        df["timestamp"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
            df["Time"].fillna(0) if "Time" in df.columns else 0, unit="s"
        )

    # ── Relative pace (useful for FP/Q sessions where position is NaN) ───────
    if "LapTime" in df.columns:
        session_best = df["LapTime"].min()
        df["RelativePace"] = df["LapTime"] / session_best if session_best > 0 else np.nan

    # ── Gap to leader (race sessions only) ───────────────────────────────────
    if "Position" in df.columns and ftype in ("R", "S"):
        df = df.sort_values(["LapNumber", "Time"])
        if "Time" in df.columns:
            leader_time = df.groupby("LapNumber")["Time"].min().rename("LeaderTime")
            df = df.merge(leader_time, on="LapNumber", how="left")
            df["GapToLeader_s"] = df["Time"] - df["LeaderTime"]
            df = df.drop(columns=["LeaderTime"])
    else:
        df["GapToLeader_s"] = np.nan

    # ── Pit flag ──────────────────────────────────────────────────────────────
    if "PitInTime" in df.columns:
        df["PitThisLap"] = df["PitInTime"].notna().astype(int)

    # ── Metadata ──────────────────────────────────────────────────────────────
    df["Year"] = year
    df["SessionType"] = ftype
    df["GrandPrix"] = session.event["EventName"]
    df["Circuit"] = session.event["Location"]
    df["RaceID"] = f"{year}_{session.event['EventName'].replace(' ', '_')}"
    df["SessionID"] = f"{df['RaceID'].iloc[0]}_{ftype}"

    # ── Final results (race / sprint sessions only) ───────────────────────────
    if ftype in ("R", "S") and hasattr(session, "results") and not session.results.empty:
        results = session.results[["Abbreviation", "Position", "Points", "Status"]].copy()
        results = results.rename(columns={
            "Abbreviation": "Driver",
            "Position": "FinalPosition",
            "Points": "PointsScored",
            "Status": "FinishStatus",
        })
        df = df.merge(results, on="Driver", how="left")
    else:
        df["FinalPosition"] = np.nan
        df["PointsScored"] = np.nan
        df["FinishStatus"] = ""

    # ── Encode tire compound ──────────────────────────────────────────────────
    if "Compound" in df.columns:
        df["Compound_enc"] = df["Compound"].map(COMPOUND_MAP).fillna(0).astype(int)

    # ── TimeCopilot series ID ─────────────────────────────────────────────────
    df["series_id"] = df["SessionID"] + "__" + df["Driver"]

    return df


def enrich_with_weather(df: pd.DataFrame, session) -> pd.DataFrame:
    """
    Merge lap-level weather snapshots onto the main DataFrame.
    Same logic as original pipeline, works for any session type.
    """
    try:
        weather = session.weather_data.copy()
        weather["Time"] = pd.to_timedelta(weather["Time"]).dt.total_seconds()
        weather = weather.sort_values("Time")

        weather_cols = [
            "AirTemp", "Humidity", "Pressure", "Rainfall",
            "TrackTemp", "WindDirection", "WindSpeed",
        ]
        weather_sub = weather[
            ["Time"] + [c for c in weather_cols if c in weather.columns]
        ]

        df = df.sort_values("LapStartTime")
        df = pd.merge_asof(
            df,
            weather_sub.rename(columns={"Time": "LapStartTime"}),
            on="LapStartTime",
            direction="backward",
        )
    except Exception:
        pass
    return df


# ── Bulk historical pull ──────────────────────────────────────────────────────

def build_historical_dataset(
    years: list[int],
    session_types: list[str] = ("FP1", "FP2", "FP3", "Q", "R"),
    gp_filter: Optional[list[str]] = None,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """
    Pull all sessions for the given years and session types.

    Args:
        years:         e.g. [2018, 2019, ..., 2025]
        session_types: subset of FP1/FP2/FP3/Q/SQ/S/R
        gp_filter:     optional list of GP names; if None pulls all races
        cache_dir:     FastF1 cache directory

    Returns:
        Long-format DataFrame ready for TimeCopilot.
    """
    setup_cache(cache_dir)
    all_laps = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            print(f"⚠️  Could not get {year} schedule: {exc}")
            continue

        for _, event in schedule.iterrows():
            gp = event["EventName"]
            if gp_filter and gp not in gp_filter:
                continue

            for stype in session_types:
                print(f"  → {year} {gp} [{stype}]...")
                df = extract_session_timeseries(year, gp, stype, cache_dir)
                if df is None or df.empty:
                    continue

                # Enrich with weather
                try:
                    ftype = SESSION_ALIASES.get(stype, stype)
                    session = fastf1.get_session(year, gp, ftype)
                    # Session already loaded by extract_session_timeseries via cache
                    session.load(telemetry=False, weather=True, messages=False)
                    df = enrich_with_weather(df, session)
                except Exception:
                    pass

                all_laps.append(df)
                print(f"     ✓ {len(df):,} laps, {df['Driver'].nunique()} drivers")

    if not all_laps:
        print("No data extracted.")
        return pd.DataFrame()

    return pd.concat(all_laps, ignore_index=True)


def build_testing_dataset(
    year: int = 2026,
    cache_dir: str = CACHE_DIR,
) -> pd.DataFrame:
    """
    Pull all pre-season testing sessions for a given year.
    FastF1 exposes test sessions via the event schedule (include_testing=True).
    Returns the same long-format DataFrame as build_historical_dataset().
    """
    setup_cache(cache_dir)
    all_laps = []

    try:
        schedule = fastf1.get_event_schedule(year, include_testing=True)
        test_events = schedule[schedule["EventFormat"] == "testing"]
    except Exception as exc:
        print(f"⚠️  Could not get {year} testing schedule: {exc}")
        return pd.DataFrame()

    for _, event in test_events.iterrows():
        gp = event["EventName"]
        # Testing days are usually Practice 1 / 2 / 3 in FastF1
        for stype in ("FP1", "FP2", "FP3"):
            print(f"  → {year} {gp} [{stype}] (testing)...")
            df = extract_session_timeseries(year, gp, stype, cache_dir)
            if df is None or df.empty:
                continue

            df["IsTestSession"] = True
            all_laps.append(df)
            print(f"     ✓ {len(df):,} laps")

    return pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()


# ── Testing pace summary ──────────────────────────────────────────────────────

def get_testing_pace_summary(
    year: int = 2026,
    cache_dir: str = CACHE_DIR,
) -> dict[str, float]:
    """
    Compute per-driver relative pace rank from pre-season testing sessions.

    Returns a dict: {driver_code: avg_relative_pace} where 1.000 = fastest
    in session, higher = slower. Averages across all testing days/sessions.

    Useful for seeding circuit series entries for rookies and new teams that
    have no historical race data but did participate in pre-season testing.
    """
    testing_df = build_testing_dataset(year=year, cache_dir=cache_dir)
    if testing_df.empty:
        return {}

    rows = []
    for session_id, grp in testing_df.groupby("SessionID"):
        if "LapTime" not in grp.columns:
            continue
        valid = grp.dropna(subset=["LapTime"])
        if valid.empty:
            continue
        session_best = valid["LapTime"].min()
        if session_best <= 0:
            continue
        per_driver = (
            valid.groupby("Driver")["LapTime"]
            .min()
            .reset_index()
            .rename(columns={"Driver": "driver_code", "LapTime": "best_lap"})
        )
        per_driver["relative_pace"] = per_driver["best_lap"] / session_best
        rows.append(per_driver[["driver_code", "relative_pace"]])

    if not rows:
        return {}

    combined = pd.concat(rows, ignore_index=True)
    summary = (
        combined.groupby("driver_code")["relative_pace"]
        .mean()
        .sort_values()
        .to_dict()
    )
    return summary


def testing_pace_to_position_estimate(
    testing_summary: dict[str, float],
    driver_code: str,
    n_drivers: int = 22,
    worst_position: float = 19.0,
    best_position: float = 12.0,
) -> float:
    """
    Convert a driver's average relative pace from testing to an estimated
    starting finishing position for synthetic circuit history rows.

    Scale: fastest tester → best_position; slowest → worst_position.
    Default range (12–19) is conservative — new teams start midfield to rear.
    """
    if not testing_summary or driver_code not in testing_summary:
        return (best_position + worst_position) / 2.0

    paces = sorted(testing_summary.values())
    driver_pace = testing_summary[driver_code]

    if len(paces) == 1:
        return (best_position + worst_position) / 2.0

    # Normalise rank 0..1 within the testing field
    min_pace, max_pace = paces[0], paces[-1]
    if max_pace == min_pace:
        return (best_position + worst_position) / 2.0

    normalised = (driver_pace - min_pace) / (max_pace - min_pace)
    return best_position + normalised * (worst_position - best_position)


# ── TimeCopilot formatting ────────────────────────────────────────────────────

def to_timecopliot_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape to TimeCopilot long format:
        series_id | timestamp | target | features...

    Keeps all session types. Use 'RelativePace' as target for FP/Q sessions,
    and 'Position' / 'GapToLeader_s' / 'LapTime' for race sessions.
    """
    keep = [
        "series_id", "timestamp",
        "Position", "GapToLeader_s", "LapTime", "RelativePace",
        "FinalPosition", "PointsScored", "FinishStatus",
        "LapNumber", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedFL", "SpeedST",
        "Compound", "Compound_enc", "TyreLife", "FreshTyre",
        "PitThisLap", "TrackStatus",
        "Year", "SessionType", "GrandPrix", "Circuit",
        "Driver", "Team",
        "AirTemp", "TrackTemp", "Rainfall", "WindSpeed", "Humidity",
    ]
    available = [c for c in keep if c in df.columns]
    out = df[available].copy()
    return out.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
