"""
F1 Time Series Data Pipeline for TimeCopilot
=============================================
LEGACY SCRIPT — Original pipeline covering 15 races (2022–2024, Race sessions only).

For the full 2026 production pipeline, use the new modular system:

    # Full pre-race pipeline (run Thursday before each race):
    python run_f1_forecast.py --race australia

    # After each session completes:
    python run_f1_forecast.py --race australia --session qualifying

    # Auto-detect current race weekend and run appropriate stage:
    python run_f1_forecast.py --current

    # Check if F1 calendar dates have changed:
    python run_f1_forecast.py --check-calendar

The new pipeline (f1_pipeline/) adds:
  - All sessions FP1/FP2/FP3/Qualifying/Sprint Qualifying/Sprint/Race (2018–2025)
  - Championship points time series (2015–2025) via Jolpica-F1 API
  - Strategy features (tire stints, pit stops, safety cars) via OpenF1 API
  - Circuit-specific performance series (per-driver, per-circuit annual series)
  - Session pace evolution (FP1→FP2→FP3→Qualifying relative pace tracking)
  - 24-race 2026 calendar with 6 sprint weekends, auto-scheduling
  - GitHub Actions automated pipeline (.github/workflows/f1_predictions.yml)
  - Twitter card + LinkedIn post + Plotly charts generation

This legacy script is kept for reference. Its core logic has been
generalised into f1_pipeline/collectors/historical_collector.py.

Install dependencies:
    pip install fastf1 pandas numpy requests timecopilot openai plotly kaleido

Legacy usage (race sessions only):
    python "F1 Timeseries Pipeline.py"
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_DIR = ".fastf1_cache"
OUTPUT_DIR = "output"

# Seasons and races to pull. Extend as needed.
# Each entry: (year, round_number_or_gp_name)
RACES_TO_PULL = [
    (2022, "Bahrain"),
    (2022, "Saudi Arabia"),
    (2022, "Australia"),
    (2022, "Imola"),
    (2022, "Monaco"),
    (2023, "Bahrain"),
    (2023, "Saudi Arabia"),
    (2023, "Australia"),
    (2023, "Imola"),
    (2023, "Monaco"),
    (2024, "Bahrain"),
    (2024, "Saudi Arabia"),
    (2024, "Australia"),
    (2024, "Imola"),
    (2024, "Monaco"),
]

# ── Setup ─────────────────────────────────────────────────────────────────────

Path(CACHE_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_race_timeseries(year: int, gp: str) -> pd.DataFrame | None:
    """
    Load a race session and return a lap-level time series DataFrame.

    Each row = one lap for one driver.
    The 'timestamp' column is wall-clock time of lap completion,
    making it a proper time series sequence.
    """
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load(telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f"  ⚠️  Could not load {year} {gp}: {e}")
        return None

    laps = session.laps.copy()
    if laps.empty:
        return None

    # ── Lap-level features ────────────────────────────────────────────────────
    df = laps[[
        "Driver", "DriverNumber", "Team",
        "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "Compound", "TyreLife", "FreshTyre",
        "PitInTime", "PitOutTime",
        "TrackStatus", "IsAccurate",
        "Time",         # cumulative race time at lap end (Timedelta)
        "LapStartTime", # cumulative race time at lap start
        "Position",
    ]].copy()

    # ── Convert timedeltas to seconds (TimeCopilot needs numeric) ─────────────
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                "PitInTime", "PitOutTime", "Time", "LapStartTime"]:
        df[col] = pd.to_timedelta(df[col]).dt.total_seconds()

    # ── Derive a wall-clock timestamp from session start + race time ──────────
    race_start = session.session_start_time  # tz-aware datetime
    if race_start is not None:
        df["timestamp"] = race_start + pd.to_timedelta(df["Time"], unit="s")
    else:
        # Fallback: use lap number as a synthetic ordinal timestamp
        df["timestamp"] = pd.to_datetime("2000-01-01") + pd.to_timedelta(
            df["Time"].fillna(0), unit="s"
        )

    # ── Gap to leader (proxy for race position trajectory) ────────────────────
    # For each lap, compute cumulative race time per driver
    df = df.sort_values(["LapNumber", "Time"])
    leader_time = df.groupby("LapNumber")["Time"].min().rename("LeaderCumulativeTime")
    df = df.merge(leader_time, on="LapNumber", how="left")
    df["GapToLeader_s"] = df["Time"] - df["LeaderCumulativeTime"]

    # ── Pit stop flag ─────────────────────────────────────────────────────────
    df["PitThisLap"] = df["PitInTime"].notna().astype(int)

    # ── Add race metadata ─────────────────────────────────────────────────────
    df["Year"] = year
    df["GrandPrix"] = session.event["EventName"]
    df["Circuit"] = session.event["Location"]
    df["RaceID"] = f"{year}_{session.event['EventName'].replace(' ', '_')}"

    # ── Final result position (join from results) ─────────────────────────────
    results = session.results[["Abbreviation", "Position", "Points", "Status"]].copy()
    results = results.rename(columns={
        "Abbreviation": "Driver",
        "Position": "FinalPosition",
        "Points": "PointsScored",
        "Status": "FinishStatus",
    })
    df = df.merge(results, on="Driver", how="left")

    # ── Drop helper column ────────────────────────────────────────────────────
    df = df.drop(columns=["LeaderCumulativeTime"])

    # ── Series ID: unique key per driver per race (for TimeCopilot) ───────────
    df["series_id"] = df["RaceID"] + "__" + df["Driver"]

    return df


# ── Weather enrichment ────────────────────────────────────────────────────────

def enrich_with_weather(df: pd.DataFrame, session) -> pd.DataFrame:
    """
    Merge lap-level weather snapshots onto the main DataFrame.
    Weather is sampled at each lap's start time.
    """
    try:
        weather = session.weather_data.copy()
        weather["Time"] = pd.to_timedelta(weather["Time"]).dt.total_seconds()
        weather = weather.sort_values("Time")

        # For each lap, find the closest preceding weather reading
        weather_cols = ["AirTemp", "Humidity", "Pressure", "Rainfall",
                        "TrackTemp", "WindDirection", "WindSpeed"]
        weather_sub = weather[["Time"] + [c for c in weather_cols if c in weather.columns]]

        df = df.sort_values("LapStartTime")
        df = pd.merge_asof(
            df,
            weather_sub.rename(columns={"Time": "LapStartTime"}),
            on="LapStartTime",
            direction="backward",
        )
    except Exception:
        pass  # Weather not available for all races

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_dataset(races: list) -> pd.DataFrame:
    all_laps = []

    for year, gp in races:
        print(f"  → {year} {gp}...")
        session = None

        df = extract_race_timeseries(year, gp)
        if df is None:
            continue

        # Try to enrich with weather
        try:
            session = fastf1.get_session(year, gp, "R")
            # session already loaded above; re-use cache
            session.load(telemetry=False, weather=True, messages=False)
            df = enrich_with_weather(df, session)
        except Exception:
            pass

        all_laps.append(df)
        print(f"     ✓ {len(df)} laps loaded")

    if not all_laps:
        print("No data extracted.")
        return pd.DataFrame()

    combined = pd.concat(all_laps, ignore_index=True)
    return combined


# ── TimeCopilot formatting ────────────────────────────────────────────────────

def to_timecopliot_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape to TimeCopilot's expected long format:

        series_id | timestamp | target | [feature_1] | [feature_2] | ...

    Target options (choose one per forecasting experiment):
      - Position          → predict lap-by-lap position (lower = better)
      - GapToLeader_s     → predict gap to leader in seconds
      - FinalPosition     → static label for classification framing
      - LapTime           → predict next lap time

    The script exports all targets; pick the one that fits your experiment.
    """
    keep_cols = [
        # ── IDs ──────────────────────────────────────────────────────────────
        "series_id", "timestamp",
        # ── Targets ──────────────────────────────────────────────────────────
        "Position",          # lap-by-lap race position  ← primary target
        "GapToLeader_s",     # seconds behind leader
        "LapTime",           # lap time in seconds
        "FinalPosition",     # end-of-race result (static per series)
        # ── Covariates ───────────────────────────────────────────────────────
        "LapNumber",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedFL", "SpeedST",
        "Compound", "TyreLife", "FreshTyre",
        "PitThisLap",
        "TrackStatus",
        "Year", "GrandPrix", "Circuit", "Driver", "Team",
        "PointsScored", "FinishStatus",
        # ── Weather (if available) ────────────────────────────────────────────
        "AirTemp", "TrackTemp", "Rainfall", "WindSpeed", "Humidity",
    ]

    available = [c for c in keep_cols if c in df.columns]
    out = df[available].copy()

    # Encode categorical tyre compound as integer
    compound_map = {
        "SOFT": 1, "MEDIUM": 2, "HARD": 3,
        "INTERMEDIATE": 4, "WET": 5, "UNKNOWN": 0,
    }
    if "Compound" in out.columns:
        out["Compound_enc"] = out["Compound"].map(compound_map).fillna(0).astype(int)

    out = out.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    return out


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🏎️  F1 TimeCopilot Pipeline")
    print("=" * 50)
    print(f"Pulling {len(RACES_TO_PULL)} races...\n")

    raw = build_dataset(RACES_TO_PULL)

    if raw.empty:
        print("Pipeline produced no data. Check FastF1 cache and connectivity.")
    else:
        tc_df = to_timecopliot_format(raw)

        # ── Export ────────────────────────────────────────────────────────────
        out_path = f"{OUTPUT_DIR}/f1_timeseries_timecopliot.csv"
        tc_df.to_csv(out_path, index=False)

        # Also export a per-race summary for quick sanity check
        summary = (
            tc_df.groupby(["Year", "GrandPrix", "Driver"])
            .agg(
                Laps=("LapNumber", "max"),
                AvgLapTime_s=("LapTime", "mean"),
                FinalPosition=("FinalPosition", "first"),
                AvgGapToLeader=("GapToLeader_s", "mean"),
            )
            .reset_index()
        )
        summary_path = f"{OUTPUT_DIR}/f1_race_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\n✅ Done!")
        print(f"   Main dataset : {out_path}  ({len(tc_df):,} rows, {tc_df['series_id'].nunique()} series)")
        print(f"   Summary      : {summary_path}")
        print(f"\n📌 Suggested TimeCopilot target column: 'Position'")
        print(f"   series_id format: '<Year>_<GrandPrix>__<DriverCode>'")
        print(f"   e.g. '2024_Bahrain_Grand_Prix__VER'")


# ── Live race mode (during a race weekend) ────────────────────────────────────
# Uncomment and run during a live session to record real-time data.
#
# from fastf1.livetiming.client import SignalRClient
# client = SignalRClient(filename="live_timing_log.txt")
# client.start()   # records to file; stop manually after session
# # Then load with: fastf1.LiveSessionManager.load("live_timing_log.txt")
