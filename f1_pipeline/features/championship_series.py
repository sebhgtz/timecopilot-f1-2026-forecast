"""
Championship Points Time Series
=================================
Builds the primary TimeCopilot input for championship prediction:

  unique_id  | ds (race date) | y (cumulative points) | covariates...

One row per driver (or constructor) per race, spanning 2015–present.
This gives ~240 data points per active driver across 10 seasons —
enough for TimeCopilot's foundation models to detect trajectory patterns.

Data source: Jolpica-F1 API (cumulative standings per round).

Usage:
    from f1_pipeline.features.championship_series import build_championship_series

    driver_df, constructor_df = build_championship_series(
        years=list(range(2015, 2026)),
        jolpica_collector=JolpicaCollector(),
    )
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from ..collectors.jolpica_collector import JolpicaCollector
from ..collectors.calendar_manager import CalendarManager

# Maps historical Jolpica constructor names → current canonical team name.
# Ensures that renamed teams (e.g. Sauber→Alfa Romeo→Kick Sauber→Audi) share
# one consolidated unique_id in the championship series, so TimeCopilot sees
# the full historical trajectory and the constructor_filter works correctly.
_CONSTRUCTOR_LINEAGE: dict[str, str] = {
    # Audi (née Kick Sauber / Alfa Romeo / Sauber)
    "Sauber": "Audi",
    "Alfa Romeo": "Audi",
    "Alfa Romeo Racing": "Audi",
    "Kick Sauber": "Audi",
    # Racing Bulls (née RB / AlphaTauri / Toro Rosso)
    "Toro Rosso": "Racing Bulls",
    "Scuderia Toro Rosso": "Racing Bulls",
    "AlphaTauri": "Racing Bulls",
    "Alpha Tauri": "Racing Bulls",
    "Scuderia AlphaTauri": "Racing Bulls",
    "RB": "Racing Bulls",
    # Aston Martin (née Racing Point / Force India)
    "Force India": "Aston Martin",
    "Racing Point": "Aston Martin",
    # Alpine (née Renault)
    "Renault": "Alpine",
    "Alpine F1 Team": "Alpine",
    # Red Bull Racing (Jolpica omits "Racing" from short name)
    "Red Bull": "Red Bull Racing",
    # Racing Bulls (Jolpica used "RB F1 Team" in 2024–2025)
    "RB F1 Team": "Racing Bulls",
    # Haas (Jolpica uses full "Haas F1 Team")
    "Haas F1 Team": "Haas",
}


# ── Circuit type encoding (used as covariate) ─────────────────────────────────
# Maps common race name fragments to circuit type integers
# (1=street, 2=power, 3=balanced, 4=technical)
_CIRCUIT_TYPE: dict[str, int] = {
    "bahrain": 3, "saudi": 1, "australia": 1, "imola": 4, "monaco": 1,
    "spain": 4, "barcelona": 4, "azerbaijan": 1, "baku": 1,
    "canada": 1, "montreal": 1, "great britain": 3, "silverstone": 3,
    "austria": 2, "france": 2, "hungary": 4, "belgium": 2, "spa": 2,
    "netherlands": 4, "zandvoort": 4, "italy": 2, "monza": 2,
    "singapore": 1, "japan": 4, "suzuka": 4, "united states": 3,
    "cota": 3, "austin": 3, "mexico": 3, "brazil": 3, "interlagos": 3,
    "las vegas": 1, "qatar": 3, "lusail": 3, "abu dhabi": 1,
    "miami": 1, "jeddah": 1, "shanghai": 3, "china": 3, "madrid": 3,
}


def _circuit_type_enc(race_name: str) -> int:
    name_lower = race_name.lower()
    for key, enc in _CIRCUIT_TYPE.items():
        if key in name_lower:
            return enc
    return 3  # default: balanced


def _dnf_rate(standings_df: pd.DataFrame, driver: str, current_round: int, window: int = 8) -> float:
    """
    DNF rate for a driver over the trailing `window` races.
    Requires a 'status' column in the race results (not always in standings).
    Returns 0.0 if data unavailable.
    """
    # This is a placeholder; the orchestrator can enrich with race results
    return 0.0


# ── Driver championship series ────────────────────────────────────────────────

def build_driver_championship_series(
    years: list[int],
    jolpica: JolpicaCollector,
    active_drivers: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build per-driver cumulative championship points time series.

    Returns TimeCopilot long-format DataFrame:
        unique_id  | ds             | y     | circuit_type | gap_to_leader | recent_form | wins_to_date
        driver_VER | 2024-03-02     | 25    | 1            | 0             | 16.7        | 1
        driver_VER | 2024-03-09     | 43    | 2            | 0             | 22.7        | 1
    """
    all_rows = []

    for year in years:
        print(f"  → Building {year} driver championship series...")
        try:
            df = jolpica.all_driver_standings(year)
        except Exception as exc:
            print(f"     ⚠️  {exc}")
            continue

        if df.empty:
            continue

        # Filter to active drivers if specified
        if active_drivers:
            df = df[df["driver_code"].isin(active_drivers)]

        # Sort by round
        df = df.sort_values(["driver_code", "round"])

        # Get all race names for circuit type encoding
        try:
            race_results = jolpica.race_results(year)
            round_to_race = (
                race_results[["round", "race_name"]]
                .drop_duplicates()
                .set_index("round")["race_name"]
                .to_dict()
            )
        except Exception:
            round_to_race = {}

        # Build covariates per driver
        for driver_code, driver_df in df.groupby("driver_code"):
            driver_df = driver_df.sort_values("round").reset_index(drop=True)

            for i, row in driver_df.iterrows():
                race_name = round_to_race.get(int(row["round"]), "")
                circ_type = _circuit_type_enc(race_name)

                # Gap to championship leader at this round
                round_standings = df[df["round"] == row["round"]]
                leader_pts = round_standings["points"].max() if not round_standings.empty else row["points"]
                gap_to_leader = leader_pts - row["points"]

                # Recent form: avg points per race over last 3 rounds
                prev_pts = driver_df.loc[:i, "points"].values
                if len(prev_pts) >= 2:
                    prev_deltas = np.diff(prev_pts)
                    recent_form = float(np.mean(prev_deltas[-3:]))
                else:
                    recent_form = float(row["points"] / max(row["round"], 1))

                all_rows.append({
                    "unique_id": f"driver_{driver_code}",
                    "ds": row["race_date"],
                    "y": float(row["points"]),
                    "driver_code": driver_code,
                    "driver_name": row.get("driver_name", ""),
                    "constructor": row.get("constructor", ""),
                    "championship_position": int(row.get("position", 0)),
                    "wins_to_date": int(row.get("wins", 0)),
                    "round": int(row["round"]),
                    "year": year,
                    "circuit_type_enc": circ_type,
                    "gap_to_leader_pts": float(gap_to_leader),
                    "recent_form_3race": recent_form,
                })

    result = pd.DataFrame(all_rows)
    if result.empty:
        return result

    result["ds"] = pd.to_datetime(result["ds"])
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return result


# ── Constructor championship series ──────────────────────────────────────────

def build_constructor_championship_series(
    years: list[int],
    jolpica: JolpicaCollector,
    active_constructors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build per-constructor cumulative championship points time series.

    Returns TimeCopilot long-format DataFrame:
        unique_id               | ds          | y     | covariates...
        constructor_Red_Bull    | 2024-03-02  | 44    | ...
    """
    all_rows = []

    for year in years:
        print(f"  → Building {year} constructor championship series...")
        try:
            df = jolpica.all_constructor_standings(year)
        except Exception as exc:
            print(f"     ⚠️  {exc}")
            continue

        if df.empty:
            continue

        if active_constructors:
            df = df[df["constructor"].isin(active_constructors)]

        try:
            race_results = jolpica.race_results(year)
            round_to_race = (
                race_results[["round", "race_name"]]
                .drop_duplicates()
                .set_index("round")["race_name"]
                .to_dict()
            )
        except Exception:
            round_to_race = {}

        df = df.sort_values(["constructor", "round"])

        for constructor, cons_df in df.groupby("constructor"):
            cons_df = cons_df.sort_values("round").reset_index(drop=True)

            # Normalize renamed teams to current canonical name so historical data
            # shares one unique_id (e.g. all Sauber/Alfa/Kick Sauber → Audi).
            canonical = _CONSTRUCTOR_LINEAGE.get(constructor, constructor)

            for i, row in cons_df.iterrows():
                race_name = round_to_race.get(int(row["round"]), "")
                circ_type = _circuit_type_enc(race_name)

                round_standings = df[df["round"] == row["round"]]
                leader_pts = round_standings["points"].max() if not round_standings.empty else row["points"]
                gap_to_leader = leader_pts - row["points"]

                prev_pts = cons_df.loc[:i, "points"].values
                if len(prev_pts) >= 2:
                    prev_deltas = np.diff(prev_pts)
                    recent_form = float(np.mean(prev_deltas[-3:]))
                else:
                    recent_form = float(row["points"] / max(row["round"], 1))

                slug = canonical.lower().replace(" ", "_").replace("-", "_")
                all_rows.append({
                    "unique_id": f"constructor_{slug}",
                    "ds": row["race_date"],
                    "y": float(row["points"]),
                    "constructor": canonical,
                    "championship_position": int(row.get("position", 0)),
                    "wins_to_date": int(row.get("wins", 0)),
                    "round": int(row["round"]),
                    "year": year,
                    "circuit_type_enc": circ_type,
                    "gap_to_leader_pts": float(gap_to_leader),
                    "recent_form_3race": recent_form,
                })

    result = pd.DataFrame(all_rows)
    if result.empty:
        return result

    result["ds"] = pd.to_datetime(result["ds"])
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return result


# ── Combined entry point ──────────────────────────────────────────────────────

def build_championship_series(
    years: list[int],
    jolpica_collector: Optional[JolpicaCollector] = None,
    active_drivers: Optional[list[str]] = None,
    active_constructors: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build both driver and constructor championship time series.

    Returns:
        (driver_df, constructor_df)  — both in TimeCopilot long format
    """
    jc = jolpica_collector or JolpicaCollector()

    print("\n📊 Building championship time series...")
    driver_df = build_driver_championship_series(years, jc, active_drivers)
    constructor_df = build_constructor_championship_series(years, jc, active_constructors)

    print(f"   ✓ Drivers:      {driver_df['unique_id'].nunique()} series, {len(driver_df):,} rows")
    print(f"   ✓ Constructors: {constructor_df['unique_id'].nunique()} series, {len(constructor_df):,} rows")

    return driver_df, constructor_df


# ── Append current-season entry point ────────────────────────────────────────

def append_current_season(
    existing_df: pd.DataFrame,
    year: int,
    jolpica: Optional[JolpicaCollector] = None,
    entity: str = "driver",
) -> pd.DataFrame:
    """
    Append the current season's standings onto an existing historical series.
    Removes any previous entries for the current year first to avoid duplicates.

    entity: "driver" | "constructor"
    """
    jc = jolpica or JolpicaCollector()
    existing_df = existing_df[existing_df["year"] != year].copy()

    if entity == "driver":
        new_df = build_driver_championship_series([year], jc)
    else:
        new_df = build_constructor_championship_series([year], jc)

    return pd.concat([existing_df, new_df], ignore_index=True).sort_values(
        ["unique_id", "ds"]
    ).reset_index(drop=True)
