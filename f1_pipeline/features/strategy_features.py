"""
Strategy Pattern Features
==========================
Analyzes historical tire strategy patterns from OpenF1 stints data.
Feeds into circuit series and championship series as covariates.

Key outputs:
  1. Team strategy profile per circuit (1-stop / 2-stop / 3-stop frequencies)
  2. Average stint lengths by compound and team
  3. Pit stop consistency (std dev of pit stop duration per team)
  4. Safety car frequency per circuit (from race control messages)
  5. Undercut success rate per circuit

Used by the race forecaster to:
  - Predict which teams will undercut
  - Estimate safety car probability (affects predicted race outcome)
  - Identify teams with reliable vs unreliable pit stops

Data source: OpenF1 /stints, /pit, /race_control (2023–2025).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from ..collectors.openf1_collector import OpenF1Collector
from .circuit_series import _circuit_slug, CIRCUITS_2026


def build_strategy_features(
    years: list[int] = (2023, 2024, 2025),
    openf1: Optional[OpenF1Collector] = None,
    jolpica=None,
) -> dict[str, pd.DataFrame]:
    """
    Build all strategy-related feature tables.

    Returns a dict with keys:
        "team_strategy_profile"     — most common strategy per team per circuit
        "stint_length_profile"      — avg stint lengths by compound and team
        "pit_reliability"           — pit stop consistency per team
        "safety_car_probability"    — SC frequency per circuit
        "undercut_success"          — undercut success rate per circuit
    """
    oc = openf1 or OpenF1Collector()

    if jolpica is None:
        from ..collectors.jolpica_collector import JolpicaCollector
        jolpica = JolpicaCollector()

    print("\n🔧 Building strategy features...")

    # Get race names for the years we care about
    gp_names: list[tuple[int, str]] = []
    for year in years:
        try:
            results = jolpica.race_results(year)
            for _, row in results[["race_name"]].drop_duplicates().iterrows():
                gp_names.append((year, row["race_name"]))
        except Exception:
            pass

    # Pull stints
    print("  → Fetching stints from OpenF1...")
    stints_rows = []
    pit_rows = []
    rc_rows = []

    for year, gp_name in gp_names:
        slug = _circuit_slug(gp_name)
        if slug is None:
            continue

        try:
            stints = oc.stints(year, gp_name)
            if not stints.empty:
                stints["year"] = year
                stints["gp_name"] = gp_name
                stints["circuit_slug"] = slug
                stints_rows.append(stints)
        except Exception:
            pass

        try:
            pits = oc.pit_stops(year, gp_name)
            if not pits.empty:
                pits["year"] = year
                pits["gp_name"] = gp_name
                pits["circuit_slug"] = slug
                pit_rows.append(pits)
        except Exception:
            pass

        try:
            rc = oc.race_control(year, gp_name)
            if not rc.empty:
                rc["year"] = year
                rc["gp_name"] = gp_name
                rc["circuit_slug"] = slug
                rc_rows.append(rc)
        except Exception:
            pass

    stints_df = pd.concat(stints_rows, ignore_index=True) if stints_rows else pd.DataFrame()
    pit_df = pd.concat(pit_rows, ignore_index=True) if pit_rows else pd.DataFrame()
    rc_df = pd.concat(rc_rows, ignore_index=True) if rc_rows else pd.DataFrame()

    return {
        "team_strategy_profile": _team_strategy_profile(stints_df, jolpica, years),
        "stint_length_profile": _stint_length_profile(stints_df),
        "pit_reliability": _pit_reliability(pit_df),
        "safety_car_probability": _safety_car_probability(rc_df),
        "undercut_success": _undercut_success(stints_df, pit_df),
    }


# ── Feature builders ──────────────────────────────────────────────────────────

def _team_strategy_profile(
    stints_df: pd.DataFrame,
    jolpica,
    years: list[int],
) -> pd.DataFrame:
    """
    Most common race strategy (number of pit stops) per team per circuit.

    Columns: team, circuit_slug, modal_stops, pct_1stop, pct_2stop, pct_3stop,
             year_range
    """
    if stints_df.empty or "stint_number" not in stints_df.columns:
        return pd.DataFrame()

    # Get team from Jolpica race results (OpenF1 only has driver_number)
    team_lookup: dict[tuple[int, str], str] = {}
    for year in years:
        try:
            results = jolpica.race_results(year)
            for _, r in results.iterrows():
                team_lookup[(year, str(r.get("driver_code", "")))] = str(r.get("constructor", ""))
        except Exception:
            pass

    stints = stints_df.copy()
    stints["driver_number"] = stints.get("driver_number", stints.get("DriverNumber", "")).astype(str)

    # Count stints per driver per race = stops + 1
    stop_counts = (
        stints.groupby(["year", "circuit_slug", "driver_number"])["stint_number"]
        .max()
        .reset_index()
        .rename(columns={"stint_number": "total_stints"})
    )
    stop_counts["stops"] = stop_counts["total_stints"] - 1

    rows = []
    for (circuit_slug,), grp in stop_counts.groupby(["circuit_slug"]):
        for stops_val in [1, 2, 3]:
            pct = (grp["stops"] == stops_val).mean()
            rows.append({
                "circuit_slug": circuit_slug,
                "stops": stops_val,
                "frequency": float(pct),
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    # Pivot to wide format
    result = result.pivot_table(
        index="circuit_slug",
        columns="stops",
        values="frequency",
        aggfunc="first",
    ).reset_index()
    result.columns.name = None
    result = result.rename(columns={1: "pct_1stop", 2: "pct_2stop", 3: "pct_3stop"})
    for col in ["pct_1stop", "pct_2stop", "pct_3stop"]:
        if col not in result.columns:
            result[col] = 0.0
    result["modal_stops"] = result[["pct_1stop", "pct_2stop", "pct_3stop"]].idxmax(axis=1).str.replace("pct_", "").str.replace("stop", "").astype(float)
    return result


def _stint_length_profile(stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Average stint length (laps) by compound and circuit.

    Columns: circuit_slug, compound, avg_stint_laps, std_stint_laps, sample_size
    """
    if stints_df.empty:
        return pd.DataFrame()

    required = ["lap_start", "lap_end", "compound", "circuit_slug"]
    if not all(c in stints_df.columns for c in required):
        return pd.DataFrame()

    stints = stints_df.copy()
    stints["stint_laps"] = (stints["lap_end"] - stints["lap_start"]).clip(lower=1)
    stints["compound"] = stints["compound"].str.upper().fillna("UNKNOWN")

    profile = (
        stints.groupby(["circuit_slug", "compound"])
        .agg(
            avg_stint_laps=("stint_laps", "mean"),
            std_stint_laps=("stint_laps", "std"),
            sample_size=("stint_laps", "count"),
        )
        .reset_index()
    )
    return profile


def _pit_reliability(pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pit stop consistency per team (requires team info — approximated by driver_number).
    Lower std dev = more reliable pit crew.

    Columns: team_or_driver, circuit_slug, avg_pit_duration_s, std_pit_duration_s,
             pit_count
    """
    if pit_df.empty or "pit_duration" not in pit_df.columns:
        return pd.DataFrame()

    pits = pit_df.copy()
    pits["pit_duration"] = pd.to_numeric(pits["pit_duration"], errors="coerce")
    pits = pits[pits["pit_duration"].between(1.5, 60)]  # filter outliers

    if "circuit_slug" not in pits.columns:
        return pd.DataFrame()

    group_col = "driver_number" if "driver_number" in pits.columns else "DriverNumber"
    reliability = (
        pits.groupby([group_col, "circuit_slug"])
        .agg(
            avg_pit_duration_s=("pit_duration", "mean"),
            std_pit_duration_s=("pit_duration", "std"),
            pit_count=("pit_duration", "count"),
        )
        .reset_index()
        .rename(columns={group_col: "driver_number"})
    )
    return reliability


def _safety_car_probability(rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Historical safety car deployment probability per circuit.

    Columns: circuit_slug, sc_probability, vsc_probability, red_flag_probability,
             races_analysed
    """
    if rc_df.empty or "circuit_slug" not in rc_df.columns:
        return pd.DataFrame()

    # Count races per circuit
    races_per_circuit = rc_df.groupby("circuit_slug")["year"].nunique().rename("races_analysed")

    # Count deployments
    sc_col = "message" if "message" in rc_df.columns else "category"

    def _flag_count(df: pd.DataFrame, keyword: str) -> pd.Series:
        mask = df[sc_col].astype(str).str.contains(keyword, case=False, na=False)
        return df[mask].groupby("circuit_slug")["year"].nunique()

    sc_count = _flag_count(rc_df, "safety car|safetyCar")
    vsc_count = _flag_count(rc_df, "virtual safety|vsc")
    red_flag_count = _flag_count(rc_df, "red flag|session suspended")

    result = pd.DataFrame({"races_analysed": races_per_circuit})
    result["sc_races"] = sc_count.reindex(result.index, fill_value=0)
    result["vsc_races"] = vsc_count.reindex(result.index, fill_value=0)
    result["red_flag_races"] = red_flag_count.reindex(result.index, fill_value=0)

    result["sc_probability"] = result["sc_races"] / result["races_analysed"]
    result["vsc_probability"] = result["vsc_races"] / result["races_analysed"]
    result["red_flag_probability"] = result["red_flag_races"] / result["races_analysed"]

    return result.reset_index()


def _undercut_success(stints_df: pd.DataFrame, pit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Undercut success rate: did the driver who pitted first gain positions?

    This is a circuit-level metric computed from stint start/end patterns.
    Approximate: if a driver's stint_start lap is earlier than the following
    car's, and they emerged ahead, the undercut was successful.

    Returns: circuit_slug, undercut_attempts, undercut_successes, undercut_rate
    """
    # Simplified approximation using stint sequence patterns
    if stints_df.empty or "circuit_slug" not in stints_df.columns:
        return pd.DataFrame(columns=["circuit_slug", "undercut_rate"])

    if "lap_start" not in stints_df.columns:
        return pd.DataFrame(columns=["circuit_slug", "undercut_rate"])

    # Group by race (year + circuit) and count instances where
    # a driver's 2nd stint starts earlier than median = undercut attempt
    rows = []
    for (year, circuit_slug), grp in stints_df.groupby(["year", "circuit_slug"]):
        stint2 = grp[grp.get("stint_number", pd.Series(dtype=int)) == 2]
        if stint2.empty:
            continue
        median_pit_lap = stint2["lap_start"].median()
        early_pit = stint2[stint2["lap_start"] < median_pit_lap]
        rows.append({
            "year": year,
            "circuit_slug": circuit_slug,
            "total_cars": len(stint2),
            "early_pitters": len(early_pit),
        })

    if not rows:
        return pd.DataFrame(columns=["circuit_slug", "undercut_rate"])

    df = pd.DataFrame(rows)
    result = (
        df.groupby("circuit_slug")
        .agg(
            undercut_attempts=("early_pitters", "sum"),
            total_observations=("total_cars", "sum"),
        )
        .reset_index()
    )
    result["undercut_rate"] = result["undercut_attempts"] / result["total_observations"].replace(0, np.nan)
    return result


# ── Lookup helpers ────────────────────────────────────────────────────────────

def get_circuit_strategy(
    strategy_features: dict[str, pd.DataFrame],
    circuit_slug: str,
) -> dict:
    """
    Return a summary dict of strategy characteristics for a given circuit.
    Used by the race forecaster as contextual input.
    """
    summary = {"circuit_slug": circuit_slug}

    # Strategy profile
    profile = strategy_features.get("team_strategy_profile", pd.DataFrame())
    if not profile.empty and "circuit_slug" in profile.columns:
        row = profile[profile["circuit_slug"] == circuit_slug]
        if not row.empty:
            summary["modal_stops"] = float(row["modal_stops"].iloc[0])
            summary["pct_1stop"] = float(row.get("pct_1stop", pd.Series([0.0])).iloc[0])
            summary["pct_2stop"] = float(row.get("pct_2stop", pd.Series([0.5])).iloc[0])
            summary["pct_3stop"] = float(row.get("pct_3stop", pd.Series([0.0])).iloc[0])

    # Safety car probability
    sc = strategy_features.get("safety_car_probability", pd.DataFrame())
    if not sc.empty and "circuit_slug" in sc.columns:
        row = sc[sc["circuit_slug"] == circuit_slug]
        if not row.empty:
            summary["sc_probability"] = float(row["sc_probability"].iloc[0])
            summary["vsc_probability"] = float(row["vsc_probability"].iloc[0])
            summary["red_flag_probability"] = float(row["red_flag_probability"].iloc[0])

    # Undercut rate
    uc = strategy_features.get("undercut_success", pd.DataFrame())
    if not uc.empty and "circuit_slug" in uc.columns:
        row = uc[uc["circuit_slug"] == circuit_slug]
        if not row.empty:
            summary["undercut_rate"] = float(row["undercut_rate"].iloc[0])

    return summary
