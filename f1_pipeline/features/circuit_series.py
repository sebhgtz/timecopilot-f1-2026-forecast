"""
Circuit-Specific Performance Series
=====================================
Builds per-driver, per-circuit historical finishing position time series
for TimeCopilot's race winner prediction (h=1).

Structure:
    unique_id       | ds    | y (finish pos) | covariates...
    VER_australia   | 2018  | 2              | grid_pos=5, weather=0, stints=2
    VER_australia   | 2019  | DNF→20         | ...
    VER_australia   | 2024  | 1              | ...

One data point per year a driver contested that circuit.
h=1 forecasts this year's finishing position, informed by the trend
(e.g. Verstappen has been improving at Australia year-over-year).

Data source: Jolpica-F1 race results (2015–present).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from ..collectors.jolpica_collector import JolpicaCollector
from ..collectors.calendar_manager import SECOND_YEAR_2026 as _SECOND_YEAR_DRIVERS

# Circuits that appear on the 2026 calendar (slug → common name fragments)
CIRCUITS_2026 = {
    "australia": ["australian", "australia"],
    "china": ["chinese", "china"],
    "japan": ["japanese", "japan", "suzuka"],
    "bahrain": ["bahrain"],
    "saudi_arabia": ["saudi", "jeddah"],
    "miami": ["miami"],
    "canada": ["canadian", "canada", "montreal"],
    "monaco": ["monaco"],
    "spain": ["spanish", "spain", "barcelona"],
    "austria": ["austrian", "austria", "styrian", "red bull ring"],
    "great_britain": ["british", "great britain", "silverstone"],
    "belgium": ["belgian", "belgium", "spa"],
    "hungary": ["hungarian", "hungary"],
    "netherlands": ["dutch", "netherlands", "zandvoort"],
    "italy": ["italian", "italy", "monza"],
    "madrid": ["madrid"],
    "azerbaijan": ["azerbaijan", "baku"],
    "singapore": ["singapore"],
    "united_states": ["united states", "cota", "austin"],
    "mexico": ["mexico"],
    "brazil": ["brazil", "são paulo", "sao paulo", "interlagos"],
    "las_vegas": ["las vegas"],
    "qatar": ["qatar", "lusail"],
    "abu_dhabi": ["abu dhabi", "yas"],
}

# Maximum finishing position (used for DNF / DNS)
DNF_POSITION = 20

# Weather encoding
_WEATHER_DRY = 0
_WEATHER_WET = 1

# Strategy encoding
_STRATEGY_1STOP = 1
_STRATEGY_2STOP = 2
_STRATEGY_3STOP = 3


def _circuit_slug(race_name: str) -> Optional[str]:
    """Map a Jolpica race name to a 2026 circuit slug."""
    name_lower = race_name.lower()
    for slug, fragments in CIRCUITS_2026.items():
        if any(f in name_lower for f in fragments):
            return slug
    return None


def _finish_position(row) -> int:
    """Convert a result row to a numeric finishing position. DNF/DNS → DNF_POSITION."""
    pos = row.get("finish_position")
    status = str(row.get("status", "")).lower()
    if pos is not None and not np.isnan(float(pos if pos else "nan") if not isinstance(pos, float) else pos):
        try:
            return int(pos)
        except (ValueError, TypeError):
            pass
    # DNF/DNS/DSQ etc.
    if any(s in status for s in ("finished", "lapped", "+1")):
        return DNF_POSITION - 1
    return DNF_POSITION


def build_circuit_series(
    years: list[int],
    jolpica: Optional[JolpicaCollector] = None,
    circuits: Optional[list[str]] = None,
    drivers: Optional[list[str]] = None,
    include_grid_position: bool = True,
) -> pd.DataFrame:
    """
    Build per-driver, per-circuit annual finishing position time series.

    Args:
        years:                e.g. list(range(2015, 2026))
        jolpica:              JolpicaCollector instance
        circuits:             filter to specific circuit slugs (defaults to all 2026 circuits)
        drivers:              filter to specific driver codes
        include_grid_position: add qualifying grid position as covariate

    Returns:
        TimeCopilot long-format DataFrame:
            unique_id, ds, y, driver_code, circuit_slug, grid_position,
            weather_enc, strategy_stops, car_pace_rank, year
    """
    jc = jolpica or JolpicaCollector()
    circuit_filter = set(circuits) if circuits else set(CIRCUITS_2026.keys())

    print(f"\n🏁 Building circuit-specific performance series ({len(years)} seasons)...")

    all_rows = []

    for year in years:
        print(f"  → {year}...")
        try:
            results_df = jc.race_results(year)
        except Exception as exc:
            print(f"     ⚠️  {exc}")
            continue

        if results_df.empty:
            continue

        # Add qualifying grid positions if requested
        if include_grid_position:
            try:
                quali_df = jc.qualifying_results(year)
                grid_lookup: dict[tuple[int, str], int] = {}
                for _, qrow in quali_df.iterrows():
                    grid_lookup[(int(qrow["round"]), str(qrow["driver_code"]))] = int(qrow.get("grid_position", 0))
            except Exception:
                grid_lookup = {}
        else:
            grid_lookup = {}

        # Compute per-constructor average finish position (proxy for car pace rank)
        team_avg = (
            results_df.groupby("constructor")["finish_position"]
            .mean()
            .rank()
            .to_dict()
        )

        for _, row in results_df.iterrows():
            circuit_slug = _circuit_slug(row.get("race_name", ""))
            if circuit_slug is None or circuit_slug not in circuit_filter:
                continue

            driver_code = str(row.get("driver_code", ""))
            if drivers and driver_code not in drivers:
                continue

            finish_pos = _finish_position(row)
            grid_pos = grid_lookup.get((int(row.get("round", 0)), driver_code), 0)
            car_pace_rank = team_avg.get(row.get("constructor", ""), 5.0)

            # Strategy stops: inferred from grid vs race position differential (proxy)
            # Will be replaced by OpenF1 stints when available
            strategy_stops = _infer_strategy_stops(int(row.get("round", 0)), year)

            # Year as datetime (Jan 1 of race year, approximate)
            race_date = row.get("race_date")
            if pd.isna(race_date):
                race_date = pd.Timestamp(f"{year}-01-01")

            all_rows.append({
                "unique_id": f"{driver_code}_{circuit_slug}",
                "ds": race_date,
                "y": float(finish_pos),
                "driver_code": driver_code,
                "driver_name": row.get("driver_name", ""),
                "constructor": row.get("constructor", ""),
                "circuit_slug": circuit_slug,
                "race_name": row.get("race_name", ""),
                "grid_position": float(grid_pos),
                "weather_enc": _WEATHER_DRY,  # enriched later from OpenF1
                "strategy_stops": float(strategy_stops),
                "car_pace_rank": float(car_pace_rank),
                "year": year,
                "status": row.get("status", ""),
                "points": float(row.get("points", 0)),
            })

    result = pd.DataFrame(all_rows)
    if result.empty:
        return result

    result["ds"] = pd.to_datetime(result["ds"])
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    print(f"   ✓ {result['unique_id'].nunique()} driver-circuit series, {len(result):,} rows")
    return result


def enrich_with_weather(circuit_df: pd.DataFrame, openf1_rc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather_enc (0=dry, 1=wet) to the circuit series using
    OpenF1 race control data (which logs 'wet' track declarations).

    openf1_rc_df must have columns: year, gp, message (from bulk_race_control).
    """
    if openf1_rc_df.empty or "message" not in openf1_rc_df.columns:
        return circuit_df

    wet_races: set[tuple[int, str]] = set()
    for _, row in openf1_rc_df.iterrows():
        msg = str(row.get("message", "")).lower()
        if "wet" in msg or "intermediate" in msg or "rain" in msg:
            slug = _circuit_slug(row.get("gp", ""))
            if slug:
                wet_races.add((int(row.get("year", 0)), slug))

    circuit_df = circuit_df.copy()
    circuit_df["weather_enc"] = circuit_df.apply(
        lambda r: _WEATHER_WET if (int(r["year"]), r["circuit_slug"]) in wet_races else _WEATHER_DRY,
        axis=1,
    )
    return circuit_df


def enrich_with_strategy(circuit_df: pd.DataFrame, stints_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the estimated strategy_stops with actual values from OpenF1 stints.

    stints_df must have: year, gp, driver_number, stint_number columns.
    """
    if stints_df.empty:
        return circuit_df

    # Max stint number = number of stints = number of stops + 1
    stint_counts = (
        stints_df.groupby(["year", "gp"])["stint_number"]
        .max()
        .reset_index()
        .rename(columns={"stint_number": "max_stints"})
    )
    stint_counts["strategy_stops"] = stint_counts["max_stints"] - 1
    stint_counts["circuit_slug"] = stint_counts["gp"].apply(_circuit_slug)
    stint_counts = stint_counts.dropna(subset=["circuit_slug"])
    stint_lookup = stint_counts.set_index(["year", "circuit_slug"])["strategy_stops"].to_dict()

    circuit_df = circuit_df.copy()
    circuit_df["strategy_stops"] = circuit_df.apply(
        lambda r: float(stint_lookup.get((int(r["year"]), r["circuit_slug"]), r["strategy_stops"])),
        axis=1,
    )
    return circuit_df


def add_current_race_covariates(
    circuit_df: pd.DataFrame,
    circuit_slug: str,
    year: int,
    grid_position: Optional[float] = None,
    weather_enc: Optional[int] = None,
    fp1_pace_rank: Optional[float] = None,
    fp2_pace_rank: Optional[float] = None,
    fp2_long_run_rank: Optional[float] = None,
    car_pace_rank: Optional[float] = None,
    car_pace_rank_map: Optional[dict] = None,
    active_drivers: Optional[set] = None,
    new_driver_entries: Optional[dict] = None,
    current_team_map: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Append a placeholder row for the current (upcoming) race for each driver,
    populated with the latest weekend covariates (grid position, FP pace, etc.).

    This row has y=NaN — TimeCopilot will forecast it.

    Args:
        circuit_df:         existing circuit series
        circuit_slug:       e.g. "australia"
        year:               e.g. 2026
        grid_position:      from qualifying (most predictive feature)
        weather_enc:        0=dry, 1=wet
        fp1_pace_rank:      relative pace rank in FP1
        fp2_pace_rank:      relative pace rank in FP2
        fp2_long_run_rank:  race pace rank from FP2 long runs
        car_pace_rank:      fallback constructor championship position for all drivers
        car_pace_rank_map:  dict of {constructor_name: championship_position} for the
                            current season. When provided, each driver's rank is looked
                            up by their constructor (from current_team_map or history),
                            overriding the global car_pace_rank fallback. Use Jolpica
                            constructor_standings(year, round=N-1) to build this.
        active_drivers:     set of driver codes currently on the grid; when provided,
                            only these drivers get a placeholder row (filters out retired
                            drivers who appear in historical data). When None, defaults
                            to a recency filter: drivers who raced in the last 3 seasons.
        new_driver_entries: dict of {driver_code: {"constructor": str,
                            "estimated_position": float}} for drivers with no circuit
                            history (e.g. rookies, new teams). Gets 3 synthetic history
                            rows so TimeCopilot has enough data points to forecast them.
        current_team_map:   dict of {driver_code: constructor_name} reflecting the
                            current season's team assignments. Overrides the historical
                            constructor from the series — handles mid-season transfers
                            and off-season moves (e.g. HAM→Ferrari, SAI→Williams).
    """
    existing = circuit_df[circuit_df["circuit_slug"] == circuit_slug].copy()

    # --- Determine which historical drivers to include ---
    if active_drivers is not None:
        # Use the provided set; intersect with drivers who have history
        historical_at_circuit = set(
            existing[existing["driver_code"].isin(active_drivers)]["driver_code"].unique()
        )
    else:
        # Default: recency filter — only drivers who raced here in the last 3 seasons
        recent = existing[existing["year"] >= year - 3]
        historical_at_circuit = set(recent["driver_code"].unique())

    new_rows = []
    race_date = pd.Timestamp(f"{year}-06-15")  # approximate placeholder

    for driver_code in sorted(historical_at_circuit):
        driver_history = existing[existing["driver_code"] == driver_code]
        # current_team_map overrides historical constructor (handles transfers like HAM→Ferrari)
        if current_team_map and driver_code in current_team_map:
            constructor = current_team_map[driver_code]
        else:
            constructor = driver_history["constructor"].iloc[-1] if not driver_history.empty else ""

        # car_pace_rank_map gives per-constructor championship position for this season
        if car_pace_rank_map and constructor in car_pace_rank_map:
            driver_car_rank = float(car_pace_rank_map[constructor])
        else:
            driver_car_rank = float(car_pace_rank) if car_pace_rank is not None else 5.0

        new_rows.append(_placeholder_row(
            driver_code, circuit_slug, year, race_date,
            constructor,
            driver_history["driver_name"].iloc[-1] if not driver_history.empty else "",
            grid_position, weather_enc, fp1_pace_rank, fp2_pace_rank,
            fp2_long_run_rank, driver_car_rank,
        ))

    # --- Add synthetic history for new drivers with no circuit data ---
    new_driver_entries = new_driver_entries or {}
    for driver_code, meta in new_driver_entries.items():
        if driver_code in historical_at_circuit:
            continue  # already has real history

        constructor = meta.get("constructor", "")
        est_pos = float(meta.get("estimated_position", 18.0))

        # 3 synthetic historical rows so TimeCopilot has min series length
        for back in (3, 2, 1):
            hist_date = pd.Timestamp(f"{year - back}-06-15")
            new_rows.append({
                "unique_id": f"{driver_code}_{circuit_slug}",
                "ds": hist_date,
                "y": est_pos + float(back - 2) * 0.5,   # slight trend (improving)
                "driver_code": driver_code,
                "driver_name": meta.get("driver_name", driver_code),
                "constructor": constructor,
                "circuit_slug": circuit_slug,
                "race_name": f"{year - back} {circuit_slug.replace('_', ' ').title()} Grand Prix",
                "grid_position": np.nan,
                "weather_enc": float(_WEATHER_DRY),
                "strategy_stops": 2.0,
                "car_pace_rank": float(meta.get("car_pace_rank", 10.0)),
                "year": year - back,
                "status": "synthetic",
                "points": 0.0,
            })

        # Placeholder row for the forecast year
        if car_pace_rank_map and constructor in car_pace_rank_map:
            new_driver_car_rank = float(car_pace_rank_map[constructor])
        else:
            new_driver_car_rank = float(car_pace_rank) if car_pace_rank is not None else float(meta.get("car_pace_rank", 10.0))
        new_rows.append(_placeholder_row(
            driver_code, circuit_slug, year, race_date,
            constructor, meta.get("driver_name", driver_code),
            grid_position, weather_enc, fp1_pace_rank, fp2_pace_rank,
            fp2_long_run_rank, new_driver_car_rank,
        ))

    # --- Backfill second-year drivers who have <3 real circuit data points ---
    # Drivers who debuted in 2025 have exactly 1 historical appearance per circuit,
    # which is below the min_len=3 threshold. We generate 2 synthetic historical rows
    # so TimeCopilot receives enough data points to produce a real forecast.
    # Synthetic rows are based on the driver's actual 2025 performance (not a generic
    # default), and carry a slight upward trend to represent career improvement.
    for driver_code, tier in _SECOND_YEAR_DRIVERS.items():
        if driver_code not in historical_at_circuit:
            continue  # no real data at all — handled by new_driver_entries path
        driver_history = existing[existing["driver_code"] == driver_code]
        circuit_history = driver_history[driver_history["circuit_slug"] == circuit_slug]
        if len(circuit_history) >= 3:
            continue  # already has enough data, no augmentation needed

        # Estimate base finishing position from actual 2025 data at this circuit;
        # fall back to overall 2025 avg finish, then car tier default.
        if not circuit_history.empty:
            est_pos = float(circuit_history["y"].mean())
        else:
            overall_avg = driver_history["y"].mean()
            est_pos = float(overall_avg) if not np.isnan(overall_avg) else _tier_to_est_pos(tier)

        if current_team_map and driver_code in current_team_map:
            constructor = current_team_map[driver_code]
        elif not driver_history.empty:
            constructor = driver_history["constructor"].iloc[-1]
        else:
            constructor = ""

        if car_pace_rank_map and constructor in car_pace_rank_map:
            driver_car_rank = float(car_pace_rank_map[constructor])
        else:
            driver_car_rank = float(car_pace_rank) if car_pace_rank is not None else 5.0

        driver_name = driver_history["driver_name"].iloc[-1] if not driver_history.empty else driver_code
        synthetic_needed = max(0, 3 - len(circuit_history))

        # Generate synthetic rows for the missing prior years (e.g. 2023, 2024 if only 2025 exists).
        # Trend: slightly worse in earlier years (est_pos + back * 0.5) → improving trajectory.
        for back in range(synthetic_needed + 1, 1, -1):
            hist_date = pd.Timestamp(f"{year - back}-06-15")
            new_rows.append({
                "unique_id": f"{driver_code}_{circuit_slug}",
                "ds": hist_date,
                "y": est_pos + float(back - 1) * 0.5,
                "driver_code": driver_code,
                "driver_name": driver_name,
                "constructor": constructor,
                "circuit_slug": circuit_slug,
                "race_name": f"{year - back} {circuit_slug.replace('_', ' ').title()} Grand Prix",
                "grid_position": np.nan,
                "weather_enc": float(_WEATHER_DRY),
                "strategy_stops": 2.0,
                "car_pace_rank": driver_car_rank,
                "year": year - back,
                "status": "synthetic_second_year",
                "points": 0.0,
            })

    if not new_rows:
        return circuit_df

    new_df = pd.DataFrame(new_rows)

    # When active_drivers is provided, strip historical rows for non-active drivers
    # so the race forecaster doesn't extrapolate positions for retired/absent drivers.
    base_df = circuit_df
    if active_drivers is not None:
        all_active = active_drivers | set(new_driver_entries or {}) | set(_SECOND_YEAR_DRIVERS.keys())
        base_df = circuit_df[circuit_df["driver_code"].isin(all_active)].copy()

    result = pd.concat([base_df, new_df], ignore_index=True)
    result["ds"] = pd.to_datetime(result["ds"])
    return result.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _placeholder_row(
    driver_code: str,
    circuit_slug: str,
    year: int,
    race_date,
    constructor: str,
    driver_name: str,
    grid_position,
    weather_enc,
    fp1_pace_rank,
    fp2_pace_rank,
    fp2_long_run_rank,
    car_pace_rank,
) -> dict:
    """Build a single NaN placeholder row for TimeCopilot to forecast."""
    return {
        "unique_id": f"{driver_code}_{circuit_slug}",
        "ds": race_date,
        "y": np.nan,                       # ← to be forecast
        "driver_code": driver_code,
        "driver_name": driver_name,
        "constructor": constructor,
        "circuit_slug": circuit_slug,
        "race_name": f"{year} {circuit_slug.replace('_', ' ').title()} Grand Prix",
        "grid_position": float(grid_position) if grid_position is not None else np.nan,
        "weather_enc": float(weather_enc) if weather_enc is not None else float(_WEATHER_DRY),
        "strategy_stops": 2.0,
        "car_pace_rank": float(car_pace_rank) if car_pace_rank is not None else 5.0,
        "fp1_pace_rank": float(fp1_pace_rank) if fp1_pace_rank is not None else np.nan,
        "fp2_pace_rank": float(fp2_pace_rank) if fp2_pace_rank is not None else np.nan,
        "fp2_long_run_rank": float(fp2_long_run_rank) if fp2_long_run_rank is not None else np.nan,
        "year": year,
        "status": "",
        "points": np.nan,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tier_to_est_pos(tier: str) -> float:
    """Convert a performance tier string to a numeric estimated finishing position."""
    return {"front": 8.0, "midfield": 12.0, "back": 15.0}.get(tier, 14.0)


def _infer_strategy_stops(round_num: int, year: int) -> int:
    """
    Simple heuristic for strategy stops when OpenF1 data isn't available.
    Street circuits tend toward 2-stop; power circuits toward 1-stop.
    Overridden by actual OpenF1 data when available.
    """
    # Default to 2-stop; enriched by actual data via enrich_with_strategy()
    return 2
