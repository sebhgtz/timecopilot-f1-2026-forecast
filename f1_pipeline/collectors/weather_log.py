"""
Weather Log — Learning Logic
============================
Records actual session weather after each race weekend and derives
wet-weather driver performance statistics from historical circuit data.

Two-component design:

1. **Race weather log** (reports/weather_log.csv)
   After every race, fetch actual qualifying + race weather from the
   Open-Meteo archive and append a row. Accumulates over the season so
   we can audit forecast accuracy and track circuit-level trends.

   Columns:
       race_slug, year, round, race_name, circuit_type,
       session, session_date, weather_enc, precip_mm, precip_prob

2. **Wet driver statistics** (computed from circuit_df)
   The historical circuit series already has `weather_enc` (0/1) per race.
   `wet_driver_stats()` computes each driver's average finishing position
   in wet vs dry conditions — a positive "advantage" means the driver
   finishes better in wet than in dry (relative to their own average).

   These stats are injected into the TimeCopilot LLM query when wet
   conditions are forecast for the race weekend.

Usage:
    from f1_pipeline.collectors.weather_log import (
        log_race_weather, wet_driver_stats, format_wet_stats_for_query,
    )

    # After a race — log actual weather:
    log_race_weather("great_britain", 2026, 11, "British Grand Prix",
                     "balanced", race_date, qualifying_date)

    # When forecasting in wet conditions — get driver stats:
    stats = wet_driver_stats(circuit_df)
    query_snippet = format_wet_stats_for_query(stats)
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .weather_fetcher import fetch_session_weather, CIRCUIT_COORDS


# ── Paths ──────────────────────────────────────────────────────────────────────

WEATHER_LOG_PATH = Path("reports/weather_log.csv")

_LOG_COLUMNS = [
    "race_slug", "year", "round", "race_name", "circuit_type",
    "session", "session_date", "weather_enc", "precip_mm", "precip_prob",
]


# ── Weather logging ────────────────────────────────────────────────────────────

def log_race_weather(
    race_slug: str,
    year: int,
    round_number: int,
    race_name: str,
    circuit_type: str,
    race_date: date,
    qualifying_date: Optional[date] = None,
) -> None:
    """
    Fetch actual weather (from Open-Meteo archive) for qualifying and race
    sessions, then append rows to reports/weather_log.csv.

    Safe to call multiple times for the same race — deduplicates on
    (race_slug, year, session) before writing.

    Args:
        race_slug:       e.g. "great_britain"
        year:            e.g. 2026
        round_number:    1-24
        race_name:       e.g. "British Grand Prix"
        circuit_type:    "street" | "power" | "balanced" | "technical"
        race_date:       race Sunday date
        qualifying_date: Saturday date (defaults to race_date - 1 day)
    """
    coords = CIRCUIT_COORDS.get(race_slug)
    if coords is None:
        print(f"  ⚠️  weather_log: no coordinates for '{race_slug}' — skipping weather log.")
        return

    lat, lon = coords
    q_date = qualifying_date or (race_date - timedelta(days=1))

    sessions_to_log = [
        ("Qualifying", q_date),
        ("Race", race_date),
    ]

    new_rows = []
    for session_name, s_date in sessions_to_log:
        w = fetch_session_weather(lat, lon, s_date)
        new_rows.append({
            "race_slug":    race_slug,
            "year":         year,
            "round":        round_number,
            "race_name":    race_name,
            "circuit_type": circuit_type,
            "session":      session_name,
            "session_date": s_date.isoformat(),
            "weather_enc":  w["enc"],
            "precip_mm":    w["precip_mm"],
            "precip_prob":  w["precip_prob"],
        })

    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows)

    # Load and deduplicate existing log
    WEATHER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if WEATHER_LOG_PATH.exists():
        existing = pd.read_csv(WEATHER_LOG_PATH)
        # Remove any prior entries for this race (allow re-run corrections)
        existing = existing[
            ~((existing["race_slug"] == race_slug) & (existing["year"] == year))
        ]
        updated = pd.concat([existing, new_df], ignore_index=True)
    else:
        updated = new_df

    updated.to_csv(WEATHER_LOG_PATH, index=False)
    print(f"  📝  Weather log updated: {race_name} {year} "
          f"(Q={new_rows[0]['weather_enc']}, R={new_rows[1]['weather_enc']})")


def load_weather_log() -> pd.DataFrame:
    """Load the accumulated weather log, or return an empty DataFrame."""
    if not WEATHER_LOG_PATH.exists():
        return pd.DataFrame(columns=_LOG_COLUMNS)
    try:
        return pd.read_csv(WEATHER_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=_LOG_COLUMNS)


# ── Wet driver statistics (learning from historical circuit data) ───────────────

def wet_driver_stats(
    circuit_df: pd.DataFrame,
    min_wet_races: int = 2,
) -> pd.DataFrame:
    """
    Compute wet-weather driver performance from the historical circuit series.

    The circuit series has one row per driver per year per circuit, with:
        y             — finishing position (1 = win; lower = better)
        weather_enc   — 0=dry, 1=wet

    For each driver we compute:
        wet_avg_pos   — mean finish in wet races
        dry_avg_pos   — mean finish in dry races
        advantage     — dry_avg_pos - wet_avg_pos
                        (positive = finishes better in wet than dry)
        n_wet_races   — number of wet races in the history

    Only drivers with >= min_wet_races wet entries are returned.

    Args:
        circuit_df:    historical circuit series DataFrame
        min_wet_races: minimum wet-race sample size (default 2)

    Returns:
        DataFrame sorted by advantage descending (best wet-weather drivers first).
        Columns: driver_code, wet_avg_pos, dry_avg_pos, advantage, n_wet_races
    """
    if circuit_df.empty or "weather_enc" not in circuit_df.columns:
        return pd.DataFrame()

    df = circuit_df.dropna(subset=["y", "weather_enc"]).copy()
    df["weather_enc"] = df["weather_enc"].astype(int)

    wet = df[df["weather_enc"] == 1]
    dry = df[df["weather_enc"] == 0]

    # Aggregates per driver
    wet_agg = (
        wet.groupby("driver_code")["y"]
        .agg(wet_avg_pos="mean", n_wet_races="count")
        .reset_index()
    )
    dry_agg = (
        dry.groupby("driver_code")["y"]
        .agg(dry_avg_pos="mean")
        .reset_index()
    )

    stats = wet_agg.merge(dry_agg, on="driver_code", how="left")
    stats = stats[stats["n_wet_races"] >= min_wet_races].copy()

    # Positive advantage = driver finishes BETTER in wet than their dry average
    stats["advantage"] = stats["dry_avg_pos"] - stats["wet_avg_pos"]
    stats = stats.sort_values("advantage", ascending=False).reset_index(drop=True)

    return stats[["driver_code", "wet_avg_pos", "dry_avg_pos", "advantage", "n_wet_races"]]


def format_wet_stats_for_query(
    stats: pd.DataFrame,
    top_n: int = 4,
) -> str:
    """
    Format wet driver stats for injection into the TimeCopilot LLM race query.

    Only called when wet conditions are forecast. Returns an empty string
    if stats are empty or insufficient.

    Example output:
        "Historical wet performance (advantage = positions gained vs dry avg): "
        "HAM +2.1 pos (8 wet races), VER +1.4 pos (10 wet races), ..."
    """
    if stats.empty:
        return ""

    top = stats.head(top_n)
    bottom = stats.tail(2) if len(stats) > top_n + 1 else pd.DataFrame()

    parts = []
    for _, row in top.iterrows():
        sign = "+" if row["advantage"] >= 0 else ""
        parts.append(
            f"{row['driver_code']} {sign}{row['advantage']:.1f} pos "
            f"({int(row['n_wet_races'])} wet races)"
        )

    # Include worst wet-weather drivers as contrast
    penalty_parts = []
    for _, row in bottom.iterrows():
        if row["advantage"] < -0.5:  # only flag meaningful disadvantage
            penalty_parts.append(
                f"{row['driver_code']} {row['advantage']:.1f} pos"
            )

    result = (
        "Historical wet-weather performance (advantage = positions gained vs dry average): "
        + ", ".join(parts)
    )
    if penalty_parts:
        result += f". Wet-weather disadvantage: {', '.join(penalty_parts)}"
    result += "."
    return result
