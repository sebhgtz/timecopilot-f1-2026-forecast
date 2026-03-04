"""
OpenF1 API Collector
=====================
REST client for the OpenF1 API (https://api.openf1.org/v1/).

Covers 2023-present. No API key required for historical data.
Provides strategy-critical data not available in FastF1:
  - Stints (tire compound + lap ranges)
  - Pit stop durations
  - Race control messages (safety cars, VSC, red flags)
  - Session-level intervals (race gaps)
  - Overtakes

Usage:
    oc = OpenF1Collector()
    stints = oc.stints(year=2024, gp="Bahrain Grand Prix")
    pit = oc.pit_stops(year=2024, gp="Bahrain Grand Prix")
    rc  = oc.race_control(year=2024, gp="Bahrain Grand Prix")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Union
import requests
import pandas as pd

BASE_URL = "https://api.openf1.org/v1"
CACHE_DIR = Path(".openf1_cache")
_REQUEST_DELAY = 0.2


class OpenF1Collector:
    def __init__(self, cache_dir: Path = CACHE_DIR, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Low-level HTTP ────────────────────────────────────────────────────────

    def _get(self, endpoint: str, params: dict) -> list[dict]:
        """
        GET /v1/{endpoint} with params, returning a list of record dicts.
        Caches by endpoint + sorted params.
        """
        cache_key = endpoint + "_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_file = self.cache_dir / f"{cache_key[:200]}.json"

        if self.use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        time.sleep(_REQUEST_DELAY)
        url = f"{BASE_URL}/{endpoint}"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if self.use_cache:
            cache_file.write_text(json.dumps(data))

        return data if isinstance(data, list) else []

    # ── Session resolution ────────────────────────────────────────────────────

    def _get_session_key(self, year: int, gp: str, session_type: str) -> Optional[int]:
        """Resolve a session key from year + GP name + session type."""
        params = {
            "year": year,
            "circuit_short_name": _circuit_short(gp),
            "session_name": _normalize_session_name(session_type),
        }
        rows = self._get("sessions", params)
        if not rows:
            # Fallback: broader search
            rows = self._get("sessions", {"year": year, "session_name": _normalize_session_name(session_type)})
            # Filter by GP name
            rows = [r for r in rows if gp.lower() in r.get("meeting_name", "").lower()
                    or _circuit_short(gp).lower() in r.get("circuit_short_name", "").lower()]
        return rows[0]["session_key"] if rows else None

    def _get_meeting_key(self, year: int, gp: str) -> Optional[int]:
        """Resolve the meeting key for a GP."""
        rows = self._get("meetings", {"year": year})
        for r in rows:
            if gp.lower() in r.get("meeting_name", "").lower():
                return r.get("meeting_key")
        return None

    # ── Stints ────────────────────────────────────────────────────────────────

    def stints(
        self, year: int, gp: str, session_type: str = "Race"
    ) -> pd.DataFrame:
        """
        Tire stints per driver for a session.

        Returns DataFrame with:
            session_key, driver_number, stint_number, compound,
            lap_start, lap_end, tyre_age_at_start
        """
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            print(f"  ⚠️  OpenF1: no session key for {year} {gp} {session_type}")
            return pd.DataFrame()

        rows = self._get("stints", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Pit stops ─────────────────────────────────────────────────────────────

    def pit_stops(self, year: int, gp: str) -> pd.DataFrame:
        """
        Pit stop records for a race weekend.

        Returns DataFrame with:
            session_key, driver_number, lap_number, pit_duration, date
        """
        sk = self._get_session_key(year, gp, "Race")
        if sk is None:
            return pd.DataFrame()

        rows = self._get("pit", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Race control messages ─────────────────────────────────────────────────

    def race_control(self, year: int, gp: str, session_type: str = "Race") -> pd.DataFrame:
        """
        Race control messages: safety cars, VSC, red flags, etc.

        Returns DataFrame with:
            session_key, date, lap_number, category, flag, message
        """
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            return pd.DataFrame()

        rows = self._get("race_control", {"session_key": sk})
        df = pd.DataFrame(rows)
        return df

    def safety_car_laps(self, year: int, gp: str) -> list[int]:
        """
        Return list of lap numbers on which a safety car (SC or VSC) was deployed.
        """
        df = self.race_control(year, gp)
        if df.empty:
            return []
        sc = df[df.get("category", pd.Series()).str.contains("SafetyCar|Vsc", case=False, na=False)]
        laps = sc["lap_number"].dropna().astype(int).tolist()
        return sorted(set(laps))

    # ── Lap data ──────────────────────────────────────────────────────────────

    def laps(self, year: int, gp: str, session_type: str = "Race") -> pd.DataFrame:
        """
        Lap-level data from OpenF1 (complements FastF1).

        Useful for live/2026 data. Returns DataFrame with:
            session_key, driver_number, lap_number, lap_duration,
            duration_sector_1/2/3, i1_speed, i2_speed, finish_line_speed,
            st_speed, date_start, is_pit_out_lap
        """
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            return pd.DataFrame()

        rows = self._get("laps", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Intervals (race gaps) ─────────────────────────────────────────────────

    def intervals(self, year: int, gp: str) -> pd.DataFrame:
        """
        Real-time race intervals (gap to ahead car + gap to leader).
        Updated ~every 4 seconds during the race.

        Returns DataFrame with: session_key, date, driver_number, gap_to_leader, interval
        """
        sk = self._get_session_key(year, gp, "Race")
        if sk is None:
            return pd.DataFrame()

        rows = self._get("intervals", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Positions ─────────────────────────────────────────────────────────────

    def positions(self, year: int, gp: str, session_type: str = "Race") -> pd.DataFrame:
        """
        Position data throughout a session.

        Returns DataFrame with: session_key, date, driver_number, position
        """
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            return pd.DataFrame()

        rows = self._get("position", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Session results ───────────────────────────────────────────────────────

    def session_result(self, year: int, gp: str, session_type: str = "Race") -> pd.DataFrame:
        """
        Final standings after a session.

        Returns DataFrame with: session_key, driver_number, position, ...
        """
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            return pd.DataFrame()

        rows = self._get("session_result", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Weather ───────────────────────────────────────────────────────────────

    def weather(self, year: int, gp: str, session_type: str = "Race") -> pd.DataFrame:
        """
        Weather data (updated every minute) for a session.

        Columns: air_temperature, track_temperature, humidity, pressure,
                 rainfall, wind_direction, wind_speed, date
        """
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            return pd.DataFrame()

        rows = self._get("weather", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Drivers ───────────────────────────────────────────────────────────────

    def drivers(self, year: int, gp: str, session_type: str = "Race") -> pd.DataFrame:
        """Driver metadata for a session (number, code, team, etc.)."""
        sk = self._get_session_key(year, gp, session_type)
        if sk is None:
            return pd.DataFrame()

        rows = self._get("drivers", {"session_key": sk})
        return pd.DataFrame(rows)

    # ── Bulk strategy data ────────────────────────────────────────────────────

    def bulk_stints(
        self, years: list[int], race_names: list[str]
    ) -> pd.DataFrame:
        """Pull stint data for multiple seasons and races."""
        dfs = []
        for year in years:
            for gp in race_names:
                try:
                    df = self.stints(year, gp)
                    if not df.empty:
                        df["year"] = year
                        df["gp"] = gp
                        dfs.append(df)
                except Exception as exc:
                    print(f"  ⚠️  {year} {gp} stints: {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def bulk_pit_stops(
        self, years: list[int], race_names: list[str]
    ) -> pd.DataFrame:
        """Pull pit stop data for multiple seasons and races."""
        dfs = []
        for year in years:
            for gp in race_names:
                try:
                    df = self.pit_stops(year, gp)
                    if not df.empty:
                        df["year"] = year
                        df["gp"] = gp
                        dfs.append(df)
                except Exception as exc:
                    print(f"  ⚠️  {year} {gp} pit: {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def bulk_race_control(
        self, years: list[int], race_names: list[str]
    ) -> pd.DataFrame:
        """Pull race control messages for multiple seasons and races."""
        dfs = []
        for year in years:
            for gp in race_names:
                try:
                    df = self.race_control(year, gp)
                    if not df.empty:
                        df["year"] = year
                        df["gp"] = gp
                        dfs.append(df)
                except Exception as exc:
                    print(f"  ⚠️  {year} {gp} race_control: {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

# Maps common GP name fragments to OpenF1 circuit_short_name
_GP_TO_SHORT: dict[str, str] = {
    "australia": "melbourne",
    "china": "shanghai",
    "japan": "suzuka",
    "bahrain": "bahrain",
    "saudi arabia": "jeddah",
    "miami": "miami",
    "canada": "montreal",
    "monaco": "monaco",
    "spain": "barcelona",
    "austria": "spielberg",
    "great britain": "silverstone",
    "belgium": "spa",
    "hungary": "budapest",
    "netherlands": "zandvoort",
    "italy": "monza",
    "madrid": "madrid",
    "azerbaijan": "baku",
    "singapore": "singapore",
    "united states": "austin",
    "mexico": "mexico_city",
    "brazil": "sao_paulo",
    "las vegas": "las_vegas",
    "qatar": "lusail",
    "abu dhabi": "yas_marina",
}


def _circuit_short(gp: str) -> str:
    gp_lower = gp.lower()
    for key, short in _GP_TO_SHORT.items():
        if key in gp_lower:
            return short
    # Fallback: return first word lowercased
    return gp_lower.split()[0]


# Maps session type names to OpenF1 session_name values
_SESSION_NAME_MAP: dict[str, str] = {
    "fp1": "Practice 1",
    "fp2": "Practice 2",
    "fp3": "Practice 3",
    "practice 1": "Practice 1",
    "practice 2": "Practice 2",
    "practice 3": "Practice 3",
    "qualifying": "Qualifying",
    "sprint qualifying": "Sprint Qualifying",
    "sprint shootout": "Sprint Qualifying",
    "sprint": "Sprint",
    "race": "Race",
    "r": "Race",
}


def _normalize_session_name(s: str) -> str:
    return _SESSION_NAME_MAP.get(s.lower(), s)
