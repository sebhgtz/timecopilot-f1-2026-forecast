"""
Jolpica-F1 Collector (Ergast API successor)
=============================================
Fetches championship standings, race results, and driver/constructor info
from the Jolpica-F1 API (http://api.jolpi.ca/ergast/f1/).

Ergast was deprecated at end of 2024; Jolpica is the drop-in replacement
with identical endpoint structure.

Usage:
    jc = JolpicaCollector()
    standings = jc.driver_standings(year=2025)  # all rounds
    standings_r5 = jc.driver_standings(year=2025, round=5)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
import requests
import pandas as pd

BASE_URL = "http://api.jolpi.ca/ergast/f1"
CACHE_DIR = Path(".jolpica_cache")
_REQUEST_DELAY = 1.2   # seconds between requests (free API rate limit: ~1 req/s)
_MAX_RETRIES = 3       # retry on 429 with exponential backoff


class JolpicaCollector:
    def __init__(self, cache_dir: Path = CACHE_DIR, use_cache: bool = True):
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Low-level HTTP ────────────────────────────────────────────────────────

    def _get(self, path: str, limit: int = 100, offset: int = 0) -> dict:
        """Fetch a single page from Jolpica with caching and 429 retry logic."""
        url = f"{BASE_URL}/{path}.json?limit={limit}&offset={offset}"
        cache_key = path.replace("/", "_") + f"_lim{limit}_off{offset}"
        cache_file = self.cache_dir / f"{cache_key}.json"

        if self.use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(_MAX_RETRIES):
            delay = _REQUEST_DELAY * (2 ** attempt)  # 1.2s, 2.4s, 4.8s
            time.sleep(delay)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    last_exc = Exception(f"429 Too Many Requests — attempt {attempt + 1}/{_MAX_RETRIES}")
                    continue
                resp.raise_for_status()
                data = resp.json()
                if self.use_cache:
                    cache_file.write_text(json.dumps(data))
                return data
            except Exception as exc:
                last_exc = exc
                if "429" not in str(exc):
                    raise  # Don't retry non-rate-limit errors

        raise last_exc

    def _get_all_pages(self, path: str, page_size: int = 100) -> dict:
        """
        Fetch all pages for an endpoint that returns paginated Ergast/Jolpica data.

        The Ergast/Jolpica API caps each response at 100 records. For full-season
        endpoints like /results or /qualifying a single season needs several pages.
        Returns a single merged dict where the inner table list contains all rows.
        """
        # Composite cache: keyed on path + "_all" to avoid collisions with _get
        cache_key = path.replace("/", "_") + "_all"
        cache_file = self.cache_dir / f"{cache_key}.json"
        if self.use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())

        offset = 0
        merged_data: dict = {}
        table_key: str = ""       # e.g. "RaceTable" or "QualifyingTable"
        inner_key: str = ""       # e.g. "Races" or "QualifyingRaces"
        all_rows: list = []

        while True:
            page = self._get(path, limit=page_size, offset=offset)
            mr = page.get("MRData", {})

            if not merged_data:
                merged_data = page      # keep first page as template
                # Discover table / inner list keys (varies by endpoint)
                for k, v in mr.items():
                    if isinstance(v, dict):
                        for ik, iv in v.items():
                            if isinstance(iv, list):
                                table_key, inner_key = k, ik
                                break

            if not table_key:
                break

            rows = mr.get(table_key, {}).get(inner_key, [])
            all_rows.extend(rows)

            total = int(mr.get("total", 0))
            offset += page_size
            if offset >= total or not rows:
                break

        # Patch merged_data with the full list
        if table_key and inner_key:
            merged_data.setdefault("MRData", {}).setdefault(table_key, {})[inner_key] = all_rows

        if self.use_cache:
            cache_file.write_text(json.dumps(merged_data))
        return merged_data

    # ── Championship standings ────────────────────────────────────────────────

    def driver_standings(
        self, year: int, round: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Driver championship standings after a given round (or final if no round).

        Returns DataFrame with columns:
            round, year, driver_code, driver_name, constructor, points, wins, position
        """
        path = f"{year}/driverStandings" if round is None else f"{year}/{round}/driverStandings"
        data = self._get(path)

        rows = []
        standings_list = (
            data.get("MRData", {})
                .get("StandingsTable", {})
                .get("StandingsLists", [])
        )
        for sl in standings_list:
            rd = int(sl.get("round", round or 0))
            for entry in sl.get("DriverStandings", []):
                driver = entry.get("Driver", {})
                constructors = entry.get("Constructors", [{}])
                rows.append({
                    "year": year,
                    "round": rd,
                    "position": int(entry.get("position", 0)),
                    "driver_code": driver.get("code", ""),
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "constructor": constructors[0].get("name", "") if constructors else "",
                    "points": float(entry.get("points", 0)),
                    "wins": int(entry.get("wins", 0)),
                })
        return pd.DataFrame(rows)

    def constructor_standings(
        self, year: int, round: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Constructor championship standings after a given round (or final if no round).

        Returns DataFrame with columns:
            round, year, constructor, points, wins, position
        """
        path = (
            f"{year}/constructorStandings"
            if round is None
            else f"{year}/{round}/constructorStandings"
        )
        data = self._get(path)

        rows = []
        standings_list = (
            data.get("MRData", {})
                .get("StandingsTable", {})
                .get("StandingsLists", [])
        )
        for sl in standings_list:
            rd = int(sl.get("round", round or 0))
            for entry in sl.get("ConstructorStandings", []):
                constructor = entry.get("Constructor", {})
                rows.append({
                    "year": year,
                    "round": rd,
                    "position": int(entry.get("position", 0)),
                    "constructor": constructor.get("name", ""),
                    "constructor_id": constructor.get("constructorId", ""),
                    "points": float(entry.get("points", 0)),
                    "wins": int(entry.get("wins", 0)),
                })
        return pd.DataFrame(rows)

    def all_driver_standings(self, year: int) -> pd.DataFrame:
        """
        Fetch driver standings after EVERY round of the season, returning
        a long-format DataFrame suitable for building championship time series.

        Columns: year, round, race_date, driver_code, driver_name, constructor,
                 points (cumulative), wins, position

        Returns empty DataFrame if no races have been run yet (e.g., pre-season).
        """
        # First get the race schedule so we have race dates
        schedule = self.race_results(year)
        if schedule.empty or not {"round", "race_date"}.issubset(schedule.columns):
            return pd.DataFrame()  # No races run yet for this year
        race_dates = schedule[["round", "race_date"]].drop_duplicates().set_index("round")["race_date"].to_dict()

        all_rows = []
        # Determine total rounds
        total_rounds = max(race_dates.keys()) if race_dates else 24
        for rd in range(1, total_rounds + 1):
            try:
                df = self.driver_standings(year, round=rd)
                if not df.empty:
                    df["race_date"] = race_dates.get(rd, pd.NaT)
                    all_rows.append(df)
            except Exception as exc:
                print(f"  ⚠️  Could not fetch {year} R{rd} driver standings: {exc}")

        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    def all_constructor_standings(self, year: int) -> pd.DataFrame:
        """Same as all_driver_standings but for constructors.

        Returns empty DataFrame if no races have been run yet (e.g., pre-season).
        """
        schedule = self.race_results(year)
        if schedule.empty or not {"round", "race_date"}.issubset(schedule.columns):
            return pd.DataFrame()  # No races run yet for this year
        race_dates = schedule[["round", "race_date"]].drop_duplicates().set_index("round")["race_date"].to_dict()

        all_rows = []
        total_rounds = max(race_dates.keys()) if race_dates else 24
        for rd in range(1, total_rounds + 1):
            try:
                df = self.constructor_standings(year, round=rd)
                if not df.empty:
                    df["race_date"] = race_dates.get(rd, pd.NaT)
                    all_rows.append(df)
            except Exception as exc:
                print(f"  ⚠️  Could not fetch {year} R{rd} constructor standings: {exc}")

        return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    # ── Race results ──────────────────────────────────────────────────────────

    def race_results(self, year: int, round: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch race results for a season or a specific round.

        Returns DataFrame with columns:
            year, round, race_name, circuit, race_date,
            driver_code, driver_name, constructor, grid, finish_position,
            points, status, fastest_lap_rank
        """
        path = f"{year}/results" if round is None else f"{year}/{round}/results"
        data = self._get_all_pages(path) if round is None else self._get(path)

        rows = []
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            rd = int(race.get("round", 0))
            race_name = race.get("raceName", "")
            circuit = race.get("Circuit", {}).get("circuitName", "")
            race_date = race.get("date", "")
            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                rows.append({
                    "year": year,
                    "round": rd,
                    "race_name": race_name,
                    "circuit": circuit,
                    "race_date": pd.to_datetime(race_date) if race_date else pd.NaT,
                    "driver_code": driver.get("code", ""),
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "constructor": constructor.get("name", ""),
                    "grid": _safe_int(result.get("grid")) or 0,
                    "finish_position": _safe_int(result.get("positionText")),
                    "points": float(result.get("points", 0)),
                    "status": result.get("status", ""),
                    "fastest_lap_rank": _safe_int(
                        result.get("FastestLap", {}).get("rank")
                    ),
                })
        return pd.DataFrame(rows)

    def qualifying_results(self, year: int, round: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch qualifying results for a season or specific round.

        Returns DataFrame with: year, round, race_name, driver_code, position, q1, q2, q3
        """
        path = f"{year}/qualifying" if round is None else f"{year}/{round}/qualifying"
        data = self._get_all_pages(path) if round is None else self._get(path)

        rows = []
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            rd = int(race.get("round", 0))
            race_name = race.get("raceName", "")
            for result in race.get("QualifyingResults", []):
                driver = result.get("Driver", {})
                rows.append({
                    "year": year,
                    "round": rd,
                    "race_name": race_name,
                    "driver_code": driver.get("code", ""),
                    "grid_position": int(result.get("position", 0)),
                    "q1": result.get("Q1", ""),
                    "q2": result.get("Q2", ""),
                    "q3": result.get("Q3", ""),
                })
        return pd.DataFrame(rows)

    # ── Bulk season pull ──────────────────────────────────────────────────────

    def bulk_race_results(
        self, years: list[int]
    ) -> pd.DataFrame:
        """Pull all race results for a list of seasons."""
        dfs = []
        for year in years:
            print(f"  → Fetching {year} race results from Jolpica...")
            try:
                df = self.race_results(year)
                dfs.append(df)
                print(f"     ✓ {len(df)} results")
            except Exception as exc:
                print(f"     ⚠️  {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def bulk_championship_standings(
        self, years: list[int], entity: str = "driver"
    ) -> pd.DataFrame:
        """
        Pull cumulative standings for every round across multiple seasons.

        entity: "driver" | "constructor"
        """
        dfs = []
        for year in years:
            print(f"  → Fetching {year} {entity} standings from Jolpica...")
            try:
                if entity == "driver":
                    df = self.all_driver_standings(year)
                else:
                    df = self.all_constructor_standings(year)
                dfs.append(df)
                print(f"     ✓ {len(df)} rows")
            except Exception as exc:
                print(f"     ⚠️  {exc}")
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None
