"""
Weather Fetcher
===============
Fetches race-weekend weather forecasts from the Open-Meteo API.

Open-Meteo is completely free and requires no API key:
  - Forecast API: https://api.open-meteo.com/v1/forecast  (up to 16 days ahead)
  - Archive API:  https://archive-api.open-meteo.com/v1/archive  (historical)

Usage:
    # Single day:
    enc = fetch_circuit_weather_enc("great_britain", date(2026, 7, 6))

    # Full weekend (all sessions):
    weather = fetch_weekend_weather("great_britain", date(2026, 7, 6),
                                    session_dates={"Qualifying": date(2026, 7, 5),
                                                   "Race": date(2026, 7, 6)})
    # Returns: {"Qualifying": {"enc": 1, "precip_mm": 4.2, "precip_prob": 80},
    #           "Race": {"enc": 0, "precip_mm": 0.0, "precip_prob": 15}}
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import requests


# ── Circuit coordinates ────────────────────────────────────────────────────────
# (latitude, longitude) for the pit/paddock area of each 2026 F1 venue.

CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    # Round 1 — Albert Park, Melbourne, Australia
    "australia":         (-37.8497,  144.9680),
    # Round 2 — Shanghai International Circuit, China
    "china":             ( 31.3389,  121.2198),
    # Round 3 — Suzuka International Racing Course, Japan
    "japan":             ( 34.8431,  136.5407),
    # Round 4 — Bahrain International Circuit, Sakhir
    "bahrain":           ( 26.0325,   50.5106),
    # Round 5 — Jeddah Corniche Circuit, Saudi Arabia
    "saudi_arabia":      ( 21.6319,   39.1044),
    # Round 6 — Miami International Autodrome, USA
    "miami":             ( 25.9581,  -80.2389),
    # Round 7 — Circuit Gilles Villeneuve, Montreal, Canada
    "canada":            ( 45.5000,  -73.5228),
    # Round 8 — Circuit de Monaco, Monte Carlo
    "monaco":            ( 43.7347,    7.4206),
    # Round 9 — Circuit de Barcelona-Catalunya, Spain
    "spain":             ( 41.5700,    2.2611),
    # Round 10 — Red Bull Ring, Spielberg, Austria
    "austria":           ( 47.2197,   14.7647),
    # Round 11 — Silverstone Circuit, UK
    "great_britain":     ( 52.0786,   -1.0169),
    # Round 12 — Circuit de Spa-Francorchamps, Belgium
    "belgium":           ( 50.4372,    5.9714),
    # Round 13 — Hungaroring, Budapest, Hungary
    "hungary":           ( 47.5789,   19.2486),
    # Round 14 — Circuit Zandvoort, Netherlands
    "netherlands":       ( 52.3888,    4.5422),
    # Round 15 — Autodromo Nazionale Monza, Italy
    "italy":             ( 45.6156,    9.2811),
    # Round 16 — Circuito de Madrid Jarama-RACE, Spain (new for 2026)
    "madrid":            ( 40.4168,   -3.7038),
    # Round 17 — Baku City Circuit, Azerbaijan
    "azerbaijan":        ( 40.3725,   49.8533),
    # Round 18 — Marina Bay Street Circuit, Singapore
    "singapore":         (  1.2914,  103.8639),
    # Round 19 — Circuit of the Americas, Austin, USA
    "united_states":     ( 30.1328,  -97.6411),
    # Round 20 — Autodromo Hermanos Rodriguez, Mexico City
    "mexico":            ( 19.4042,  -99.0907),
    # Round 21 — Autodromo Jose Carlos Pace (Interlagos), São Paulo, Brazil
    "brazil":            (-23.7036,  -46.6997),
    # Round 22 — Las Vegas Street Circuit, USA
    "las_vegas":         ( 36.1699, -115.1398),
    # Round 23 — Lusail International Circuit, Qatar
    "qatar":             ( 25.4900,   51.4536),
    # Round 24 — Yas Marina Circuit, Abu Dhabi
    "abu_dhabi":         ( 24.4672,   54.6031),
    # Pre-season test venues
    "test_barcelona_2026":  ( 41.5700,   2.2611),
    "test_bahrain_2026a":   ( 26.0325,  50.5106),
    "test_bahrain_2026b":   ( 26.0325,  50.5106),
}

# Wet thresholds
_WET_PROBABILITY_THRESHOLD = 50   # precipitation_probability_max >= 50 % → wet
_WET_PRECIPITATION_MM = 1.0       # precipitation_sum >= 1 mm → wet

_TIMEOUT = 10  # seconds


def _query_open_meteo(
    lat: float,
    lon: float,
    query_date: date,
    timeout: int = _TIMEOUT,
) -> dict:
    """
    Internal: call the appropriate Open-Meteo endpoint for a single day.

    Returns a dict with keys "precip_mm" and "precip_prob" (both may be None).
    Uses the archive API for past dates and the forecast API for future dates.
    """
    date_str = query_date.isoformat()
    today = date.today()

    if query_date <= today:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": "auto",
            "start_date": date_str,
            "end_date": date_str,
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_probability_max,precipitation_sum",
            "timezone": "auto",
            "start_date": date_str,
            "end_date": date_str,
        }

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    daily = data.get("daily", {})

    precip_prob = (daily.get("precipitation_probability_max") or [None])[0]
    precip_mm = (daily.get("precipitation_sum") or [None])[0]
    return {"precip_mm": precip_mm, "precip_prob": precip_prob}


def _enc_from_raw(precip_mm: Optional[float], precip_prob: Optional[float]) -> int:
    """Convert raw precipitation values to weather_enc (0=dry, 1=wet)."""
    if precip_prob is not None and precip_prob >= _WET_PROBABILITY_THRESHOLD:
        return 1
    if precip_mm is not None and precip_mm >= _WET_PRECIPITATION_MM:
        return 1
    return 0


def fetch_session_weather(
    lat: float,
    lon: float,
    session_date: date,
    timeout: int = _TIMEOUT,
) -> dict:
    """
    Fetch weather for a single session date.

    Returns:
        {
            "enc":        0 (dry) or 1 (wet),
            "precip_mm":  float | None,
            "precip_prob": float | None  (% probability, forecast only),
        }
    On error returns {"enc": None, "precip_mm": None, "precip_prob": None}.
    """
    try:
        raw = _query_open_meteo(lat, lon, session_date, timeout)
        enc = _enc_from_raw(raw["precip_mm"], raw["precip_prob"])
        return {"enc": enc, **raw}
    except Exception as exc:
        print(f"  ⚠️  weather_fetcher: {session_date} ({lat:.2f}, {lon:.2f}) — {exc}")
        return {"enc": None, "precip_mm": None, "precip_prob": None}


def fetch_weekend_weather(
    circuit_slug: str,
    race_date: date,
    session_dates: Optional[dict[str, date]] = None,
) -> dict[str, dict]:
    """
    Fetch weather for all sessions in a race weekend.

    Args:
        circuit_slug:  e.g. "great_britain"
        race_date:     the race Sunday
        session_dates: optional dict {session_name: date} for per-session lookup.
                       If None, only the race_date is fetched (key "Race").
                       If provided, each session_name is queried independently
                       (FP1 / FP2 / FP3 may be on Friday; Qualifying on Saturday).

    Returns:
        dict {session_name: {"enc": 0/1/None, "precip_mm": ..., "precip_prob": ...}}
        Always includes "Race" key; additional keys match session_dates.

    Example:
        {
            "FP1":        {"enc": 1, "precip_mm": 3.1, "precip_prob": 72},
            "Qualifying": {"enc": 1, "precip_mm": 1.8, "precip_prob": 65},
            "Race":       {"enc": 0, "precip_mm": 0.2, "precip_prob": 20},
        }
    """
    coords = CIRCUIT_COORDS.get(circuit_slug)
    if coords is None:
        print(f"  ⚠️  weather_fetcher: no coordinates for '{circuit_slug}'")
        return {"Race": {"enc": None, "precip_mm": None, "precip_prob": None}}

    lat, lon = coords
    result: dict[str, dict] = {}

    # Fetch per-session dates if provided
    sessions = session_dates or {}
    if "Race" not in sessions:
        sessions = {**sessions, "Race": race_date}

    for session_name, s_date in sessions.items():
        result[session_name] = fetch_session_weather(lat, lon, s_date)

    # Log summary
    conditions = ", ".join(
        f"{s}={'wet' if v['enc'] == 1 else ('dry' if v['enc'] == 0 else '?')}"
        for s, v in result.items()
    )
    print(f"  🌦  Weather [{circuit_slug}]: {conditions}")
    return result


def fetch_circuit_weather_enc(
    circuit_slug: str,
    race_date: date,
) -> Optional[int]:
    """
    Convenience: fetch race-day weather_enc (0=dry, 1=wet) for a circuit by slug.
    Returns None if the circuit is not in CIRCUIT_COORDS or API is unavailable.
    """
    coords = CIRCUIT_COORDS.get(circuit_slug)
    if coords is None:
        return None
    lat, lon = coords
    result = fetch_session_weather(lat, lon, race_date)
    return result["enc"]
