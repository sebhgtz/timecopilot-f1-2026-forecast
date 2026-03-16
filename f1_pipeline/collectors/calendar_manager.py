"""
F1 2026 Calendar Manager
=========================
Hard-coded 2026 F1 calendar with session schedules and sprint weekend flags.
Provides helpers to determine the next race, whether we're in a race weekend,
and what happened since the last race.

Includes check_for_updates() which queries the live FastF1 event schedule
and flags any races where dates have shifted from the hard-coded baseline.

Sources:
    - Official F1 calendar: https://www.formula1.com/en/racing/2026
    - Sprint calendar announcement (6 sprint weekends)
    - Pre-season testing: Barcelona Jan 26-30, Bahrain Feb 11-13 & 18-20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Session:
    """A single on-track session within a race weekend."""
    name: str          # "FP1", "FP2", "FP3", "Qualifying", "Sprint Qualifying", "Sprint", "Race"
    date: date
    # Approximate UTC start time (hour). None = unknown.
    utc_hour: Optional[int] = None

    @property
    def datetime_utc(self) -> Optional[datetime]:
        if self.utc_hour is None:
            return None
        return datetime(self.date.year, self.date.month, self.date.day,
                        self.utc_hour, 0, tzinfo=timezone.utc)


@dataclass
class CalendarDiff:
    """Describes a discrepancy between hard-coded and live calendar."""
    round: int
    race_name: str
    slug: str
    field: str            # "race_date", "session_FP1", etc.
    hardcoded_value: str
    live_value: str

    def __str__(self) -> str:
        return (
            f"Round {self.round} [{self.race_name}] — "
            f"{self.field}: hard-coded={self.hardcoded_value!r}, "
            f"live={self.live_value!r}"
        )


@dataclass
class Race:
    """A Grand Prix weekend (or pre-season test)."""
    round: int                        # 0 = pre-season test, 1..24 = race rounds
    name: str                         # e.g. "Australian Grand Prix"
    slug: str                         # e.g. "australia"
    circuit: str                      # e.g. "Albert Park Circuit"
    location: str                     # e.g. "Melbourne, Australia"
    circuit_type: str                 # "street" | "power" | "technical" | "balanced"
    race_date: date
    is_sprint: bool = False
    sessions: list[Session] = field(default_factory=list)

    @property
    def is_test(self) -> bool:
        return self.round == 0

    def get_session(self, name: str) -> Optional[Session]:
        for s in self.sessions:
            if s.name == name:
                return s
        return None

    def completed_sessions(self, as_of: Optional[date] = None) -> list[Session]:
        cutoff = as_of or date.today()
        return [s for s in self.sessions if s.date <= cutoff]

    def pending_sessions(self, as_of: Optional[date] = None) -> list[Session]:
        cutoff = as_of or date.today()
        return [s for s in self.sessions if s.date > cutoff]


# ── Session builders ──────────────────────────────────────────────────────────

def _make_standard_weekend(race_date: date) -> list[Session]:
    """FP1+FP2 on Friday, FP3+Quali on Saturday, Race on Sunday."""
    fri = race_date - timedelta(days=2)
    sat = race_date - timedelta(days=1)
    return [
        Session("FP1", fri, utc_hour=11),
        Session("FP2", fri, utc_hour=15),
        Session("FP3", sat, utc_hour=11),
        Session("Qualifying", sat, utc_hour=15),
        Session("Race", race_date, utc_hour=14),
    ]


def _make_sprint_weekend(race_date: date) -> list[Session]:
    """FP1+Sprint Qualifying on Friday, Sprint+Qualifying on Saturday, Race on Sunday."""
    fri = race_date - timedelta(days=2)
    sat = race_date - timedelta(days=1)
    return [
        Session("FP1", fri, utc_hour=11),
        Session("Sprint Qualifying", fri, utc_hour=15),
        Session("Sprint", sat, utc_hour=11),
        Session("Qualifying", sat, utc_hour=15),
        Session("Race", race_date, utc_hour=14),
    ]


def _make_test_sessions(start_date: date, days: int) -> list[Session]:
    return [
        Session(f"Test Day {i+1}", start_date + timedelta(days=i), utc_hour=8)
        for i in range(days)
    ]


# ── 2026 Active Driver Grid ───────────────────────────────────────────────────
# Maps 3-letter driver code → constructor for all 22 drivers on the 2026 grid.
# Source: formula1.com/en/teams (verified Feb 2026).
# Used to filter circuit series predictions to only active drivers.

DRIVERS_2026: dict[str, str] = {
    # Red Bull Racing
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",       # Isack Hadjar
    # Ferrari
    "HAM": "Ferrari",
    "LEC": "Ferrari",
    # McLaren
    "NOR": "McLaren",
    "PIA": "McLaren",
    # Mercedes
    "RUS": "Mercedes",
    "ANT": "Mercedes",              # Andrea Kimi Antonelli
    # Aston Martin
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    # Alpine
    "GAS": "Alpine",
    "COL": "Alpine",                # Franco Colapinto
    # Williams
    "SAI": "Williams",
    "ALB": "Williams",
    # Haas F1 Team
    "BEA": "Haas",
    "OCO": "Haas",
    # Audi (née Sauber)
    "HUL": "Audi",
    "BOR": "Audi",                  # Gabriel Bortoleto
    # Racing Bulls (née RB / VCARB)
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",          # Arvid Lindblad (no prior F1 race history)
    # Cadillac (new team for 2026)
    "BOT": "Cadillac",
    "PER": "Cadillac",
}

# Drivers with no prior F1 race history in Jolpica — performance seeded from
# pre-season testing pace when building circuit series.
ROOKIES_2026: set[str] = {"LIN"}

# Drivers who debuted in 2025 and are in their second F1 season in 2026.
# They have exactly 1 year of historical data — enough for championship forecasting
# but insufficient for per-circuit forecasts (need 3+ data points per circuit).
# tier: "front" = top-3 constructor expected, "midfield" = 4th–7th, "back" = 8th–10th.
# Used by circuit_series.py (synthetic row backfill) and forecasting LLM queries.
SECOND_YEAR_2026: dict[str, str] = {
    "ANT": "front",    # Andrea Kimi Antonelli — Mercedes (won China 2026 R2)
    "BOR": "midfield", # Gabriel Bortoleto — Audi
    "HAD": "front",    # Isack Hadjar — Red Bull Racing
    "BEA": "back",     # Oliver Bearman — Haas
}


# ── 2026 Calendar ─────────────────────────────────────────────────────────────

CALENDAR_2026: list[Race] = [
    # ── Pre-season tests ──────────────────────────────────────────────────────
    Race(
        round=0, name="Pre-Season Test 1", slug="test_barcelona_2026",
        circuit="Circuit de Barcelona-Catalunya", location="Barcelona, Spain",
        circuit_type="technical", race_date=date(2026, 1, 28),
        sessions=_make_test_sessions(date(2026, 1, 26), 5),
    ),
    Race(
        round=0, name="Pre-Season Test 2", slug="test_bahrain_2026a",
        circuit="Bahrain International Circuit", location="Sakhir, Bahrain",
        circuit_type="balanced", race_date=date(2026, 2, 12),
        sessions=_make_test_sessions(date(2026, 2, 11), 3),
    ),
    Race(
        round=0, name="Pre-Season Test 3", slug="test_bahrain_2026b",
        circuit="Bahrain International Circuit", location="Sakhir, Bahrain",
        circuit_type="balanced", race_date=date(2026, 2, 19),
        sessions=_make_test_sessions(date(2026, 2, 18), 3),
    ),

    # ── Round 1: Australia ────────────────────────────────────────────────────
    Race(
        round=1, name="Australian Grand Prix", slug="australia",
        circuit="Albert Park Circuit", location="Melbourne, Australia",
        circuit_type="street", race_date=date(2026, 3, 8),
        sessions=_make_standard_weekend(date(2026, 3, 8)),
    ),
    # ── Round 2: China (SPRINT) ───────────────────────────────────────────────
    Race(
        round=2, name="Chinese Grand Prix", slug="china",
        circuit="Shanghai International Circuit", location="Shanghai, China",
        circuit_type="balanced", race_date=date(2026, 3, 15),
        is_sprint=True, sessions=_make_sprint_weekend(date(2026, 3, 15)),
    ),
    # ── Round 3: Japan ────────────────────────────────────────────────────────
    Race(
        round=3, name="Japanese Grand Prix", slug="japan",
        circuit="Suzuka International Racing Course", location="Suzuka, Japan",
        circuit_type="technical", race_date=date(2026, 3, 29),
        sessions=_make_standard_weekend(date(2026, 3, 29)),
    ),
    # ── Round 4: Bahrain ──────────────────────────────────────────────────────
    Race(
        round=4, name="Bahrain Grand Prix", slug="bahrain",
        circuit="Bahrain International Circuit", location="Sakhir, Bahrain",
        circuit_type="balanced", race_date=date(2026, 4, 12),
        sessions=_make_standard_weekend(date(2026, 4, 12)),
    ),
    # ── Round 5: Saudi Arabia ─────────────────────────────────────────────────
    Race(
        round=5, name="Saudi Arabian Grand Prix", slug="saudi_arabia",
        circuit="Jeddah Corniche Circuit", location="Jeddah, Saudi Arabia",
        circuit_type="street", race_date=date(2026, 4, 19),
        sessions=_make_standard_weekend(date(2026, 4, 19)),
    ),
    # ── Round 6: Miami (SPRINT) ───────────────────────────────────────────────
    Race(
        round=6, name="Miami Grand Prix", slug="miami",
        circuit="Miami International Autodrome", location="Miami, USA",
        circuit_type="street", race_date=date(2026, 5, 3),
        is_sprint=True, sessions=_make_sprint_weekend(date(2026, 5, 3)),
    ),
    # ── Round 7: Canada (SPRINT) ──────────────────────────────────────────────
    Race(
        round=7, name="Canadian Grand Prix", slug="canada",
        circuit="Circuit Gilles Villeneuve", location="Montreal, Canada",
        circuit_type="street", race_date=date(2026, 5, 24),
        is_sprint=True, sessions=_make_sprint_weekend(date(2026, 5, 24)),
    ),
    # ── Round 8: Monaco ───────────────────────────────────────────────────────
    Race(
        round=8, name="Monaco Grand Prix", slug="monaco",
        circuit="Circuit de Monaco", location="Monte Carlo, Monaco",
        circuit_type="street", race_date=date(2026, 6, 7),
        sessions=_make_standard_weekend(date(2026, 6, 7)),
    ),
    # ── Round 9: Spain ────────────────────────────────────────────────────────
    Race(
        round=9, name="Spanish Grand Prix", slug="spain",
        circuit="Circuit de Barcelona-Catalunya", location="Barcelona, Spain",
        circuit_type="technical", race_date=date(2026, 6, 14),
        sessions=_make_standard_weekend(date(2026, 6, 14)),
    ),
    # ── Round 10: Austria ─────────────────────────────────────────────────────
    Race(
        round=10, name="Austrian Grand Prix", slug="austria",
        circuit="Red Bull Ring", location="Spielberg, Austria",
        circuit_type="power", race_date=date(2026, 6, 28),
        sessions=_make_standard_weekend(date(2026, 6, 28)),
    ),
    # ── Round 11: Great Britain (SPRINT) ─────────────────────────────────────
    Race(
        round=11, name="British Grand Prix", slug="great_britain",
        circuit="Silverstone Circuit", location="Silverstone, UK",
        circuit_type="balanced", race_date=date(2026, 7, 5),
        is_sprint=True, sessions=_make_sprint_weekend(date(2026, 7, 5)),
    ),
    # ── Round 12: Belgium ─────────────────────────────────────────────────────
    Race(
        round=12, name="Belgian Grand Prix", slug="belgium",
        circuit="Circuit de Spa-Francorchamps", location="Spa, Belgium",
        circuit_type="power", race_date=date(2026, 7, 19),
        sessions=_make_standard_weekend(date(2026, 7, 19)),
    ),
    # ── Round 13: Hungary ─────────────────────────────────────────────────────
    Race(
        round=13, name="Hungarian Grand Prix", slug="hungary",
        circuit="Hungaroring", location="Budapest, Hungary",
        circuit_type="technical", race_date=date(2026, 7, 26),
        sessions=_make_standard_weekend(date(2026, 7, 26)),
    ),
    # ── Round 14: Netherlands (SPRINT) ───────────────────────────────────────
    Race(
        round=14, name="Dutch Grand Prix", slug="netherlands",
        circuit="Circuit Zandvoort", location="Zandvoort, Netherlands",
        circuit_type="technical", race_date=date(2026, 8, 23),
        is_sprint=True, sessions=_make_sprint_weekend(date(2026, 8, 23)),
    ),
    # ── Round 15: Italy ───────────────────────────────────────────────────────
    Race(
        round=15, name="Italian Grand Prix", slug="italy",
        circuit="Autodromo Nazionale Monza", location="Monza, Italy",
        circuit_type="power", race_date=date(2026, 9, 6),
        sessions=_make_standard_weekend(date(2026, 9, 6)),
    ),
    # ── Round 16: Madrid ──────────────────────────────────────────────────────
    Race(
        round=16, name="Madrid Grand Prix", slug="madrid",
        circuit="Circuito de Madrid Jarama-RACE", location="Madrid, Spain",
        circuit_type="balanced", race_date=date(2026, 9, 13),
        sessions=_make_standard_weekend(date(2026, 9, 13)),
    ),
    # ── Round 17: Azerbaijan ──────────────────────────────────────────────────
    Race(
        round=17, name="Azerbaijan Grand Prix", slug="azerbaijan",
        circuit="Baku City Circuit", location="Baku, Azerbaijan",
        circuit_type="street", race_date=date(2026, 9, 26),
        sessions=_make_standard_weekend(date(2026, 9, 26)),
    ),
    # ── Round 18: Singapore (SPRINT) ─────────────────────────────────────────
    Race(
        round=18, name="Singapore Grand Prix", slug="singapore",
        circuit="Marina Bay Street Circuit", location="Singapore",
        circuit_type="street", race_date=date(2026, 10, 11),
        is_sprint=True, sessions=_make_sprint_weekend(date(2026, 10, 11)),
    ),
    # ── Round 19: United States ───────────────────────────────────────────────
    Race(
        round=19, name="United States Grand Prix", slug="united_states",
        circuit="Circuit of the Americas", location="Austin, USA",
        circuit_type="balanced", race_date=date(2026, 10, 25),
        sessions=_make_standard_weekend(date(2026, 10, 25)),
    ),
    # ── Round 20: Mexico ──────────────────────────────────────────────────────
    Race(
        round=20, name="Mexico City Grand Prix", slug="mexico",
        circuit="Autodromo Hermanos Rodriguez", location="Mexico City, Mexico",
        circuit_type="balanced", race_date=date(2026, 11, 1),
        sessions=_make_standard_weekend(date(2026, 11, 1)),
    ),
    # ── Round 21: Brazil ──────────────────────────────────────────────────────
    Race(
        round=21, name="São Paulo Grand Prix", slug="brazil",
        circuit="Autodromo Jose Carlos Pace (Interlagos)", location="São Paulo, Brazil",
        circuit_type="balanced", race_date=date(2026, 11, 8),
        sessions=_make_standard_weekend(date(2026, 11, 8)),
    ),
    # ── Round 22: Las Vegas ───────────────────────────────────────────────────
    Race(
        round=22, name="Las Vegas Grand Prix", slug="las_vegas",
        circuit="Las Vegas Street Circuit", location="Las Vegas, USA",
        circuit_type="street", race_date=date(2026, 11, 21),
        sessions=_make_standard_weekend(date(2026, 11, 21)),
    ),
    # ── Round 23: Qatar ───────────────────────────────────────────────────────
    Race(
        round=23, name="Qatar Grand Prix", slug="qatar",
        circuit="Lusail International Circuit", location="Lusail, Qatar",
        circuit_type="balanced", race_date=date(2026, 11, 29),
        sessions=_make_standard_weekend(date(2026, 11, 29)),
    ),
    # ── Round 24: Abu Dhabi ───────────────────────────────────────────────────
    Race(
        round=24, name="Abu Dhabi Grand Prix", slug="abu_dhabi",
        circuit="Yas Marina Circuit", location="Abu Dhabi, UAE",
        circuit_type="street", race_date=date(2026, 12, 6),
        sessions=_make_standard_weekend(date(2026, 12, 6)),
    ),
]

# Lookups
_BY_SLUG: dict[str, Race] = {r.slug: r for r in CALENDAR_2026}
_BY_ROUND: dict[int, Race] = {r.round: r for r in CALENDAR_2026 if r.round > 0}


# ── CalendarManager ───────────────────────────────────────────────────────────

class CalendarManager:
    """Query the 2026 F1 calendar and detect live date changes."""

    CIRCUIT_TYPE_ENC = {"street": 1, "power": 2, "balanced": 3, "technical": 4}

    def __init__(self, calendar: list[Race] = None):
        self._calendar = calendar or CALENDAR_2026

    # ── Access ────────────────────────────────────────────────────────────────

    def all_races(self, include_tests: bool = False) -> list[Race]:
        return [r for r in self._calendar if include_tests or r.round > 0]

    def get_race(self, slug: str) -> Race:
        race = _BY_SLUG.get(slug)
        if race is None:
            raise ValueError(
                f"Unknown race slug '{slug}'. "
                f"Valid slugs: {sorted(_BY_SLUG.keys())}"
            )
        return race

    def get_race_by_round(self, round_num: int) -> Race:
        race = _BY_ROUND.get(round_num)
        if race is None:
            raise ValueError(f"Unknown round {round_num}.")
        return race

    # ── Temporal queries ──────────────────────────────────────────────────────

    def next_race(self, as_of: Optional[date] = None) -> Optional[Race]:
        """Return the next race that hasn't started yet."""
        cutoff = as_of or date.today()
        upcoming = [r for r in self.all_races() if r.race_date >= cutoff]
        return upcoming[0] if upcoming else None

    def last_race(self, as_of: Optional[date] = None) -> Optional[Race]:
        """Return the most recently completed race."""
        cutoff = as_of or date.today()
        past = [r for r in self.all_races() if r.race_date < cutoff]
        return past[-1] if past else None

    def current_race_weekend(self, as_of: Optional[date] = None) -> Optional[Race]:
        """Return the race whose weekend window includes today (Fri–Mon).

        The window extends to race_date + 1 so that Monday's 09:00 UTC run
        can detect that the race completed on Sunday and fire the post-race
        championship update.
        """
        cutoff = as_of or date.today()
        for race in self.all_races():
            weekend_start = race.race_date - timedelta(days=2)
            if weekend_start <= cutoff <= race.race_date + timedelta(days=1):
                return race
        return None

    def races_since(self, after_race_slug: str, as_of: Optional[date] = None) -> list[Race]:
        """
        Return all races that occurred between after_race_slug (exclusive)
        and as_of date (inclusive).
        """
        cutoff = as_of or date.today()
        anchor = self.get_race(after_race_slug)
        return [
            r for r in self.all_races()
            if r.race_date > anchor.race_date and r.race_date <= cutoff
        ]

    def remaining_races(self, as_of: Optional[date] = None) -> list[Race]:
        cutoff = as_of or date.today()
        return [r for r in self.all_races() if r.race_date > cutoff]

    def completed_races(self, as_of: Optional[date] = None) -> list[Race]:
        cutoff = as_of or date.today()
        return [r for r in self.all_races() if r.race_date < cutoff]

    def sprint_weekends(self) -> list[Race]:
        return [r for r in self.all_races() if r.is_sprint]

    def is_sprint_weekend(self, slug: str) -> bool:
        return self.get_race(slug).is_sprint

    def available_sessions(self, slug: str, as_of: Optional[date] = None) -> list[str]:
        """Sessions from this race weekend that have already completed."""
        race = self.get_race(slug)
        return [s.name for s in race.completed_sessions(as_of)]

    def circuit_type_enc(self, slug: str) -> int:
        race = self.get_race(slug)
        return self.CIRCUIT_TYPE_ENC.get(race.circuit_type, 3)

    def preseason_tests(self) -> list[Race]:
        return [r for r in self._calendar if r.round == 0]

    # ── Live calendar update check ────────────────────────────────────────────

    def check_for_updates(self, verbose: bool = True) -> list[CalendarDiff]:
        """
        Compare the hard-coded 2026 calendar against the live FastF1 event
        schedule and report any discrepancies in race dates or session dates.

        Requires fastf1 to be installed. Does NOT modify the in-memory calendar;
        it only reports differences so you can decide whether to update the
        hard-coded values.

        Returns:
            List of CalendarDiff objects (empty if everything matches).
        """
        try:
            import fastf1
        except ImportError:
            print("fastf1 not installed — cannot check for calendar updates.")
            return []

        if verbose:
            print("Fetching live 2026 F1 event schedule from FastF1...")

        try:
            live_schedule = fastf1.get_event_schedule(2026, include_testing=True)
        except Exception as exc:
            print(f"Could not fetch live schedule: {exc}")
            return []

        diffs: list[CalendarDiff] = []

        # Build a mapping from round number to live event row
        for _, live_event in live_schedule.iterrows():
            round_num = int(live_event.get("RoundNumber", 0))
            if round_num not in _BY_ROUND and round_num != 0:
                continue

            # Try to find matching hard-coded race by round number
            hc_race = _BY_ROUND.get(round_num)
            if hc_race is None:
                continue

            # Compare race date
            live_race_date_raw = live_event.get("EventDate") or live_event.get("Session5Date")
            if live_race_date_raw is not None:
                try:
                    live_race_date = _to_date(live_race_date_raw)
                    if live_race_date != hc_race.race_date:
                        diffs.append(CalendarDiff(
                            round=round_num,
                            race_name=hc_race.name,
                            slug=hc_race.slug,
                            field="race_date",
                            hardcoded_value=hc_race.race_date.isoformat(),
                            live_value=live_race_date.isoformat(),
                        ))
                except Exception:
                    pass

            # Compare session dates using FastF1's SessionNDate + SessionNName columns
            # so we get an exact session name match instead of fuzzy substring matching.
            for n in range(1, 6):
                date_col = f"Session{n}Date"
                name_col = f"Session{n}"   # FastF1 also exposes SessionN (the name)
                live_val = live_event.get(date_col)
                if live_val is None:
                    continue
                try:
                    live_session_date = _to_date(live_val)
                except Exception:
                    continue

                # Prefer exact session name from FastF1 if available
                live_name = str(live_event.get(name_col, "")).strip()
                hc_session = _find_session_exact(hc_race, live_name) if live_name else None

                if hc_session and hc_session.date != live_session_date:
                    diffs.append(CalendarDiff(
                        round=round_num,
                        race_name=hc_race.name,
                        slug=hc_race.slug,
                        field=f"session_{live_name or date_col}",
                        hardcoded_value=hc_session.date.isoformat(),
                        live_value=live_session_date.isoformat(),
                    ))

        if verbose:
            if not diffs:
                print("✅  Hard-coded calendar matches live FastF1 schedule — no changes detected.")
            else:
                print(f"⚠️  {len(diffs)} discrepancy(ies) found between hard-coded and live calendar:\n")
                for d in diffs:
                    print(f"  • {d}")
                print(
                    "\nUpdate the hard-coded CALENDAR_2026 in calendar_manager.py "
                    "if these changes are official."
                )

        return diffs

    # ── Pretty print ──────────────────────────────────────────────────────────

    def print_calendar(self) -> None:
        print(f"\n{'Rd':>3}  {'Race':<35} {'Date':<12} {'Sprint':^6}")
        print("─" * 60)
        for r in self.all_races():
            sprint = "✓" if r.is_sprint else ""
            print(f"{r.round:>3}  {r.name:<35} {r.race_date.isoformat():<12} {sprint:^6}")
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_date(val) -> date:
    """Convert a pandas Timestamp, datetime, or ISO string to a date."""
    if hasattr(val, "date"):
        return val.date()
    return date.fromisoformat(str(val)[:10])


def _find_session(race: Race, label: str) -> Optional[Session]:
    """Find the first session whose name contains any word from label (fuzzy)."""
    words = label.lower().split(" / ")
    for s in race.sessions:
        name_lower = s.name.lower()
        if any(w in name_lower for w in words):
            return s
    return None


def _find_session_exact(race: Race, live_name: str) -> Optional[Session]:
    """
    Find a session by exact name match (case-insensitive) against the live FastF1
    session name. Falls back to fuzzy matching if no exact match found.

    FastF1 session names: 'Practice 1', 'Practice 2', 'Practice 3',
                          'Sprint Qualifying', 'Sprint', 'Qualifying', 'Race'
    Our session names:    'FP1', 'FP2', 'FP3',
                          'Sprint Qualifying', 'Sprint', 'Qualifying', 'Race'
    """
    _LIVE_TO_HC = {
        "practice 1": "fp1",
        "practice 2": "fp2",
        "practice 3": "fp3",
        "sprint qualifying": "sprint qualifying",
        "sprint shootout": "sprint qualifying",
        "sprint": "sprint",
        "qualifying": "qualifying",
        "race": "race",
    }
    target = _LIVE_TO_HC.get(live_name.lower(), live_name.lower())
    for s in race.sessions:
        if s.name.lower() == target:
            return s
    # Fallback to fuzzy
    return _find_session(race, live_name)
