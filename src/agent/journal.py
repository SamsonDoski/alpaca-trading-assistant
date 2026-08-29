"""The record of what the agent did, and what it decided not to do.

The broker knows what is open right now. It does not know what closed last
Tuesday, why a symbol was skipped, or what the model was thinking when it
declined. That history lives here.

**Two tables, because there are two different kinds of fact.**

`events` records things that changed the account -- a position opened, a position
closed. That is the trade log, and it is what the cooldown rule and the daily
totals read.

`decisions` records every symbol the agent looked at on every pass, whether or
not anything happened. Most rows are refusals: the model had no view, a gate
said no, a spread was too wide. Storing them is the point. An agent that only
logs its trades leaves you unable to answer the most useful question about it --
*what did it consider, and why did it say no?* -- and that question is most of
what makes automated trading reviewable rather than merely observable.

Append-only, and deliberately kept separate from any cache. This is the audit
trail; mixing it with data that gets pruned or refetched would put the record of
real orders at risk of a routine cleanup.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    day         TEXT NOT NULL,
    action      TEXT NOT NULL,   -- opened | closed | alert
    underlying  TEXT NOT NULL,
    symbol      TEXT NOT NULL,   -- the OCC contract symbol
    detail      TEXT NOT NULL,
    pnl         REAL             -- set on close, else NULL
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day DESC);
CREATE INDEX IF NOT EXISTS idx_events_action ON events(action);

CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    day         TEXT NOT NULL,
    underlying  TEXT NOT NULL,
    approved    INTEGER NOT NULL,
    reason      TEXT NOT NULL,
    direction   TEXT,
    confidence  REAL,
    rationale   TEXT,
    thinking    TEXT,
    gate_trace  TEXT             -- JSON: [[gate name, decision, reason], ...]
);
CREATE INDEX IF NOT EXISTS idx_decisions_day ON decisions(day DESC);
"""


@dataclass(frozen=True, slots=True)
class Event:
    at: str
    day: str
    action: str
    underlying: str
    symbol: str
    detail: str
    pnl: float | None


@dataclass(frozen=True, slots=True)
class DecisionRow:
    at: str
    underlying: str
    approved: bool
    reason: str
    direction: str | None
    confidence: float | None
    rationale: str | None
    thinking: str | None


class Journal:
    """Append-only log of everything the agent did and considered."""

    def __init__(self, path: Path | str = Path("state/journal.db")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    # -- writing -----------------------------------------------------------

    def record(self, action: str, underlying: str, symbol: str, detail: str,
               *, pnl: float | None = None) -> None:
        now = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (at, day, action, underlying, symbol, detail, pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (now.isoformat(), now.date().isoformat(), action,
                 underlying, symbol, detail, pnl),
            )

    def record_decision(self, underlying: str, *, approved: bool, reason: str,
                        proposal=None, trace=()) -> None:
        """Record one symbol's outcome for one pass, with the reasoning behind it.

        The gate trace is stored as JSON rather than as rows in a third table.
        It is only ever read back whole, for one decision at a time, so a
        normalised schema would buy nothing and cost a join.
        """
        now = datetime.now(UTC)
        encoded = json.dumps([
            [name, verdict.decision.value, verdict.reason] for name, verdict in trace
        ])
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO decisions (at, day, underlying, approved, reason, "
                "direction, confidence, rationale, thinking, gate_trace) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now.isoformat(), now.date().isoformat(), underlying,
                 1 if approved else 0, reason,
                 proposal.direction.value if proposal else None,
                 proposal.confidence if proposal else None,
                 proposal.rationale if proposal else None,
                 proposal.thinking_summary if proposal else None,
                 encoded),
            )

    # -- reading -----------------------------------------------------------

    def cooling_off(self, *, within_days: int, as_of: date | None = None) -> dict[str, int]:
        """Underlyings still inside their post-stop-loss cooldown.

        Returns days remaining per underlying, which is what the gate wants --
        it can then say "2 days left" rather than making the caller subtract
        dates to find out.

        Only stop-loss closes count. A take-profit close means the reasoning
        worked, and re-entering after a win is not the behaviour this rule
        exists to prevent.
        """
        today = as_of or datetime.now(UTC).date()
        earliest = (today - timedelta(days=within_days)).isoformat()

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT underlying, MAX(day) AS day FROM events "
                "WHERE action = 'closed' AND detail LIKE ? AND day >= ? "
                "GROUP BY underlying",
                (f"{_STOP_PREFIX}%", earliest),
            ).fetchall()

        remaining: dict[str, int] = {}
        for row in rows:
            elapsed = (today - date.fromisoformat(row["day"])).days
            left = within_days - elapsed
            if left > 0:
                remaining[row["underlying"]] = left
        return remaining

    def counts_for_day(self, day: date | None = None) -> dict[str, int]:
        target = (day or datetime.now(UTC).date()).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT action, COUNT(*) AS n FROM events WHERE day = ? GROUP BY action",
                (target,),
            ).fetchall()
        return {r["action"]: r["n"] for r in rows}

    def realized_for_day(self, day: date | None = None) -> float:
        target = (day or datetime.now(UTC).date()).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS total FROM events "
                "WHERE day = ? AND action = 'closed'",
                (target,),
            ).fetchone()
        return float(row["total"])

    def recent(self, limit: int = 50) -> list[Event]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT at, day, action, underlying, symbol, detail, pnl FROM events "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [Event(**dict(r)) for r in rows]

    def decisions_for_day(self, day: date | None = None,
                          limit: int = 200) -> list[DecisionRow]:
        target = (day or datetime.now(UTC).date()).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT at, underlying, approved, reason, direction, confidence, "
                "rationale, thinking FROM decisions WHERE day = ? "
                "ORDER BY id DESC LIMIT ?", (target, limit)).fetchall()
        return [DecisionRow(at=r["at"], underlying=r["underlying"],
                            approved=bool(r["approved"]), reason=r["reason"],
                            direction=r["direction"], confidence=r["confidence"],
                            rationale=r["rationale"], thinking=r["thinking"])
                for r in rows]

    def summary(self) -> dict:
        """Lifetime totals."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE action='closed') AS closed, "
                "  COUNT(*) FILTER (WHERE action='closed' AND pnl > 0) AS wins, "
                "  COALESCE(SUM(pnl) FILTER (WHERE action='closed'), 0) AS total_pnl "
                "FROM events").fetchone()
        closed = row["closed"] or 0
        wins = row["wins"] or 0
        return {"closed": closed, "wins": wins, "losses": closed - wins,
                "win_rate": (wins / closed) if closed else 0.0,
                "total_pnl": float(row["total_pnl"])}


# The detail text of a stop-loss close begins with this, and `cooling_off`
# matches on it. Named rather than repeated so the writer and the reader cannot
# drift apart -- a cooldown that silently stops matching would be invisible.
_STOP_PREFIX = "stop loss"
