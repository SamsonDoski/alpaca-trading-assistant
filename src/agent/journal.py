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
    action      TEXT NOT NULL,   -- ordered | opened | unfilled | sold | closed | alert
                                 -- ordered/sold = sent to the broker
                                 -- opened/closed = the broker confirmed it
                                 -- Only `closed` carries realised P&L, because
                                 -- only it happens exactly once per trade.
    underlying  TEXT NOT NULL,
    symbol      TEXT NOT NULL,   -- the OCC contract symbol
    detail      TEXT NOT NULL,
    pnl         REAL             -- set on close, else NULL
);
CREATE INDEX IF NOT EXISTS idx_events_day ON events(day DESC);
CREATE INDEX IF NOT EXISTS idx_events_action ON events(action);

-- What the world looked like when we opened a position.
--
-- The broker knows what we hold and what we paid. It does not know what the
-- UNDERLYING was worth at that moment, and without that number an
-- underlying-keyed stop is impossible: you cannot say "exit if the stock falls
-- 4% from entry" if you never wrote down where entry was.
--
-- Keyed by contract symbol and deleted on close, so this table holds only live
-- positions -- it is working state, not history. History lives in `events`.
CREATE TABLE IF NOT EXISTS holdings (
    occ_symbol   TEXT PRIMARY KEY,
    underlying   TEXT NOT NULL,
    opened_at    TEXT NOT NULL,
    direction    TEXT NOT NULL,   -- up | down, the thesis being expressed
    entry_spot   REAL NOT NULL,   -- the underlying's price when we bought
    entry_premium REAL NOT NULL,
    stop_spot    REAL NOT NULL,   -- underlying level that invalidates the thesis
    target_spot  REAL NOT NULL
);

-- The numbers behind an order, written when it is sent.
--
-- `decisions` already records WHY something was refused, which is what you want
-- while watching a pass. It records nothing measurable about what was ALLOWED,
-- and that turned out to be the binding constraint on improving this system.
--
-- Pairing the model's stated confidence against outcomes found a real effect
-- worth roughly $10,000 -- and that analysis was only possible because
-- confidence happened to be stored. The same question could not be asked of the
-- spread paid, the implied volatility, the delta or the days to expiry, because
-- none of them survived the pass that used them.
--
-- So every number a gate judged is kept, whether or not anything reads it
-- today. The cost is a row per order; the alternative is discovering in a month
-- that the one column needed was the one not written.
CREATE TABLE IF NOT EXISTS entry_metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    at             TEXT NOT NULL,
    day            TEXT NOT NULL,
    occ_symbol     TEXT NOT NULL,
    underlying     TEXT NOT NULL,
    direction      TEXT NOT NULL,
    confidence     REAL,
    quantity       INTEGER,
    limit_price    REAL,       -- what we asked to pay, per share
    cost           REAL,       -- the whole position
    spot           REAL,       -- the underlying when the decision was made
    strike         REAL,
    days_to_expiry INTEGER,
    delta          REAL,
    implied_vol    REAL,
    realized_vol   REAL,
    iv_to_realized REAL,       -- what premium_richness judged
    spread_pct     REAL,       -- what spread_width judged
    daily_decay    REAL,       -- what decay_burden judged
    open_interest  INTEGER,
    atr_pct        REAL,
    range_position REAL,       -- 0 = at the 82-day low, 1 = at the high
    change_5d      REAL,
    change_20d     REAL,
    change_60d     REAL,
    stop_spot      REAL,
    target_spot    REAL
);
CREATE INDEX IF NOT EXISTS idx_metrics_symbol ON entry_metrics(occ_symbol);

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
class Holding:
    """Where the underlying stood when a position was opened, and the levels
    that decide its fate."""

    occ_symbol: str
    underlying: str
    opened_at: str
    direction: str
    entry_spot: float
    entry_premium: float
    stop_spot: float
    target_spot: float

    def breached_stop(self, spot: float) -> bool:
        """Whether the underlying has moved through the level that invalidates
        the thesis. Direction decides which side counts as through."""
        return spot <= self.stop_spot if self.direction == "up" else spot >= self.stop_spot

    def reached_target(self, spot: float) -> bool:
        return spot >= self.target_spot if self.direction == "up" else spot <= self.target_spot

    def move_pct(self, spot: float) -> float:
        """How far the underlying has travelled from entry, signed so that
        positive always means 'in favour of the thesis'."""
        if self.entry_spot <= 0:
            return 0.0
        raw = (spot - self.entry_spot) / self.entry_spot
        return raw if self.direction == "up" else -raw


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

    def __init__(self, path: Path | str | None = None) -> None:
        # Defaults to the active profile's own journal. Two accounts sharing one
        # would produce a cooldown on the second because the first stopped out,
        # and `holdings` rows keyed by contract symbol would collide the moment
        # both bought the same option.
        from agent import profile
        self.path = Path(path) if path is not None else profile.journal_path()
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

    def record_entry_metrics(self, **fields) -> None:
        """Store the measured inputs behind one order.

        Takes keywords rather than a long positional signature so that adding a
        column later does not break the call site, and so a caller that cannot
        compute something simply omits it rather than passing a placeholder that
        later reads as a measurement.
        """
        now = datetime.now(UTC)
        fields = {"at": now.isoformat(), "day": now.date().isoformat(), **fields}
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO entry_metrics ({columns}) VALUES ({placeholders})",
                tuple(fields.values()))

    def entry_metrics(self, occ_symbol: str | None = None) -> list[dict]:
        """Every recorded order, or just the ones for a contract."""
        sql = "SELECT * FROM entry_metrics"
        args: tuple = ()
        if occ_symbol:
            sql += " WHERE occ_symbol = ?"
            args = (occ_symbol,)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY id", args)]

    def open_holding(self, *, occ_symbol: str, underlying: str, direction: str,
                     entry_spot: float, entry_premium: float,
                     stop_spot: float, target_spot: float) -> None:
        """Record where the underlying stood when a position was opened.

        `REPLACE` rather than `INSERT` so a re-opened contract overwrites the
        stale row instead of failing. The alternative -- a duplicate-key error
        inside the trading loop -- would turn a harmless edge case into a failed
        pass.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO holdings (occ_symbol, underlying, opened_at, "
                "direction, entry_spot, entry_premium, stop_spot, target_spot) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (occ_symbol, underlying, datetime.now(UTC).isoformat(), direction,
                 entry_spot, entry_premium, stop_spot, target_spot),
            )

    def close_holding(self, occ_symbol: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM holdings WHERE occ_symbol = ?", (occ_symbol,))

    def holding(self, occ_symbol: str) -> Holding | None:
        """The recorded entry conditions for one position, if we opened it.

        Returns None for anything this agent did not open -- a position from an
        earlier system, or one placed by hand. Those are not errors: the exit
        logic falls back to a premium-based stop for them, which is the only
        rule available when there is no entry level to reason from.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT occ_symbol, underlying, opened_at, direction, entry_spot, "
                "entry_premium, stop_spot, target_spot FROM holdings "
                "WHERE occ_symbol = ?", (occ_symbol,)).fetchone()
        return Holding(**dict(row)) if row else None

    def open_holdings(self) -> list[Holding]:
        """Every position we believe we are holding.

        The table is working state rather than history, so this is what the
        agent thinks the book is. Comparing it against what the broker actually
        reports is how a position closed by someone else gets noticed.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT occ_symbol, underlying, opened_at, direction, entry_spot, "
                "entry_premium, stop_spot, target_spot FROM holdings").fetchall()
        return [Holding(**dict(row)) for row in rows]

    def unbooked_sales(self) -> list[dict]:
        """Sell orders that have gone out but whose trade is not booked yet.

        A contract can be sold more than once before it leaves the book -- a
        patient limit that does not fill is cancelled and replaced the next
        pass. Only the LATEST sale is returned for each contract, and only while
        no `closed` row has been written since it, so one trade books once no
        matter how many orders it took.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.symbol, s.underlying, s.detail, s.pnl "
                "FROM events s "
                "WHERE s.action = 'sold' "
                "  AND s.id = (SELECT MAX(id) FROM events "
                "              WHERE action = 'sold' AND symbol = s.symbol) "
                "  AND NOT EXISTS (SELECT 1 FROM events c "
                "                  WHERE c.action = 'closed' "
                "                    AND c.symbol = s.symbol AND c.id > s.id)"
            ).fetchall()
        return [dict(r) for r in rows]

    def unannounced_fills(self, held: set[str]) -> list[Holding]:
        """Positions we hold whose fill has not been recorded yet.

        An order and a position are different events. The agent records
        `ordered` when it sends one, and `opened` only once the broker reports
        the contract in the book. This finds the gap between the two: a holdings
        row for a contract the broker now shows as held, with no `opened` event
        written since that row was created.

        Compared against the row's own timestamp rather than "ever", so buying
        the same contract a second time is announced again instead of being
        mistaken for the first fill.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT h.occ_symbol, h.underlying, h.opened_at, h.direction, "
                "h.entry_spot, h.entry_premium, h.stop_spot, h.target_spot "
                "FROM holdings h WHERE NOT EXISTS ("
                "  SELECT 1 FROM events e WHERE e.action = 'opened' "
                "  AND e.symbol = h.occ_symbol AND e.at >= h.opened_at)").fetchall()
        return [Holding(**dict(r)) for r in rows if r["occ_symbol"] in held]

    def order_story(self, holding: Holding) -> tuple[str, str]:
        """What was said about this position when it was ordered.

        The detail line from the `ordered` event and the model's reasoning from
        the decision that approved it -- both written at submission, and both
        needed again when the fill is finally announced, possibly a pass later.
        Empty strings where nothing was recorded, which is the case for anything
        ordered before fills were tracked separately.
        """
        with self._connect() as conn:
            ordered = conn.execute(
                "SELECT detail FROM events WHERE action = 'ordered' AND symbol = ? "
                "AND at >= ? ORDER BY id DESC LIMIT 1",
                (holding.occ_symbol, holding.opened_at)).fetchone()
            decision = conn.execute(
                "SELECT thinking FROM decisions WHERE underlying = ? AND approved = 1 "
                "AND at <= ? ORDER BY id DESC LIMIT 1",
                (holding.underlying, holding.opened_at)).fetchone()
        return ((ordered["detail"] if ordered else ""),
                (decision["thinking"] or "" if decision else ""))

    def record_external_close(self, holding: Holding, *, note: str) -> None:
        """A position left the book without this agent closing it.

        Someone closed it by hand, or the broker did -- assignment, expiry, a
        liquidation. Either way the agent's picture of the book was wrong until
        now, and two things have to happen: the stale `holdings` row goes, and
        the underlying starts a cooldown.

        **The cooldown is the deliberate part.** The agent did not make this
        exit and cannot see why it happened, so it has no basis for concluding
        the idea is still good -- and re-buying a name on the next pass after
        someone stepped in to close it is exactly the behaviour cooldown exists
        to prevent. Cooling off is the conservative reading, and the cost of
        being wrong is a trade skipped rather than a trade repeated.
        """
        self.record("closed", holding.underlying, holding.occ_symbol,
                    f"{_EXTERNAL_PREFIX} -- {note}")
        self.close_holding(holding.occ_symbol)

    # -- reading -----------------------------------------------------------

    def cooling_off(self, *, within_days: int, as_of: date | None = None) -> dict[str, int]:
        """Underlyings still inside their post-stop-loss cooldown.

        Returns sessions remaining per underlying, which is what the gate wants
        -- it can then say "2 day(s) left" rather than making the caller work it
        out.

        **Measured in TRADING SESSIONS, not calendar days, and that distinction
        is the whole point.** Counting calendar days meant a Friday stop-out was
        cold by Monday, so the rule did nothing on exactly the days it mattered
        most. Observed live: F stopped out on Friday 4 Sep 2026 and was re-bought
        on Tuesday the 8th -- four calendar days, which cleared a two-day
        cooldown, but the next trading session, because the 7th was Labor Day.

        A session is a day the agent recorded decisions on, which is every day
        the market was open and the agent was running. That needs no holiday
        calendar to maintain and cannot drift out of date.

        Only today's *completed* predecessors count. Today itself is excluded so
        that the answer does not change between the first pass of a day and the
        last -- decision rows accumulate as the day goes on, and a cooldown that
        quietly expired at lunchtime would be worse than no cooldown at all.

        **Any losing close counts, whatever reason it carries.** A take-profit
        does not: the reasoning worked, and re-entering after a win is not the
        behaviour this rule exists to prevent.

        That used to be narrower. Only a stop loss started a cooldown, on the
        argument that a premium backstop means the option collapsed while the
        thesis survived -- nothing disproved, nothing to cool off from. On
        9 Oct 2026 both accounts closed a GLD put at about -26% under that label
        and bought a GLD CALL on the same name within forty minutes.

        The argument had assumed the only re-entry worth preventing was the same
        trade again. Taking the opposite side of a name that just cost a full
        position is worse, and it refutes its own premise: if the bearish view
        was still alive, the bullish one cannot be. The money is gone either
        way, which is the fact the rule should key on.

        The label still means what it meant -- a failed thesis reads differently
        from a broken quote in the journal, and that is worth keeping. It just
        no longer decides this.
        """
        today = as_of or datetime.now(UTC).date()
        # Bounded so this does not scan the whole history. Generous, because the
        # bound is now in sessions and a long agent outage stretches how many
        # calendar days a handful of sessions can span.
        floor = (today - timedelta(days=max(30, within_days * 10))).isoformat()

        with self._connect() as conn:
            # A loss is a loss: `pnl < 0` covers stops, premium backstops and
            # anything else that closed down. Positions closed outside the agent
            # carry no P&L to judge, so those are matched by name.
            rows = conn.execute(
                "SELECT underlying, MAX(day) AS day FROM events "
                "WHERE action = 'closed' AND day >= ? "
                "AND (pnl < 0 OR detail LIKE ?) "
                "GROUP BY underlying",
                (floor, f"{_EXTERNAL_PREFIX}%"),
            ).fetchall()

            remaining: dict[str, int] = {}
            for row in rows:
                sessions = conn.execute(
                    "SELECT COUNT(DISTINCT day) FROM decisions "
                    "WHERE day > ? AND day < ?",
                    (row["day"], today.isoformat()),
                ).fetchone()[0]
                left = within_days - sessions
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

    def directional_views(self, *, since: str | None = None) -> list[dict]:
        """Every directional opinion the model expressed, one per symbol per day.

        Most of these never became trades -- the book was full, three calls were
        already open, the spread was too wide -- and that is exactly why they are
        worth having. A refusal for a reason that has nothing to do with the
        model is a free test of the model.

        One row per symbol per day, taking the highest-conviction view of that
        day, because the same symbol is examined every fifteen minutes and
        twenty-six near-identical opinions would weight one day like a month.
        `traded` says whether any of that day's passes actually approved it.
        """
        sql = (
            "SELECT day, underlying, direction, MAX(confidence) AS confidence, "
            "       MAX(approved) AS traded, "
            "       MIN(reason) AS reason "
            "FROM decisions "
            "WHERE confidence IS NOT NULL AND direction IN ('up', 'down') "
        )
        args: tuple = ()
        if since:
            sql += "AND day >= ? "
            args = (since,)
        sql += "GROUP BY day, underlying, direction ORDER BY day, underlying"

        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql, args)]

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


# `cooling_off` no longer matches on the words in a close at all -- it keys on
# money lost, so every losing close counts and no wording can exempt one.
# `loop.py` still writes "stop loss" or "premium backstop", because a failed
# thesis and a broken quote read differently in the journal and that is worth
# keeping. It just no longer decides anything.

# A position that left the book without the agent closing it. Kept separate from
# the stop-loss prefix rather than borrowed, because the journal should not
# report a hand-placed exit as a stop the agent decided on. Both start a
# cooldown; only one of them is a claim about why the trade ended.
_EXTERNAL_PREFIX = "closed outside the agent"
