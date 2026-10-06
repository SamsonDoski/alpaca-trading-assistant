"""Marking the model's homework, including the answers it was not allowed to give.

Every pass asks the model about every symbol, and almost all of those opinions
die unrecorded in consequence: the book was full, the spread was too wide, three
other calls were already open. Only about one view in five ever became a trade,
which means four in five carried information nobody ever checked.

This module checks them. For every directional view the model expressed, it
reconstructs the trade that view implied -- the same entry price, the same
ATR-derived stop and target that `exits.py` would have used -- and walks the
underlying's daily bars forward to see which level came first.

**It measures direction, not profit, and the difference matters.** A shadow
trade pays no spread, loses nothing to theta and is never hurt by an implied
volatility crush. Its win rate will therefore be flattering compared to the
account. That is the point: the model's only job is direction, and judging it on
option returns means judging it on four other things at once. Dollars are
measured where dollars happen, against the broker's fills.

The second thing it buys is sample size. 81 closed trades took five weeks and
left every question underpowered. The same five weeks produced 559 directional
views, and they can all be marked today.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from agent.domain import PriceBar
from agent.exits import stop_and_target
from agent.pricing import average_true_range_pct
from agent.settings import Settings


@dataclass(frozen=True, slots=True)
class ShadowTrade:
    """The trade a directional view implied, whether or not it was taken."""

    day: str
    underlying: str
    direction: str
    confidence: float
    traded: bool             # did this view actually reach the broker?
    reason: str              # why it was approved, or why it was refused
    entry_spot: float
    stop_spot: float
    target_spot: float


@dataclass(frozen=True, slots=True)
class ShadowResult:
    """How the underlying resolved it."""

    trade: ShadowTrade
    outcome: str             # "target" | "stop" | "expired" | "open"
    days_held: int | None
    move_pct: float | None   # the underlying's move, signed so + favours the view

    @property
    def won(self) -> bool:
        return self.outcome == "target"

    @property
    def decided(self) -> bool:
        """Whether this has run far enough to count either way."""
        return self.outcome in ("target", "stop", "expired")


def build(*, day: str, underlying: str, direction: str, confidence: float,
          traded: bool, reason: str, history: list[PriceBar],
          settings: Settings) -> ShadowTrade | None:
    """Reconstruct the trade a view implied, from the bars available that day.

    `history` must end on the decision's own day: the levels have to be derived
    from what was knowable then, or the whole exercise measures hindsight.
    Returns None when there is not enough history to place a stop.
    """
    if not history:
        return None

    entry_spot = history[-1].close
    if entry_spot <= 0:
        return None

    atr_pct = average_true_range_pct(history)
    stop_spot, target_spot = stop_and_target(
        entry_spot, direction, atr_pct=atr_pct, settings=settings)

    return ShadowTrade(day=day, underlying=underlying, direction=direction,
                       confidence=confidence, traded=traded, reason=reason,
                       entry_spot=entry_spot, stop_spot=stop_spot,
                       target_spot=target_spot)


def resolve(trade: ShadowTrade, forward: list[PriceBar],
            *, max_days: int) -> ShadowResult:
    """Walk the bars after the decision and report which level came first.

    Uses each day's high and low rather than its close, because a stop is a
    level the price touches, not a level it finishes at -- judging on closes
    would quietly turn every intraday stop-out into a position still running.

    **When both levels fall inside one day, the stop wins.** Daily bars cannot
    say which came first, and assuming the good outcome would flatter every
    volatile name in the sample. This is the conservative reading and it is
    applied without exception.
    """
    up = trade.direction == "up"

    for index, bar in enumerate(forward[:max_days], start=1):
        hit_stop = bar.low <= trade.stop_spot if up else bar.high >= trade.stop_spot
        hit_target = bar.high >= trade.target_spot if up else bar.low <= trade.target_spot

        if hit_stop:
            move = (trade.stop_spot - trade.entry_spot) / trade.entry_spot
            return ShadowResult(trade, "stop", index, move if up else -move)
        if hit_target:
            move = (trade.target_spot - trade.entry_spot) / trade.entry_spot
            return ShadowResult(trade, "target", index, move if up else -move)

    if len(forward) >= max_days:
        last = forward[max_days - 1].close
        move = (last - trade.entry_spot) / trade.entry_spot
        return ShadowResult(trade, "expired", max_days, move if up else -move)

    # Not enough days have passed yet for this one to have finished.
    if forward:
        last = forward[-1].close
        move = (last - trade.entry_spot) / trade.entry_spot
        return ShadowResult(trade, "open", len(forward), move if up else -move)
    return ShadowResult(trade, "open", 0, None)


def summarise(results: list[ShadowResult]) -> dict:
    """Win rate and counts over a set of marked views."""
    decided = [r for r in results if r.decided]
    wins = [r for r in decided if r.won]
    return {
        "views": len(results),
        "decided": len(decided),
        "wins": len(wins),
        "win_rate": (len(wins) / len(decided)) if decided else 0.0,
        "targets": sum(1 for r in decided if r.outcome == "target"),
        "stops": sum(1 for r in decided if r.outcome == "stop"),
        "expired": sum(1 for r in decided if r.outcome == "expired"),
        "still_open": sum(1 for r in results if r.outcome == "open"),
        "avg_move": (sum(r.move_pct or 0.0 for r in decided) / len(decided))
                    if decided else 0.0,
    }
