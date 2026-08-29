"""Deciding when to close a position.

The mirror image of `entry.py`, and deliberately much simpler. Entries are
optional -- there is always the choice of doing nothing, and the whole apparatus
of screening, reasoning and gating exists to make "nothing" the easy answer.
Exits are not optional. A position that has hit its stop must close, and no rule
in this system is allowed to prevent that.

So there is no model call here, no gate chain, and no discretion. Three plain
comparisons against numbers fixed when the position was opened, evaluated fresh
on every pass.

**Why this runs every fifteen minutes rather than twice a day.** Alpaca supports
trailing stops for stocks, not for options. There is no protective order resting
at the broker, so a stop only exists while this code is running. The gap between
passes is the distance a position can travel unwatched, which makes the schedule
interval a risk parameter rather than an operational convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from agent.domain import OpenPosition
from agent.settings import Settings


class ExitReason(str, Enum):
    STOP_LOSS = "stop loss"
    TAKE_PROFIT = "take profit"
    EXPIRY = "expiry approaching"


@dataclass(frozen=True, slots=True)
class ExitDecision:
    """A position that should be closed, and why."""

    position: OpenPosition
    reason: ExitReason
    detail: str

    @property
    def urgent(self) -> bool:
        """Whether to skip the patient limit and close at market immediately.

        A stop loss is urgent and a take profit is not, which sounds arbitrary
        until you look at what each one is racing. A stop fires because the
        position is moving against us, so every minute the order sits unfilled
        costs more -- and measurement on the earlier system showed exactly that:
        patient stop orders timing out repeatedly while the price fell, filling
        between -27% and -33% on a rule meant to cut at -25%.

        A take profit is the opposite situation. The position is moving in our
        favour, so waiting for a better fill costs nothing worse than a smaller
        gain, and there is no runaway to outrun.
        """
        return self.reason is ExitReason.STOP_LOSS


def check_exit(position: OpenPosition, settings: Settings,
               today: date) -> ExitDecision | None:
    """Whether this position should be closed now.

    Checked in order of consequence. Expiry comes first because it is the only
    condition that is certain -- a contract in its final week stops behaving
    like the directional bet it was opened as regardless of where the price is,
    and time decay accelerates whether the trade is winning or losing.
    """
    days = position.days_to_expiry(today)
    if days is not None and days <= settings.close_before_expiry:
        return ExitDecision(
            position, ExitReason.EXPIRY,
            f"{days} day(s) to expiry, inside the "
            f"{settings.close_before_expiry}-day close-out window")

    ret = position.return_pct

    if ret <= -settings.stop_loss_pct:
        return ExitDecision(
            position, ExitReason.STOP_LOSS,
            f"down {ret:.1%}, past the {settings.stop_loss_pct:.0%} stop")

    if ret >= settings.take_profit_pct:
        return ExitDecision(
            position, ExitReason.TAKE_PROFIT,
            f"up {ret:.1%}, at the {settings.take_profit_pct:.0%} target")

    return None


def exit_limit_price(bid: float, ask: float, aggression: float) -> float:
    """Where to place a closing limit inside the spread.

    Aggression runs the other way from an entry: 1.0 means hitting the bid --
    accepting what someone is actually offering right now -- rather than asking
    for the midpoint and hoping. The default is 1.0 for the reason in
    `ExitDecision.urgent`: a missed exit is not a harmless outcome.
    """
    span = ask - bid
    return round(ask - span * aggression, 2)
