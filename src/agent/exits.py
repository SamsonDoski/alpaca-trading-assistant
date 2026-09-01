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
from agent.journal import Holding
from agent.settings import Settings


class ExitReason(str, Enum):
    STOP_LOSS = "stop loss"
    TAKE_PROFIT = "take profit"
    EXPIRY = "expiry approaching"
    # A premium collapse with the underlying thesis still intact -- usually a
    # volatility crush. Kept separate from STOP_LOSS because it means something
    # different and, unlike a real stop, does not start a cooldown: the thesis
    # was never disproved, so there is nothing to cool off from.
    PREMIUM_BACKSTOP = "premium backstop"


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
        return self.reason in (ExitReason.STOP_LOSS, ExitReason.PREMIUM_BACKSTOP)


def check_exit(position: OpenPosition, settings: Settings, today: date,
               *, holding: Holding | None = None,
               spot: float | None = None) -> ExitDecision | None:
    """Whether this position should be closed now.

    **The stop is keyed to the underlying, not the premium, and that is the
    substantive change.** A 25% fall in a 0.65-delta option is under a 2% move
    in the stock -- comfortably inside a normal day -- so a premium-keyed stop
    fires on noise and on volatility crushes while the directional thesis is
    completely intact. Worse, it fires *most often* on exactly the high-volatility
    names where the premium swings hardest, which is the opposite of a risk rule.

    So the primary rule asks the only question that matters: has the underlying
    moved through the level that says we were wrong?

    The premium rule survives as a BACKSTOP at a much wider level. It catches the
    case the underlying cannot see -- implied volatility collapsing so far that
    the option is worthless even though the stock did nothing.

    Falls back to the old premium-only behaviour when there is no recorded entry
    level. That happens for positions this agent did not open, and pretending
    otherwise would silently leave them unmanaged.
    """
    days = position.days_to_expiry(today)
    if days is not None and days <= settings.close_before_expiry:
        return ExitDecision(
            position, ExitReason.EXPIRY,
            f"{days} day(s) to expiry, inside the "
            f"{settings.close_before_expiry}-day close-out window")

    ret = position.return_pct

    # No entry level recorded: this is not ours, or predates the holdings table.
    # The premium stop is the only rule available, so it stays primary here.
    if holding is None or spot is None or spot <= 0:
        if ret <= -settings.stop_loss_pct:
            return ExitDecision(
                position, ExitReason.STOP_LOSS,
                f"down {ret:.1%}, past the {settings.stop_loss_pct:.0%} stop "
                f"(no entry level recorded, so keyed to premium)")
        if ret >= settings.take_profit_pct:
            return ExitDecision(
                position, ExitReason.TAKE_PROFIT,
                f"up {ret:.1%}, at the {settings.take_profit_pct:.0%} target")
        return None

    move = holding.move_pct(spot)

    if holding.breached_stop(spot):
        return ExitDecision(
            position, ExitReason.STOP_LOSS,
            f"{position.underlying} moved {move:+.1%} against the thesis "
            f"(through {holding.stop_spot:,.2f} from {holding.entry_spot:,.2f}); "
            f"premium {ret:+.1%}")

    if holding.reached_target(spot):
        return ExitDecision(
            position, ExitReason.TAKE_PROFIT,
            f"{position.underlying} moved {move:+.1%} in favour "
            f"(through {holding.target_spot:,.2f}); premium {ret:+.1%}")

    # The thesis is alive but the option has been gutted anyway -- almost always
    # implied volatility collapsing. Set far wider than the real stop, because
    # its job is to catch a broken position rather than to manage a live one.
    if ret <= -settings.premium_backstop_pct:
        return ExitDecision(
            position, ExitReason.PREMIUM_BACKSTOP,
            f"premium down {ret:.1%} past the {settings.premium_backstop_pct:.0%} "
            f"backstop while {position.underlying} is only {move:+.1%} from entry "
            f"-- the option collapsed, not the thesis")

    return None


def stop_and_target(spot: float, direction: str, *, atr_pct: float | None,
                    settings: Settings) -> tuple[float, float]:
    """The underlying levels that will decide a position's fate.

    Distance is measured in the stock's own units. A stop three percent away
    means something entirely different on a name that moves one percent a day
    and one that moves four -- on the second it sits inside the noise and will
    fire on a day where nothing happened. So the distance is a multiple of
    average true range, and only falls back to a flat percentage when there is
    not enough history to measure one.

    The target is the stop distance scaled by the reward-to-risk ratio, so the
    two cannot drift apart and quietly move the break-even win rate.
    """
    band = atr_pct if atr_pct and atr_pct > 0 else settings.fallback_stop_pct
    stop_distance = band * settings.stop_atr_multiple
    target_distance = stop_distance * settings.reward_to_risk

    if direction == "up":
        return spot * (1 - stop_distance), spot * (1 + target_distance)
    return spot * (1 + stop_distance), spot * (1 - target_distance)


def exit_limit_price(bid: float, ask: float, aggression: float) -> float:
    """Where to place a closing limit inside the spread.

    Aggression runs the other way from an entry: 1.0 means hitting the bid --
    accepting what someone is actually offering right now -- rather than asking
    for the midpoint and hoping. The default is 1.0 for the reason in
    `ExitDecision.urgent`: a missed exit is not a harmless outcome.
    """
    span = ask - bid
    return round(ask - span * aggression, 2)
