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
    # The mirror of that on the winning side: the option has gained enough to be
    # worth banking even though the underlying has not reached the level the
    # thesis was aiming at.
    PREMIUM_TARGET = "premium target"


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

    Two stops run together, and the position closes on whichever comes first.

    **The premium stop is an absolute floor.** It answers "has this trade lost
    what we budgeted for it to lose?", and the sizing arithmetic depends on the
    answer being no more than `stop_loss_pct`. It is checked first and applies to
    every position without exception.

    **The underlying stop asks a different question:** has the stock moved
    through the level that says the thesis was wrong? That one is keyed to the
    underlying because a premium-keyed rule fires on noise -- a 25% fall in a
    0.65-delta option is under a 2% move in the stock.

    Both are needed, and running only the second one was a real defect. Measured
    on the live book on 4 Sep 2026: the underlying stop sat at 2x ATR, which
    translated to an average premium loss of **52%** by the time it triggered --
    against sizing that assumed 25%. Ford sat at -34% with its underlying stop
    still 2.2% away and no rule able to close it. A stop the position can travel
    past is not a stop.

    Falls back to premium-only when there is no recorded entry level, which
    happens for positions this agent did not open.
    """
    days = position.days_to_expiry(today)
    if days is not None and days <= settings.close_before_expiry:
        return ExitDecision(
            position, ExitReason.EXPIRY,
            f"{days} day(s) to expiry, inside the "
            f"{settings.close_before_expiry}-day close-out window")

    ret = position.return_pct

    # --- the floor, checked before anything else --------------------------
    #
    # Whatever the underlying is doing, the budget for this trade is spent. The
    # only judgement left is what to CALL it, because a stop loss starts a
    # cooldown and a collapsed premium does not.
    if ret <= -settings.stop_loss_pct:
        if holding is None or spot is None or spot <= 0:
            return ExitDecision(
                position, ExitReason.STOP_LOSS,
                f"down {ret:.1%}, past the {settings.stop_loss_pct:.0%} stop "
                f"(no entry level recorded, so keyed to premium)")

        move = holding.move_pct(spot)
        against = -move
        to_stop = (abs(holding.stop_spot - holding.entry_spot) / holding.entry_spot
                   if holding.entry_spot > 0 else 0.0)

        # The stock has barely moved and the option lost a quarter of its value
        # anyway -- a volatility crush, not a disproved thesis. Half the distance
        # to the stop is the dividing line: past that the underlying is genuinely
        # going the wrong way and this is an ordinary stop loss.
        if to_stop > 0 and against < to_stop / 2:
            return ExitDecision(
                position, ExitReason.PREMIUM_BACKSTOP,
                f"premium down {ret:.1%} past the {settings.stop_loss_pct:.0%} "
                f"stop while {position.underlying} is only {move:+.1%} from "
                f"entry -- the option collapsed, not the thesis")

        return ExitDecision(
            position, ExitReason.STOP_LOSS,
            f"premium down {ret:.1%} past the {settings.stop_loss_pct:.0%} stop, "
            f"with {position.underlying} {move:+.1%} from entry "
            f"(underlying stop {holding.stop_spot:,.2f})")

    # No entry level recorded: this is not ours, or predates the holdings table.
    if holding is None or spot is None or spot <= 0:
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

    # The thesis has not finished playing out, but the option has already gained
    # enough to be worth taking.
    #
    # Keying exits to the underlying is right for the STOP -- a premium stop
    # fires on noise. The target side is not symmetric: a large premium gain is
    # not noise, it is money, and holding it while waiting for the underlying to
    # travel further risks giving it back. Observed directly on 2 Sep 2026, when
    # a PLTR put stood at +50.7% with its underlying target still 5% away.
    #
    # So the underlying target still governs the thesis, and this catches the
    # case where the position has already paid regardless.
    if ret >= settings.premium_target_backstop_pct:
        return ExitDecision(
            position, ExitReason.PREMIUM_TARGET,
            f"premium up {ret:.1%}, past the "
            f"{settings.premium_target_backstop_pct:.0%} take-profit backstop, "
            f"with {position.underlying} at {spot:,.2f} still short of the "
            f"{holding.target_spot:,.2f} target")

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
