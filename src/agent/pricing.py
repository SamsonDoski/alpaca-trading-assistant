"""What an option actually costs to hold, and whether it is expensive.

Everything here is arithmetic over numbers the agent already fetches. No network,
no state, no dependencies -- which makes this the easiest module to test and the
one most worth testing, because a directional strategy that ignores these two
quantities is buying blind.

**The two questions this module answers.**

*Is the premium expensive?* Implied volatility is the price of an option, and a
number on its own -- "34% IV" -- means nothing. 34% is cheap on a stock that
routinely moves 45% annualised and dear on one that moves 20%. So the useful
measure is IV against the underlying's own REALIZED volatility. When implied
runs well above realized, you are paying for movement the stock has not actually
been delivering; the gap is the variance risk premium, and as a premium *buyer*
you are on the paying side of it.

That is the honest form of an "IV rank" gate here. A true IV rank compares
today's implied volatility against its own history, which needs an IV time series
per contract that this system does not keep and cannot cheaply reconstruct --
contracts expire and roll, so the series is not continuous. IV against realized
answers the same question, uses only data already in hand, and has a clearer
economic meaning.

*What does holding it cost per day?* Theta. A 0.65-delta option 35 days out
bleeds real money whether or not the underlying moves, and until now nothing in
this system could see that. Expressed as a fraction of the premium paid, it says
plainly how much of the position evaporates each day the thesis takes to work.

**On the model used.** Black-Scholes with the risk-free rate set to zero. That is
a simplification and worth naming: at 30-45 days the rate term contributes very
little to theta, and the alternative -- carrying a rate the agent has no reliable
source for -- would add a dependency for a rounding error. These numbers are used
to compare and to reject, never to price a trade.
"""

from __future__ import annotations

import math
from statistics import fmean, stdev

from agent.domain import OptionContract, PriceBar

# Trading days in a year. Volatility is quoted annualised by convention, and
# daily returns have to be scaled by the square root of this to get there.
TRADING_DAYS = 252

# Calendar days, not trading days, for decay. An option loses time value over a
# weekend too -- less than over two trading days, but not nothing, and the
# position is held through it either way.
CALENDAR_DAYS = 365


def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def realized_volatility(bars: tuple[PriceBar, ...] | list[PriceBar],
                        lookback: int = 20) -> float | None:
    """Annualised volatility of the underlying's own recent returns.

    Close-to-close log returns, standard deviation, scaled to a year. Returns
    None rather than a number when there is not enough history -- the same rule
    the trend calculations follow, and for the same reason: a value that means
    "we could not measure this" must not be spelled like a measurement.
    """
    if lookback < 2 or len(bars) < lookback + 1:
        return None

    window = bars[-(lookback + 1):]
    returns = []
    # window[:-1] against window[1:] pairs each bar with the one after it.
    # Zipping the whole window against its own tail is off by one, and
    # strict=True is what turned that into a loud failure instead of a
    # silently short series.
    for previous, current in zip(window[:-1], window[1:], strict=True):
        if previous.close <= 0 or current.close <= 0:
            return None
        returns.append(math.log(current.close / previous.close))

    if len(returns) < 2:
        return None

    daily = stdev(returns)
    return daily * math.sqrt(TRADING_DAYS)


def premium_richness(implied_vol: float | None,
                     realized_vol: float | None) -> float | None:
    """How much implied volatility exceeds what the stock actually does.

    1.0 means the option is priced for exactly the movement the underlying has
    been delivering. 1.5 means you are paying half again as much as recent
    behaviour justifies. Below 1.0 means implied is *under* realized, which for
    a premium buyer is the favourable side of the trade.

    This is the number the richness gate reads, and the single most important
    quantity a long-premium strategy can look at.
    """
    if not implied_vol or not realized_vol or realized_vol <= 0:
        return None
    return implied_vol / realized_vol


def black_scholes_theta(spot: float, strike: float, implied_vol: float,
                        days_to_expiry: int, right: str) -> float | None:
    """Time decay per calendar day, in dollars per share.

    Returned as a negative number, because that is what it is: a long option
    loses this much value every day the underlying stands still.

    With the risk-free rate at zero the rate term drops out and the expression is
    the same for calls and puts, which is why `right` is accepted but only
    validated. Keeping the argument means the signature does not change if a rate
    is ever introduced.
    """
    if spot <= 0 or strike <= 0 or implied_vol <= 0 or days_to_expiry <= 0:
        return None
    if right not in ("call", "put"):
        return None

    years = days_to_expiry / CALENDAR_DAYS
    denominator = implied_vol * math.sqrt(years)
    if denominator <= 0:
        return None

    d1 = (math.log(spot / strike) + 0.5 * implied_vol * implied_vol * years) / denominator
    annual_theta = -(spot * _normal_pdf(d1) * implied_vol) / (2 * math.sqrt(years))
    return annual_theta / CALENDAR_DAYS


def daily_decay_pct(contract: OptionContract, spot: float,
                    today) -> float | None:
    """What fraction of the premium decays each day, as a positive number.

    0.02 means the position loses two percent of what you paid for it every day
    the underlying does nothing. Over a two-week thesis that is roughly a fifth
    of the position gone before direction has had a chance to matter.

    Expressed against the premium rather than in dollars because that is the
    scale every other rule in this system uses -- the stop, the target and the
    risk budget are all fractions of premium, and a number in dollars would not
    compare to any of them.
    """
    if contract.implied_volatility is None or contract.mid <= 0:
        return None

    theta = black_scholes_theta(
        spot=spot,
        strike=contract.strike,
        implied_vol=float(contract.implied_volatility),
        days_to_expiry=contract.days_to_expiry(today),
        right=contract.right,
    )
    if theta is None:
        return None

    return abs(theta) / contract.mid


def decay_to_target_pct(contract: OptionContract, spot: float, today,
                        holding_days: int) -> float | None:
    """Roughly how much decay a thesis has to overcome before it pays.

    A linear projection of the daily rate, which understates the real cost
    because decay accelerates as expiry approaches. Deliberately kept simple: a
    number used to compare contracts and reject the worst ones does not need to
    be exact, and a more elaborate model would imply a precision this input
    does not have.
    """
    daily = daily_decay_pct(contract, spot, today)
    if daily is None:
        return None
    return daily * max(0, holding_days)


def underlying_move_for(contract: OptionContract, spot: float,
                        premium_change_pct: float) -> float | None:
    """How far the underlying must move to change the premium by a given amount.

    The translation that makes an underlying-keyed stop possible. A 25% fall in
    premium on a 0.65-delta option is a much smaller move in the stock than the
    number suggests -- and knowing which one is being measured is the difference
    between a stop that protects a thesis and a stop that fires on noise.

    Returned as a signed fraction of spot: -0.03 means the stock falling three
    percent produces the given premium change.
    """
    if spot <= 0 or contract.abs_delta <= 0 or contract.mid <= 0:
        return None

    # Premium change in dollars per share, then divided by delta to get the
    # underlying move that produces it. Delta is dPremium/dSpot, so this is
    # simply that relationship rearranged.
    premium_change = contract.mid * premium_change_pct
    move = premium_change / contract.abs_delta

    # A put gains when the underlying falls, so the sign of the move that
    # produces a given premium change is inverted relative to a call.
    if contract.right == "put":
        move = -move
    return move / spot


def summarise(contract: OptionContract, spot: float, today,
              realized_vol: float | None) -> dict:
    """Every derived number for one contract, for the brief and the journal.

    Gathered in one place so the prompt, the gates and the audit trail all read
    the same figures rather than each computing their own.
    """
    return {
        "richness": premium_richness(contract.implied_volatility, realized_vol),
        "daily_decay": daily_decay_pct(contract, spot, today),
        "realized_vol": realized_vol,
    }


def average_true_range_pct(bars: tuple[PriceBar, ...] | list[PriceBar],
                           lookback: int = 14) -> float | None:
    """Typical daily range as a fraction of price.

    Used to size an underlying-keyed stop in units the stock itself defines. A
    stop placed three percent away means something very different on a name that
    moves one percent a day and one that moves four -- on the second it is inside
    the noise, and will fire on a day where nothing happened.
    """
    if len(bars) < lookback + 1:
        return None

    ranges = []
    for previous, current in zip(bars[-(lookback + 1):-1], bars[-lookback:], strict=True):
        true_range = max(
            current.high - current.low,
            abs(current.high - previous.close),
            abs(current.low - previous.close),
        )
        ranges.append(true_range)

    if not ranges:
        return None
    last_close = bars[-1].close
    if last_close <= 0:
        return None
    return fmean(ranges) / last_close
