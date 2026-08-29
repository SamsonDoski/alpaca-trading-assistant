"""The vocabulary of the system: what a contract, an account, a position and a
proposal *are*, independent of where they came from.

Nothing in this module opens a socket, reads a file, or knows that Alpaca exists.
That is the point. Every other module depends on this one and this one depends on
nothing, so the direction of coupling always points inward at a stable centre.
Replace the MCP reader with a REST reader tomorrow and none of these types change.

The second principle at work here is that **the object holding the data answers
the questions about it**. A contract knows its own spread; an account knows what
it can afford; a position knows how far it has moved against us. Callers ask
rather than compute, so a rule like "what counts as too wide a spread" is written
once, in the place that owns the numbers, instead of being re-derived at every
call site that happens to need it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

# An option contract controls 100 shares, so a quote of $3.00 costs $300 in cash.
# Naming the number keeps a bare 100 from appearing in the middle of a sizing
# calculation, where it reads as a mystery constant.
SHARES_PER_CONTRACT = 100


class Direction(str, Enum):
    """Which way the agent believes the underlying is going.

    Subclassing `str` means a Direction serialises straight into JSON for the
    model's structured output and into SQLite for the journal, with no conversion
    step in between to forget.
    """

    UP = "up"
    DOWN = "down"

    @property
    def option_right(self) -> str:
        """The kind of contract that expresses this view."""
        return "call" if self is Direction.UP else "put"


@dataclass(frozen=True, slots=True)
class OptionContract:
    """One listed option, priced, with its Greeks attached.

    Frozen because a contract is a fact about the market at a moment, not a
    workspace. When the price moves you fetch a new one; you never edit this.
    """

    occ_symbol: str          # e.g. AAPL260918C00230000
    underlying: str
    right: str               # "call" or "put"
    strike: float
    expiry: date
    bid: float
    ask: float
    delta: float | None      # signed: positive for calls, negative for puts
    implied_volatility: float | None
    open_interest: int | None

    @property
    def mid(self) -> float:
        """The midpoint of the quote, and the price every rule here is written
        against.

        The ask overstates what a patient limit order actually pays and the bid
        understates it. Sizing and the spread rule both need one number that is
        not systematically wrong in either direction, and this is it.
        """
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        """How wide the quote is, as a fraction of the midpoint.

        This is the most useful liquidity measure available without an order
        book. A 5% spread means a position is down 5% the moment it opens, purely
        from the cost of crossing, which is larger than the edge on an average
        trade. Returns 1.0 when there is no usable quote, so a market that is not
        really there can never look cheap to the gate reading this.
        """
        if self.mid <= 0:
            return 1.0
        return (self.ask - self.bid) / self.mid

    @property
    def cost_per_contract(self) -> float:
        """Cash out the door for one contract, at the midpoint."""
        return self.mid * SHARES_PER_CONTRACT

    @property
    def abs_delta(self) -> float:
        """Delta without its sign, so calls and puts compare on one scale.

        A call at +0.65 and a put at -0.65 behave identically in the ways that
        matter to sizing and time decay. The sign records only direction, which
        the agent has already decided by the time anything reads this.
        """
        return abs(self.delta) if self.delta is not None else 0.0

    def days_to_expiry(self, on: date) -> int:
        return (self.expiry - on).days


@dataclass(frozen=True, slots=True)
class AccountState:
    """What the brokerage says the account is worth right now."""

    equity: float
    options_buying_power: float
    cash: float

    @property
    def available(self) -> float:
        """Cash the agent may commit to a new position.

        The smaller of buying power and equity, floored at zero. The two are
        normally close; taking the minimum means a stale or surprising value in
        either field can only ever make the agent more cautious, never less.
        """
        return max(0.0, min(self.options_buying_power, self.equity))


@dataclass(frozen=True, slots=True)
class OpenPosition:
    """A position the brokerage is currently holding for us.

    Long options only. There is no margin call on a long option -- the premium
    paid is the whole of the possible loss -- which is why the risk rules in this
    system are about position size and exits rather than maintenance margin.
    """

    occ_symbol: str
    underlying: str
    quantity: int
    entry_price: float       # per share, so 3.00 means $300 a contract
    current_price: float
    expiry: date | None

    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.quantity * SHARES_PER_CONTRACT

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity * SHARES_PER_CONTRACT

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def return_pct(self) -> float:
        """Gain or loss as a fraction of what was paid.

        Measured against cost rather than account equity because every exit rule
        in this system is stated that way: a 25% stop means a quarter of the
        premium, not a quarter of the account.
        """
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def days_to_expiry(self, on: date) -> int | None:
        return (self.expiry - on).days if self.expiry else None


@dataclass(frozen=True, slots=True)
class Proposal:
    """What the model produced: a view on one underlying, and its reasoning.

    Deliberately holds no contract, no quantity and no price. The model proposes
    a *direction on a symbol* and nothing else. Choosing the contract is
    mechanical, sizing it is arithmetic, and permitting it belongs to the gates,
    so none of those are decisions the model is allowed to make. Leaving them out
    of this type is what makes that boundary real instead of a promise in a
    comment somewhere.
    """

    underlying: str
    direction: Direction
    confidence: float            # 0.0 to 1.0, the model's own stated conviction
    rationale: str               # a sentence or two, written for a human reader
    thinking_summary: str = ""   # the model's summarised reasoning, for the journal

    @property
    def is_actionable(self) -> bool:
        """Whether this proposal asks for a trade at all.

        A model that declines to trade is working correctly, not failing, so "no
        view" is carried as a real proposal with zero confidence rather than as
        an absent one. The journal then records the pass and the reasoning behind
        it instead of showing a silent gap.
        """
        return self.confidence > 0.0


@dataclass(frozen=True, slots=True)
class OrderDraft:
    """A fully specified trade, before anything has approved it.

    This is what the order gates inspect. It exists so that "what we intend to
    do" is one value that can be logged, tested and argued about with no broker
    connection anywhere in sight.
    """

    proposal: Proposal
    contract: OptionContract
    quantity: int
    limit_price: float       # per share

    @property
    def total_cost(self) -> float:
        return self.limit_price * self.quantity * SHARES_PER_CONTRACT

    def with_quantity(self, quantity: int) -> OrderDraft:
        """A copy of this draft, resized.

        Returning a new object instead of mutating keeps every draft immutable,
        which is what lets the gate runner hold on to the original and report
        both the requested size and the final one in the audit trail.
        """
        return OrderDraft(self.proposal, self.contract, quantity, self.limit_price)

    def __str__(self) -> str:
        c = self.contract
        return (f"{c.underlying} {c.right} {c.strike:g} exp {c.expiry} "
                f"x{self.quantity} at {self.limit_price:.2f} "
                f"(${self.total_cost:,.0f})")
