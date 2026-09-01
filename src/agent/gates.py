"""The risk gates: the only thing standing between a proposal and a real order.

The whole design of this agent rests on one sentence. **The model proposes; the
gates dispose.** A language model reading option chains and news is good at
noticing things and bad at being consistently disciplined, so it is given the
first job and denied the second. Every trade it suggests must survive a chain of
plain, deterministic Python rules that have no opinions and cannot be talked out
of anything.

Two invariants make that claim real rather than decorative, and both are asserted
in the runners below rather than merely described here:

1. **A gate may reject or shrink. It may never enlarge, and it may never invent
   a trade.** The worst thing the gate chain can do to the account is nothing.
2. **Exits are not gated.** The chains here screen *entries*. Nothing in this
   module may ever prevent a position from being closed, because a rule that can
   block an exit is a rule that can trap a loss.

The chain is split in two, and the split is not cosmetic:

    screen()     runs on an underlying, BEFORE the model is called.
                 Free checks only -- no network, no contract data, no tokens.

    authorise()  runs on a fully sized OrderDraft, AFTER the model has proposed
                 and a contract has been chosen and priced.

Ordering cheap-and-absolute first means the agent never spends a model call
deciding about a symbol that a free check was always going to refuse. That is
worth real money over a week and, more importantly, it makes the expensive part
of the system the part that runs least often.

Each gate is a small object with one rule and no dependencies beyond the context
it is handed. It cannot reach the broker, cannot read a file and cannot see the
model. That is what allows the entire risk system to be tested at full speed with
no network and no account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Protocol

from agent.domain import AccountState, OpenPosition, OrderDraft
from agent.pricing import daily_decay_pct, premium_richness
from agent.settings import Settings


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    SHRINK = "shrink"


@dataclass(frozen=True, slots=True)
class Verdict:
    """One gate's answer, always with a reason attached.

    The reason is not optional and is written for a human. Every refusal ends up
    in the journal and in the end-of-day summary, and "denied by gate 4" helps
    nobody at 3pm on a Thursday.
    """

    decision: Decision
    reason: str
    quantity: int | None = None    # set only on SHRINK

    @classmethod
    def allow(cls, reason: str = "") -> Verdict:
        return cls(Decision.ALLOW, reason)

    @classmethod
    def deny(cls, reason: str) -> Verdict:
        return cls(Decision.DENY, reason)

    @classmethod
    def shrink(cls, quantity: int, reason: str) -> Verdict:
        return cls(Decision.SHRINK, reason, quantity)


@dataclass(frozen=True, slots=True)
class GateContext:
    """Everything the gates are allowed to know.

    This type is the reason the risk rules are testable. It is a plain snapshot
    of the world -- the account, the open book, the clock, the cooling-off list
    -- assembled once at the top of a pass and then handed down. No gate fetches
    anything for itself, so no gate can be slow, flaky, or surprising, and a test
    can put the system into any state it likes by building one of these.
    """

    today: date
    market_open: bool
    trading_halted: bool
    account: AccountState
    open_positions: tuple[OpenPosition, ...] = ()
    # Underlying -> days still remaining on its post-stop-loss cooldown.
    cooling_off: dict[str, int] = field(default_factory=dict)
    # Underlyings with an order resting at the broker that has not filled yet.
    pending: frozenset[str] = frozenset()
    settings: Settings = field(default_factory=Settings)

    def holds(self, underlying: str) -> bool:
        return any(p.underlying == underlying for p in self.open_positions)

    @property
    def committed_slots(self) -> int:
        """Positions plus resting orders.

        An unfilled order is a slot we have already spent, even though nothing
        is held yet. Counting only positions is how the account ends up with
        more exposure than max_positions allows -- see PositionSlots.
        """
        return len(self.open_positions) + len(self.pending)

    @property
    def committed(self) -> float:
        """Cash currently tied up in open positions, at market value."""
        return sum(p.market_value for p in self.open_positions)


# --------------------------------------------------------------------------
# Entry gates: cheap, absolute, and run before the model is ever called.
# --------------------------------------------------------------------------

class EntryGate(Protocol):
    """A rule that can refuse an underlying before any work is done on it."""

    name: str

    def check(self, underlying: str, ctx: GateContext) -> Verdict: ...


@dataclass(frozen=True, slots=True)
class KillSwitch:
    """A manual freeze on opening new positions.

    Deliberately surgical: it stops buying and nothing else. Exits, stop
    evaluation and reporting all keep running while it is on. Disabling the
    schedule instead would stop those too, which would switch off the safety
    systems in precisely the situation where they are wanted most. Halting
    entries is the correct emergency action; halting everything is not.
    """

    name: str = "kill_switch"

    def check(self, underlying: str, ctx: GateContext) -> Verdict:
        if ctx.trading_halted:
            return Verdict.deny("kill switch is on -- no new positions this run")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class MarketOpen:
    """No entries into a closed market.

    An order placed while the market is shut queues until the next open and
    fills at a price nobody has seen yet. That is a different trade from the one
    the analysis asked for, and the difference is invisible in the logs
    afterwards.
    """

    name: str = "market_open"

    def check(self, underlying: str, ctx: GateContext) -> Verdict:
        if not ctx.market_open:
            return Verdict.deny("market is closed -- quotes are stale and fills would queue")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class PositionSlots:
    """A ceiling on how many positions may be open at once.

    This reads like a risk limit and is really a quality filter. Candidates are
    ranked and the best are taken first, so raising the cap does not produce more
    trades like the ones being taken -- it reaches down into worse ones.
    """

    name: str = "position_slots"

    def check(self, underlying: str, ctx: GateContext) -> Verdict:
        # Resting orders count. A limit order that has not filled has still
        # spent a slot and reserved the cash behind it, and on a feed where
        # fills are slow that gap can stay open for hours. Counting only
        # filled positions let a live run place four orders and then remain
        # willing to place four more.
        used = ctx.committed_slots
        limit = ctx.settings.max_positions
        if used >= limit:
            held, resting = len(ctx.open_positions), len(ctx.pending)
            detail = f"{held} held" + (f" plus {resting} resting" if resting else "")
            return Verdict.deny(f"all {limit} position slots are in use ({detail})")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class SectorConcentration:
    """Caps how many positions may sit in one correlated group.

    The hole this closes: every other gate reasons about ONE trade. Position
    slots counted eight, the risk budget sized each at 4%, and nothing anywhere
    asked whether the eight were secretly the same bet. Eight long calls on
    mega-cap technology is not a diversified book -- it is one macro position
    with eight commission charges and eight chances to be wrong together.

    Grouping is declared rather than computed. A rolling correlation matrix over
    daily returns would be more precise and would also be unstable, expensive,
    and impossible to explain to anyone reading a refusal. "Three tech names is
    enough tech" is a rule a person can check.
    """

    name: str = "sector_concentration"

    def check(self, underlying: str, ctx: GateContext) -> Verdict:
        groups = ctx.settings.correlation_groups
        group = groups.get(underlying)
        if group is None:
            # Ungrouped names are treated as their own group of one, so an
            # unclassified symbol can never quietly bypass the cap.
            return Verdict.allow()

        held = sum(1 for p in ctx.open_positions if groups.get(p.underlying) == group)
        limit = ctx.settings.max_per_group
        if held >= limit:
            names = ", ".join(sorted(p.underlying for p in ctx.open_positions
                                     if groups.get(p.underlying) == group))
            return Verdict.deny(
                f"already holding {held} position(s) in '{group}' ({names}), "
                f"at the limit of {limit}")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class Cooldown:
    """Blocks re-entry into a name that recently stopped out.

    Applied after a stop loss and never after a win, because a win means the
    reasoning worked and there is nothing to cool off from. This exists because
    of an observed live failure, not a backtest: the earlier system re-bought a
    name minutes after being stopped out of it, on the same signal that had just
    failed.
    """

    name: str = "cooldown"

    def check(self, underlying: str, ctx: GateContext) -> Verdict:
        remaining = ctx.cooling_off.get(underlying, 0)
        if remaining > 0:
            return Verdict.deny(
                f"stopped out recently -- {remaining} day(s) of cooldown left")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class NotAlreadyHeld:
    """One position per underlying.

    Two positions on the same name are one position with extra steps and twice
    the correlation. Position slots would count them separately, which would
    quietly let a single idea take over the account.
    """

    name: str = "not_already_held"

    def check(self, underlying: str, ctx: GateContext) -> Verdict:
        if ctx.holds(underlying):
            return Verdict.deny("already holding a position in this underlying")
        # An unfilled order is an intention to hold. Without this, a limit that
        # sits unfilled for an hour invites a fresh order on the same name every
        # fifteen minutes -- and if the price then moves through all of them at
        # once, every one fills.
        if underlying in ctx.pending:
            return Verdict.deny("an order in this underlying is already resting unfilled")
        return Verdict.allow()


# --------------------------------------------------------------------------
# Order gates: run on a sized draft, after the model has proposed.
# --------------------------------------------------------------------------

class OrderGate(Protocol):
    """A rule that can refuse or shrink a fully specified order."""

    name: str

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict: ...


@dataclass(frozen=True, slots=True)
class MinimumConfidence:
    """Discards proposals the model itself is not convinced by.

    The model is asked to state a conviction and is taken at its word. A skipped
    entry costs nothing here -- the same symbol is re-examined on the next pass
    fifteen minutes later -- so there is no reason to act on a view its author
    described as weak.
    """

    name: str = "minimum_confidence"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        stated = draft.proposal.confidence
        floor = ctx.settings.min_confidence
        if stated < floor:
            return Verdict.deny(f"confidence {stated:.2f} is below the {floor:.2f} floor")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class DeltaBand:
    """Keeps the contract in the intended part of the chain.

    Delta is the honest way to say "slightly in the money" across underlyings at
    very different prices: 3% of a $60 stock and 3% of a $600 stock are not
    comparable instruments, but 0.65 delta and 0.65 delta are. This gate is the
    reason the strategy stays a directional bet rather than drifting into
    lottery tickets when a chain happens to be thin.
    """

    name: str = "delta_band"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        delta = draft.contract.abs_delta
        low, high = ctx.settings.delta_min, ctx.settings.delta_max
        if delta == 0.0:
            return Verdict.deny("no delta available for this contract")
        if not low <= delta <= high:
            return Verdict.deny(
                f"delta {delta:.2f} is outside the {low:.2f}-{high:.2f} band")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class SpreadWidth:
    """Refuses contracts whose quote is too wide to trade profitably.

    A position opened across a 5% spread starts 5% down, before the market has
    done anything at all. On a rule that needs roughly a third of its trades to
    win, paying that on entry is a larger drag than the edge being pursued.
    """

    name: str = "spread_width"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        spread = draft.contract.spread_pct
        limit = ctx.settings.max_spread_pct
        if spread > limit:
            return Verdict.deny(
                f"quote is {spread:.1%} wide, above the {limit:.1%} limit")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class ExpiryWindow:
    """Enforces the days-to-expiry band, including the near-expiry exclusion.

    The upper and lower bounds keep the contract in the part of the curve the
    strategy was designed around. The `close_before_expiry` check matters more:
    opening a position that the exit rules would immediately want to close is
    incoherent, and near expiry a directional bet stops behaving like one.
    """

    name: str = "expiry_window"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        days = draft.contract.days_to_expiry(ctx.today)
        s = ctx.settings

        if days <= s.close_before_expiry:
            return Verdict.deny(
                f"{days} day(s) to expiry is inside the {s.close_before_expiry}-day "
                f"close-out window")
        if not s.dte_min <= days <= s.dte_max:
            return Verdict.deny(
                f"{days} day(s) to expiry is outside the {s.dte_min}-{s.dte_max} band")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class DirectionalBalance:
    """Caps how far the whole book may lean one way.

    Sector grouping catches "all technology". This catches the subtler version:
    eight positions across eight unrelated sectors that are all long calls are
    still one bet, on the market going up. In a broad selloff they lose together
    regardless of how carefully the sectors were spread.

    Runs as an ORDER gate rather than an entry gate because direction is the
    model's answer, and the model has not answered yet when the entry screen
    runs.
    """

    name: str = "directional_balance"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        right = draft.contract.right
        same = sum(1 for p in ctx.open_positions if p.right == right)
        limit = ctx.settings.max_same_direction

        if same >= limit:
            leaning = "bullish" if right == "call" else "bearish"
            return Verdict.deny(
                f"the book already holds {same} {right} position(s) -- at the "
                f"{limit} limit on how far it may lean {leaning}")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class PremiumRichness:
    """Refuses options priced for more movement than the stock actually delivers.

    The gate this strategy was missing. Implied volatility is the price of an
    option, and a directional bet bought at rich implied volatility can be right
    about direction and still lose, because implied collapses toward realized
    once whatever was being priced in passes.

    Needs the underlying's realized volatility, which arrives on the draft's
    brief. When it cannot be computed -- too little history -- the gate refuses
    rather than waves the trade through. An unmeasurable price is not a cheap
    one, and this system's convention throughout is that missing data blocks
    rather than permits.
    """

    name: str = "premium_richness"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        richness = premium_richness(draft.contract.implied_volatility,
                                    draft.realized_vol)
        if richness is None:
            return Verdict.deny(
                "cannot price the premium: implied or realized volatility unavailable")

        limit = ctx.settings.max_iv_to_realized
        if richness > limit:
            return Verdict.deny(
                f"implied volatility is {richness:.2f}x realized, above the "
                f"{limit:.2f}x limit -- the premium is rich")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class DecayBurden:
    """Refuses options that bleed faster than a thesis can reasonably work.

    Long premium is a race between the move and the clock, and until this gate
    existed the system could not see the clock at all. A contract losing 2% of
    its value a day gives a two-week thesis a 28% hole to climb out of before
    direction has earned anything.
    """

    name: str = "decay_burden"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        decay = daily_decay_pct(draft.contract, draft.spot, ctx.today)
        if decay is None:
            return Verdict.deny("cannot compute time decay for this contract")

        limit = ctx.settings.max_daily_decay
        if decay > limit:
            # Stated as a two-week total as well, because a daily percentage is
            # hard to feel and the cumulative number is the one that matters.
            return Verdict.deny(
                f"decays {decay:.2%} per day ({decay * 14:.0%} over two weeks), "
                f"above the {limit:.2%} daily limit")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class RiskBudget:
    """Caps one trade at its share of equity, shrinking rather than refusing.

    This is the first gate that can return SHRINK. If four contracts exceed the
    per-trade budget but one does not, buying one is the correct answer -- the
    rule is about how much of the account a single idea may command, and a
    smaller position honours it exactly.

    Dropping to zero is a refusal, not a shrink: an expensive contract on a
    high-priced underlying sometimes will not fit at all, and skipping it is
    right. The alternative is quietly breaking the rule that created the budget.
    """

    name: str = "risk_budget"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        budget = ctx.account.equity * ctx.settings.risk_per_trade
        per_contract = draft.limit_price * 100

        if per_contract <= 0:
            return Verdict.deny("no usable price for this contract")

        affordable = int(budget // per_contract)
        affordable = min(affordable, ctx.settings.max_contracts)

        if affordable < 1:
            return Verdict.deny(
                f"one contract costs ${per_contract:,.0f}, above the "
                f"${budget:,.0f} this trade may use")
        if affordable < draft.quantity:
            return Verdict.shrink(
                affordable,
                f"trimmed to {affordable} to stay inside the ${budget:,.0f} per-trade budget")
        return Verdict.allow()


@dataclass(frozen=True, slots=True)
class BuyingPower:
    """The last word before an order is sent.

    Deliberately placed last and deliberately measured against cash already
    committed to open positions. Every earlier gate reasons about one trade in
    isolation; this one is the only place that asks whether the account can
    actually pay for the whole book at once.
    """

    name: str = "buying_power"

    def check(self, draft: OrderDraft, ctx: GateContext) -> Verdict:
        available = ctx.account.available - ctx.committed
        per_contract = draft.limit_price * 100

        if per_contract <= 0:
            return Verdict.deny("no usable price for this contract")

        affordable = int(available // per_contract)
        if affordable < 1:
            return Verdict.deny(
                f"only ${available:,.0f} uncommitted, and one contract "
                f"costs ${per_contract:,.0f}")
        if affordable < draft.quantity:
            return Verdict.shrink(
                affordable,
                f"trimmed to {affordable} by available cash (${available:,.0f})")
        return Verdict.allow()


# --------------------------------------------------------------------------
# The runners.
# --------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class GateOutcome:
    """The result of running a chain, with the full trace kept.

    The trace is not debug output. Every refusal the agent makes is evidence
    that it is behaving deliberately, and the whole trace is written to the
    journal so the end-of-day summary can say what the agent considered and
    declined, not only what it bought.
    """

    approved: bool
    reason: str
    draft: OrderDraft | None = None
    trace: tuple[tuple[str, Verdict], ...] = ()

    @property
    def refusals(self) -> tuple[tuple[str, Verdict], ...]:
        return tuple((n, v) for n, v in self.trace if v.decision is Decision.DENY)


# The default chains. Order is meaningful in both: cheapest and most absolute
# first, so an expensive check never runs for a trade a free one would refuse.
ENTRY_GATES: tuple[EntryGate, ...] = (
    KillSwitch(),
    MarketOpen(),
    PositionSlots(),
    SectorConcentration(),
    Cooldown(),
    NotAlreadyHeld(),
)

ORDER_GATES: tuple[OrderGate, ...] = (
    MinimumConfidence(),
    DirectionalBalance(),
    DeltaBand(),
    SpreadWidth(),
    ExpiryWindow(),
    # The two price gates sit here deliberately: after the structural checks
    # that decide whether this is the right CONTRACT, and before the sizing
    # gates that decide how much of it. A contract that is the right shape but
    # the wrong price should be refused before anyone works out how many to buy.
    PremiumRichness(),
    DecayBurden(),
    RiskBudget(),
    BuyingPower(),
)


def screen(underlying: str, ctx: GateContext,
           gates: tuple[EntryGate, ...] = ENTRY_GATES) -> GateOutcome:
    """Decide whether an underlying is worth spending a model call on.

    Stops at the first refusal. There is no value in collecting every reason a
    symbol is ineligible when one is sufficient, and short-circuiting is what
    keeps this pass free.
    """
    trace: list[tuple[str, Verdict]] = []
    for gate in gates:
        verdict = gate.check(underlying, ctx)
        trace.append((gate.name, verdict))

        # Entry gates screen a symbol, not a size, so SHRINK is meaningless
        # here. Treating it as a programming error rather than ignoring it stops
        # a future gate from failing silently in the wrong chain.
        assert verdict.decision is not Decision.SHRINK, (
            f"entry gate {gate.name} returned SHRINK, which only order gates may do")

        if verdict.decision is Decision.DENY:
            return GateOutcome(False, verdict.reason, None, tuple(trace))

    return GateOutcome(True, "passed entry screening", None, tuple(trace))


def authorise(draft: OrderDraft, ctx: GateContext,
              gates: tuple[OrderGate, ...] = ORDER_GATES) -> GateOutcome:
    """Run the full chain on a sized order and return what may actually be sent.

    Unlike `screen`, this does not stop at the first SHRINK -- it carries the
    reduced size forward so that later gates judge the order as it now stands.
    A draft trimmed by the risk budget must still be checked against available
    cash, and checking the original size there would defeat the trim.

    The invariant is enforced, not assumed: a gate that tries to raise the
    quantity raises AssertionError. Written as an assert on purpose -- it is a
    claim about this code being correct, not a condition the market can cause.
    """
    trace: list[tuple[str, Verdict]] = []
    current = draft

    for gate in gates:
        verdict = gate.check(current, ctx)
        trace.append((gate.name, verdict))

        if verdict.decision is Decision.DENY:
            return GateOutcome(False, verdict.reason, None, tuple(trace))

        if verdict.decision is Decision.SHRINK:
            assert verdict.quantity is not None, f"{gate.name} shrank without a quantity"
            assert verdict.quantity < current.quantity, (
                f"gate {gate.name} tried to raise the order from "
                f"{current.quantity} to {verdict.quantity}; gates may only "
                f"reject or shrink")
            assert verdict.quantity > 0, f"{gate.name} shrank to zero; that is a denial"
            current = current.with_quantity(verdict.quantity)

    return GateOutcome(True, "approved", current, tuple(trace))
