"""Turning a directional view into an order the gates have approved.

The model said "up on AAPL, confidence 0.72". This module is what stands between
that sentence and a real order: it picks the contract that expresses the view,
prices it, states how much of it we would take, and then submits the whole thing
to the risk chain for approval.

**Where the size actually gets decided, and why it is only one place.**

The obvious way to write this would be to work out an affordable quantity here,
and then have the gates check it. That would put the same budget arithmetic in
two files, which is how the two versions eventually disagree -- and a disagreement
about position size is not a cosmetic bug.

So the draft is created with the strategy's *appetite*: the most contracts this
strategy would ever want in one position, ignoring the account entirely. The gate
chain then trims that down to what the per-trade budget allows, and trims again
for cash already committed elsewhere. Reality is applied in exactly one place,
by the component whose whole job is applying it.

Read that way, an OrderDraft before the gates is a request ("we would take up to
50 of these") and an OrderDraft after them is a decision ("take 2"). The gates
are the only thing that can turn one into the other, and they can only ever move
the number down.
"""

from __future__ import annotations

from agent.domain import Direction, MarketBrief, OptionContract, OrderDraft, Proposal
from agent.gates import GateContext, GateOutcome, authorise
from agent.settings import Settings


def is_monthly(contract: OptionContract) -> bool:
    """Whether this contract expires on a standard monthly expiry.

    Monthly options expire on the third Friday of the month, and that is where
    the open interest sits. The difference between a liquid expiry and an
    illiquid one measured several percent of the premium in earlier testing --
    larger than the edge on an average trade, so it is worth preferring even
    when a weekly sits closer to the delta target.

    The third Friday is the first Friday falling on the 15th through the 21st.
    """
    return contract.expiry.weekday() == 4 and 15 <= contract.expiry.day <= 21


def pick_contract(candidates: list[OptionContract] | tuple[OptionContract, ...],
                  direction: Direction, settings: Settings) -> OptionContract | None:
    """The contract that best expresses the view, or None if nothing qualifies.

    Ranked on three things in order:

      1. A standard monthly expiry, for the liquidity reason above.
      2. Distance from the middle of the delta band -- rounded to two decimals,
         which is the part worth explaining. Delta differences below 0.01 are
         noise: two strikes at 0.647 and 0.652 are the same trade. Rounding
         groups those together so that the third criterion decides between them
         instead of a meaningless third decimal place.
      3. The tighter quoted spread. Among contracts that are genuinely
         equivalent, this is the one difference that is real money.

    Contracts with no bid are dropped outright. A one-sided quote is not a
    market -- there is nothing to sell it back to -- and its midpoint is a
    fiction that would flow straight into the sizing arithmetic.
    """
    target_delta = (settings.delta_min + settings.delta_max) / 2
    right = direction.option_right

    usable = [
        c for c in candidates
        if c.right == right
        and c.bid > 0
        and settings.delta_min <= c.abs_delta <= settings.delta_max
    ]
    if not usable:
        return None

    def rank(contract: OptionContract) -> tuple[int, float, float]:
        return (
            0 if is_monthly(contract) else 1,
            round(abs(contract.abs_delta - target_delta), 2),
            contract.spread_pct,
        )

    return min(usable, key=rank)


def limit_price(contract: OptionContract, aggression: float) -> float:
    """Where to place the limit inside the spread.

    0.0 sits on the bid, 1.0 on the ask. Entries are patient by default because
    a missed entry costs nothing here -- the same symbol is re-examined on the
    next pass fifteen minutes later, or another candidate takes the slot. Paying
    the whole spread to guarantee a fill would be a certain cost incurred to
    avoid a harmless outcome.

    Rounded to the cent, because that is the smallest increment an option order
    can actually express.
    """
    span = contract.ask - contract.bid
    return round(contract.bid + span * aggression, 2)


def decide_entry(proposal: Proposal, brief: MarketBrief, ctx: GateContext) -> GateOutcome:
    """The whole post-model entry decision, start to finish.

    Returns a GateOutcome either way, so a caller has one thing to handle rather
    than a success type and three failure types. Every path carries a reason
    written for a human, because every one of them ends up in the journal.
    """
    settings = ctx.settings

    # The model declining is the most common outcome by far, and it is a normal
    # one. Its own words become the recorded reason.
    if not proposal.is_actionable:
        return GateOutcome(False, proposal.rationale or "no directional view")

    contract = pick_contract(brief.contracts_for(proposal.direction),
                             proposal.direction, settings)
    if contract is None:
        return GateOutcome(
            False,
            f"no {proposal.direction.option_right} in the "
            f"{settings.delta_min:.2f}-{settings.delta_max:.2f} delta band with a live quote")

    draft = OrderDraft(
        proposal=proposal,
        contract=contract,
        # Appetite, not affordability. See the module docstring: the gates are
        # the single authority on how much the account can actually take.
        quantity=settings.max_contracts,
        limit_price=limit_price(contract, settings.entry_aggression),
    )

    return authorise(draft, ctx)
