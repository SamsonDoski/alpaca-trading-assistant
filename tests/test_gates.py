"""Tests for the risk gates.

These run with no network, no broker and no model, which is the whole reason the
gates take a GateContext instead of fetching what they need. The risk system is
the part of this agent that must not be wrong, so it is also the part that can be
exercised exhaustively in under a second.
"""

from __future__ import annotations

from datetime import date

import pytest

from agent.domain import (
    AccountState,
    Direction,
    OpenPosition,
    OptionContract,
    OrderDraft,
    Proposal,
)
from agent.gates import (
    Decision,
    GateContext,
    authorise,
    screen,
)
from agent.settings import Settings

TODAY = date(2026, 8, 31)


def make_context(**overrides) -> GateContext:
    """A healthy, permissive context. Tests override only what they exercise."""
    defaults = dict(
        today=TODAY,
        market_open=True,
        trading_halted=False,
        account=AccountState(equity=100_000, options_buying_power=100_000, cash=100_000),
        open_positions=(),
        cooling_off={},
        settings=Settings(),
    )
    return GateContext(**{**defaults, **overrides})


def make_contract(**overrides) -> OptionContract:
    """A well-behaved contract: 0.65 delta, tight spread, 35 days out."""
    defaults = dict(
        occ_symbol="AAPL260930C00230000",
        underlying="AAPL",
        right="call",
        strike=230.0,
        expiry=date(2026, 10, 5),   # 35 days from TODAY
        bid=4.95,
        ask=5.05,
        delta=0.65,
        implied_volatility=0.15,
        open_interest=1500,
    )
    return OptionContract(**{**defaults, **overrides})


def make_draft(quantity: int = 1, confidence: float = 0.8, **contract_overrides) -> OrderDraft:
    proposal = Proposal(
        underlying="AAPL",
        direction=Direction.UP,
        confidence=confidence,
        rationale="test proposal",
    )
    contract = make_contract(**contract_overrides)
    # Spot and realized volatility make the contract internally coherent: a
    # $5.00 option struck at 230 on a $232 stock at 15% implied, against a stock
    # actually delivering 15%. The price gates read these, and a fixture whose
    # economics do not hang together would exercise them meaninglessly.
    return OrderDraft(proposal, contract, quantity, limit_price=contract.mid,
                      spot=232.0, realized_vol=0.15)


def make_position(underlying: str = "AAPL", **overrides) -> OpenPosition:
    defaults = dict(
        occ_symbol=f"{underlying}260930C00230000",
        underlying=underlying,
        quantity=1,
        entry_price=5.00,
        current_price=5.00,
        expiry=date(2026, 10, 5),
    )
    return OpenPosition(**{**defaults, **overrides})


# --- Entry screening ------------------------------------------------------

def test_healthy_symbol_passes_screening():
    assert screen("AAPL", make_context()).approved


def test_kill_switch_blocks_entry():
    outcome = screen("AAPL", make_context(trading_halted=True))
    assert not outcome.approved
    assert "kill switch" in outcome.reason


def test_kill_switch_is_checked_before_anything_expensive():
    """The halt must be the first gate, so a frozen run spends nothing."""
    outcome = screen("AAPL", make_context(trading_halted=True))
    assert [name for name, _ in outcome.trace] == ["kill_switch"]


def test_closed_market_blocks_entry():
    outcome = screen("AAPL", make_context(market_open=False))
    assert not outcome.approved
    assert "closed" in outcome.reason


def test_full_position_slots_block_entry():
    positions = tuple(make_position(s) for s in ("SPY", "QQQ", "MSFT", "NVDA", "TSLA"))
    outcome = screen("AAPL", make_context(open_positions=positions))
    assert not outcome.approved
    assert "slots" in outcome.reason


def test_cooldown_blocks_recently_stopped_name():
    outcome = screen("AAPL", make_context(cooling_off={"AAPL": 2}))
    assert not outcome.approved
    assert "cooldown" in outcome.reason


def test_cooldown_does_not_block_a_different_name():
    assert screen("MSFT", make_context(cooling_off={"AAPL": 2})).approved


def test_already_held_blocks_a_second_position():
    outcome = screen("AAPL", make_context(open_positions=(make_position("AAPL"),)))
    assert not outcome.approved
    assert "already holding" in outcome.reason


def test_a_resting_order_blocks_a_second_order_on_the_same_name():
    """Found live: four limit orders sat unfilled for half an hour while the
    gates, which only counted filled positions, remained willing to place four
    more on the same names."""
    outcome = screen("AAPL", make_context(pending=frozenset({"AAPL"})))
    assert not outcome.approved
    assert "resting" in outcome.reason


def test_a_resting_order_elsewhere_does_not_block_this_name():
    assert screen("MSFT", make_context(pending=frozenset({"AAPL"}))).approved


def test_resting_orders_consume_position_slots():
    """An unfilled order has spent a slot and reserved the cash behind it."""
    held = (make_position("SPY"), make_position("QQQ"))
    ctx = make_context(open_positions=held, pending=frozenset({"NVDA", "TSLA", "META"}))
    outcome = screen("AAPL", ctx)
    assert not outcome.approved
    assert "slots" in outcome.reason


def test_the_slot_count_reports_held_and_resting_separately():
    held = (make_position("SPY"),)
    ctx = make_context(open_positions=held,
                       pending=frozenset({"NVDA", "TSLA", "META", "AMD"}))
    outcome = screen("AAPL", ctx)
    assert "1 held" in outcome.reason and "4 resting" in outcome.reason


def test_positions_and_resting_orders_add_up_to_committed_slots():
    ctx = make_context(open_positions=(make_position("SPY"),),
                       pending=frozenset({"NVDA", "TSLA"}))
    assert ctx.committed_slots == 3


def test_screening_short_circuits_at_the_first_refusal():
    """A halted, closed market should report the halt and stop there."""
    outcome = screen("AAPL", make_context(trading_halted=True, market_open=False))
    assert len(outcome.trace) == 1


# --- Order authorisation --------------------------------------------------

def test_healthy_draft_is_approved():
    outcome = authorise(make_draft(quantity=2), make_context())
    assert outcome.approved
    assert outcome.draft is not None
    assert outcome.draft.quantity == 2


def test_low_confidence_is_refused():
    outcome = authorise(make_draft(confidence=0.2), make_context())
    assert not outcome.approved
    assert "confidence" in outcome.reason


def test_delta_outside_the_band_is_refused():
    outcome = authorise(make_draft(delta=0.20), make_context())
    assert not outcome.approved
    assert "delta" in outcome.reason


def test_missing_delta_is_refused_rather_than_assumed():
    outcome = authorise(make_draft(delta=None), make_context())
    assert not outcome.approved
    assert "no delta" in outcome.reason


def test_wide_spread_is_refused():
    # 4.50 / 5.50 is a 20% spread on a 5.00 midpoint.
    outcome = authorise(make_draft(bid=4.50, ask=5.50), make_context())
    assert not outcome.approved
    assert "wide" in outcome.reason


def test_contract_with_no_quote_is_refused_not_treated_as_free():
    """A contract with no market must never look cheap to the spread gate."""
    outcome = authorise(make_draft(bid=0.0, ask=0.0), make_context())
    assert not outcome.approved


def test_expiry_inside_the_closeout_window_is_refused():
    outcome = authorise(make_draft(expiry=date(2026, 9, 3)), make_context())
    assert not outcome.approved
    assert "close-out window" in outcome.reason


def test_expiry_beyond_the_band_is_refused():
    outcome = authorise(make_draft(expiry=date(2026, 12, 18)), make_context())
    assert not outcome.approved
    assert "outside" in outcome.reason


# --- Concentration --------------------------------------------------------
#
# The hole these close: every other gate reasons about ONE trade. Slots counted
# eight, the budget sized each at 4%, and nothing asked whether the eight were
# secretly the same bet.

TECH = {"AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech",
        "GLD": "metals", "SLV": "metals", "TLT": "bonds"}


def grouped(**overrides) -> Settings:
    return Settings(correlation_groups=TECH, **overrides)


def test_a_third_position_in_one_group_is_refused():
    held = (make_position("MSFT"), make_position("NVDA"), make_position("GOOGL"))
    ctx = make_context(open_positions=held,
                       settings=grouped(max_per_group=3))
    outcome = screen("AAPL", ctx)
    assert not outcome.approved
    assert "tech" in outcome.reason


def test_the_refusal_names_which_positions_filled_the_group():
    """A refusal a person can check beats one they have to investigate."""
    held = (make_position("MSFT"), make_position("NVDA"))
    ctx = make_context(open_positions=held, settings=grouped(max_per_group=2))
    outcome = screen("AAPL", ctx)
    assert "MSFT" in outcome.reason and "NVDA" in outcome.reason


def test_a_different_group_is_unaffected():
    held = (make_position("MSFT"), make_position("NVDA"), make_position("GOOGL"))
    ctx = make_context(open_positions=held, settings=grouped(max_per_group=3))
    assert screen("GLD", ctx).approved


def test_an_ungrouped_symbol_is_never_blocked_by_the_cap():
    held = (make_position("MSFT"), make_position("NVDA"), make_position("GOOGL"))
    ctx = make_context(open_positions=held, settings=grouped(max_per_group=3))
    assert screen("IBIT", ctx).approved


def test_no_groups_configured_means_no_cap():
    """Opt-in: the universe that needs grouping is defined in config."""
    held = tuple(make_position(s) for s in ("MSFT", "NVDA", "GOOGL"))
    assert screen("AAPL", make_context(open_positions=held)).approved


def test_a_book_leaning_entirely_one_way_is_refused():
    """Eight well-spread sectors that are all long calls is still one bet."""
    held = tuple(make_position(s) for s in ("SPY", "GLD", "TLT", "XLE", "EEM"))
    ctx = make_context(open_positions=held, settings=Settings(max_same_direction=5))
    outcome = authorise(make_draft(quantity=1), ctx)
    assert not outcome.approved
    assert "lean bullish" in outcome.reason


def test_the_opposite_side_is_still_allowed_when_one_side_is_full():
    held = tuple(make_position(s) for s in ("SPY", "GLD", "TLT", "XLE", "EEM"))
    ctx = make_context(open_positions=held, settings=Settings(max_same_direction=5))
    puts = make_draft(quantity=1, right="put", delta=-0.65,
                      occ_symbol="AAPL260930P00230000")
    assert authorise(puts, ctx).approved


def test_puts_and_calls_are_counted_separately():
    held = tuple(make_position(s, right="put") for s in ("SPY", "GLD", "TLT"))
    ctx = make_context(open_positions=held, settings=Settings(max_same_direction=3))
    assert authorise(make_draft(quantity=1), ctx).approved


# --- What the premium costs -----------------------------------------------
#
# The gap this strategy had: implied volatility was shown to the model and acted
# on by nothing. SPY at 13% IV and SMCI at 70% went through identical machinery,
# which for a premium buyer is trading blind on the one number that decides
# whether the trade is cheap.

def test_a_rich_premium_is_refused():
    """20% implied against 12% realized is 1.67x -- paying well over what the
    stock has actually been delivering."""
    draft = make_draft(implied_volatility=0.20)
    outcome = authorise(OrderDraft(draft.proposal, draft.contract, 1,
                                   draft.limit_price, spot=232.0, realized_vol=0.12),
                        make_context())
    assert not outcome.approved
    assert "rich" in outcome.reason


def test_a_cheap_premium_passes():
    """Implied BELOW realized is the favourable side of the trade."""
    draft = make_draft(implied_volatility=0.12)
    outcome = authorise(OrderDraft(draft.proposal, draft.contract, 1,
                                   draft.limit_price, spot=232.0, realized_vol=0.20),
                        make_context())
    assert outcome.approved


def test_an_unmeasurable_premium_is_refused_not_waved_through():
    """An unmeasurable price is not a cheap one. Missing data blocks."""
    draft = make_draft()
    outcome = authorise(OrderDraft(draft.proposal, draft.contract, 1,
                                   draft.limit_price, spot=232.0, realized_vol=None),
                        make_context())
    assert not outcome.approved
    assert "cannot price" in outcome.reason


def test_the_richness_limit_is_configurable():
    # Both limits are relaxed so this exercises richness alone. Raising implied
    # volatility to make the premium rich also makes it decay faster, so a test
    # that moved only one limit would be refused by the other gate and prove
    # nothing about either.
    draft = make_draft(implied_volatility=0.20)
    permissive = make_context(
        settings=Settings(max_iv_to_realized=2.0, max_daily_decay=0.05))
    outcome = authorise(OrderDraft(draft.proposal, draft.contract, 1,
                                   draft.limit_price, spot=232.0, realized_vol=0.12),
                        permissive)
    assert outcome.approved


def test_a_fast_decaying_contract_is_refused():
    """Long premium is a race between the move and the clock. A contract at 45%
    implied on a $232 stock bleeds far too fast for a two-week thesis."""
    draft = make_draft(implied_volatility=0.45)
    outcome = authorise(OrderDraft(draft.proposal, draft.contract, 1,
                                   draft.limit_price, spot=232.0, realized_vol=0.40),
                        make_context())
    assert not outcome.approved
    assert "decays" in outcome.reason


def test_the_decay_refusal_states_the_two_week_cost():
    """A daily percentage is hard to feel; the cumulative number is the one
    that decides whether a thesis has room to work."""
    draft = make_draft(implied_volatility=0.45)
    outcome = authorise(OrderDraft(draft.proposal, draft.contract, 1,
                                   draft.limit_price, spot=232.0, realized_vol=0.40),
                        make_context())
    assert "over two weeks" in outcome.reason


def test_a_slowly_decaying_contract_passes():
    outcome = authorise(make_draft(quantity=1), make_context())
    assert outcome.approved


def test_the_price_gates_run_before_the_sizing_gates():
    """A contract that is the right shape but the wrong price should be refused
    before anyone works out how many to buy."""
    names = [name for name, _ in authorise(make_draft(quantity=1),
                                           make_context()).trace]
    assert names.index("premium_richness") < names.index("risk_budget")
    assert names.index("decay_burden") < names.index("risk_budget")


# --- Shrinking ------------------------------------------------------------

def test_risk_budget_shrinks_rather_than_refusing():
    """4% of $100k is $4,000; a $500 contract allows 8, so 20 must be trimmed."""
    outcome = authorise(make_draft(quantity=20), make_context())
    assert outcome.approved
    assert outcome.draft.quantity == 8


def test_risk_budget_refuses_when_one_contract_will_not_fit():
    small = make_context(
        account=AccountState(equity=5_000, options_buying_power=5_000, cash=5_000))
    outcome = authorise(make_draft(quantity=1), small)
    assert not outcome.approved
    assert "above the" in outcome.reason


def test_buying_power_accounts_for_cash_already_committed():
    """Open positions consume the budget the next trade is measured against.

    Two positions worth $9,900 each leave $200 of a $20,000 account uncommitted,
    which does not cover a $500 contract. The per-trade risk budget alone would
    have allowed this trade -- 4% of $20,000 is $800 -- so only a gate that reads
    the open book can catch it.
    """
    held = tuple(make_position(s, quantity=11, entry_price=9.0, current_price=9.0)
                 for s in ("SPY", "QQQ"))
    ctx = make_context(
        account=AccountState(equity=20_000, options_buying_power=20_000, cash=20_000),
        open_positions=held,
    )
    outcome = authorise(make_draft(quantity=1), ctx)
    assert not outcome.approved
    assert "uncommitted" in outcome.reason


def test_a_shrink_is_carried_into_later_gates():
    """The trimmed size, not the original, is what buying power must judge."""
    ctx = make_context(
        account=AccountState(equity=30_000, options_buying_power=1_200, cash=30_000))
    outcome = authorise(make_draft(quantity=10), ctx)
    assert outcome.approved
    # Risk budget allows 2 ($1,200 of $30k equity); cash allows 2. Both agree.
    assert outcome.draft.quantity == 2


# --- The invariants -------------------------------------------------------

def test_gates_can_never_enlarge_an_order():
    """The core safety claim, asserted directly rather than trusted."""

    class GreedyGate:
        name = "greedy"

        def check(self, draft, ctx):
            from agent.gates import Verdict
            return Verdict.shrink(draft.quantity + 100, "more is better")

    with pytest.raises(AssertionError, match="may only"):
        authorise(make_draft(quantity=1), make_context(), gates=(GreedyGate(),))


def test_an_approved_outcome_always_carries_the_draft_it_approved():
    outcome = authorise(make_draft(quantity=1), make_context())
    assert outcome.approved and outcome.draft is not None


def test_a_refused_outcome_never_carries_a_draft():
    outcome = authorise(make_draft(confidence=0.0), make_context())
    assert not outcome.approved and outcome.draft is None


def test_every_verdict_in_a_trace_has_a_reason_when_it_refuses():
    outcome = authorise(make_draft(delta=0.1), make_context())
    for _, verdict in outcome.refusals:
        assert verdict.reason.strip(), "a refusal with no explanation is unusable"


def test_the_trace_records_every_gate_that_ran():
    outcome = authorise(make_draft(quantity=1), make_context())
    names = [name for name, _ in outcome.trace]
    assert names == ["minimum_confidence", "directional_balance", "delta_band",
                     "spread_width", "expiry_window", "premium_richness",
                     "decay_burden", "risk_budget", "buying_power"]


def test_refusals_are_extractable_for_the_journal():
    outcome = authorise(make_draft(delta=0.99), make_context())
    assert len(outcome.refusals) == 1
    assert outcome.refusals[0][0] == "delta_band"


# --- Settings -------------------------------------------------------------

def test_take_profit_is_derived_from_the_stop():
    s = Settings(stop_loss_pct=0.25, reward_to_risk=2.0)
    assert s.take_profit_pct == pytest.approx(0.50)


def test_break_even_win_rate_matches_the_ratio():
    assert Settings(reward_to_risk=2.0).break_even_win_rate == pytest.approx(1 / 3)


def test_verdict_decisions_are_the_only_three():
    assert {d.value for d in Decision} == {"allow", "deny", "shrink"}
