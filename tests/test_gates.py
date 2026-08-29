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
        implied_volatility=0.28,
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
    return OrderDraft(proposal, contract, quantity, limit_price=contract.mid)


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
    assert names == ["minimum_confidence", "delta_band", "spread_width",
                     "expiry_window", "risk_budget", "buying_power"]


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
