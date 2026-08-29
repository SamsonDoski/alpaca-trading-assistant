"""Tests for the entry bridge.

The bridge is where a sentence from a model becomes a number of contracts, so
these tests care about two things above all: that the contract chosen is the one
the rules say to choose, and that the quantity arriving at the far end is the
gates' answer rather than anything this module decided on its own.
"""

from __future__ import annotations

from datetime import date

import pytest

from agent.domain import (
    AccountState,
    Direction,
    MarketBrief,
    OpenPosition,
    OptionContract,
    Proposal,
)
from agent.entry import decide_entry, is_monthly, limit_price, pick_contract
from agent.gates import GateContext
from agent.settings import Settings

TODAY = date(2026, 9, 4)
MONTHLY = date(2026, 10, 16)      # third Friday of October 2026, 42 days out
WEEKLY = date(2026, 10, 9)        # a Friday, but not the third one, 35 days out

SETTINGS = Settings()


def contract(occ="AAPL261016C00310000", *, right="call", strike=310.0, expiry=MONTHLY,
             bid=15.50, ask=16.00, delta=0.65) -> OptionContract:
    return OptionContract(occ, "AAPL", right, strike, expiry, bid, ask, delta, 0.30, 900)


def context(**overrides) -> GateContext:
    defaults = dict(
        today=TODAY,
        market_open=True,
        trading_halted=False,
        account=AccountState(equity=100_000, options_buying_power=100_000, cash=100_000),
        open_positions=(),
        cooling_off={},
        settings=SETTINGS,
    )
    return GateContext(**{**defaults, **overrides})


def brief(*candidates) -> MarketBrief:
    return MarketBrief(underlying="AAPL", as_of=TODAY, bars=(), candidates=tuple(candidates))


def proposal(direction=Direction.UP, confidence=0.75, rationale="trend intact") -> Proposal:
    return Proposal("AAPL", direction, confidence, rationale)


# --- Recognising a monthly expiry -----------------------------------------

def test_the_third_friday_is_a_monthly():
    assert is_monthly(contract(expiry=date(2026, 10, 16)))


def test_a_non_third_friday_is_not_a_monthly():
    assert not is_monthly(contract(expiry=date(2026, 10, 9)))


def test_a_thursday_in_the_third_week_is_not_a_monthly():
    assert not is_monthly(contract(expiry=date(2026, 10, 15)))


# --- Choosing the contract -------------------------------------------------

def test_the_contract_closest_to_the_middle_of_the_delta_band_wins():
    """The band is 0.55-0.75, so the target is 0.65."""
    near = contract("A", delta=0.65)
    far = contract("B", delta=0.56)
    chosen = pick_contract([far, near], Direction.UP, SETTINGS)
    assert chosen.occ_symbol == "A"


def test_a_monthly_beats_a_closer_weekly():
    """Liquidity outranks a small delta difference, because the spread cost of
    an illiquid expiry is larger than the edge on an average trade."""
    weekly = contract("WEEKLY", expiry=WEEKLY, delta=0.65)
    monthly = contract("MONTHLY", expiry=MONTHLY, delta=0.62)
    chosen = pick_contract([weekly, monthly], Direction.UP, SETTINGS)
    assert chosen.occ_symbol == "MONTHLY"


def test_the_tighter_spread_decides_between_equivalent_deltas():
    """0.647 and 0.652 are the same trade. Rounding groups them so that the
    real difference -- the spread -- is what chooses."""
    wide = contract("WIDE", delta=0.647, bid=15.00, ask=16.50)
    tight = contract("TIGHT", delta=0.652, bid=15.45, ask=15.55)
    chosen = pick_contract([wide, tight], Direction.UP, SETTINGS)
    assert chosen.occ_symbol == "TIGHT"


def test_a_contract_with_no_bid_is_never_chosen():
    """A one-sided quote is not a market, and its midpoint is a fiction that
    would flow straight into the sizing arithmetic."""
    assert pick_contract([contract(bid=0.0, ask=16.0)], Direction.UP, SETTINGS) is None


def test_a_down_view_selects_puts_not_calls():
    call = contract("CALL", right="call", delta=0.65)
    put = contract("PUT", right="put", delta=-0.65)
    chosen = pick_contract([call, put], Direction.DOWN, SETTINGS)
    assert chosen.occ_symbol == "PUT"


def test_a_put_is_matched_on_absolute_delta():
    """Puts quote negative delta; -0.65 and +0.65 are the same distance from
    the target."""
    assert pick_contract([contract("P", right="put", delta=-0.65)],
                         Direction.DOWN, SETTINGS) is not None


def test_nothing_in_the_delta_band_returns_none():
    assert pick_contract([contract(delta=0.20)], Direction.UP, SETTINGS) is None


def test_an_empty_chain_returns_none():
    assert pick_contract([], Direction.UP, SETTINGS) is None


# --- Pricing the order -----------------------------------------------------

def test_the_limit_sits_inside_the_spread_at_the_configured_aggression():
    # 15.50 bid, 16.00 ask, 0.6 aggression -> 15.50 + 0.6 * 0.50 = 15.80
    assert limit_price(contract(), 0.6) == pytest.approx(15.80)


def test_zero_aggression_sits_on_the_bid():
    assert limit_price(contract(), 0.0) == pytest.approx(15.50)


def test_full_aggression_sits_on_the_ask():
    assert limit_price(contract(), 1.0) == pytest.approx(16.00)


def test_the_limit_is_rounded_to_the_cent():
    priced = limit_price(contract(bid=1.111, ask=2.999), 0.5)
    assert priced == round(priced, 2)


# --- The whole decision ----------------------------------------------------

def test_an_approved_entry_carries_a_gate_decided_quantity():
    """4% of $100,000 is $4,000. At $15.80 a share the contract costs $1,580,
    so the budget allows two -- not the 50 the draft asked for."""
    outcome = decide_entry(proposal(), brief(contract()), context())
    assert outcome.approved
    assert outcome.draft.quantity == 2


def test_the_draft_asks_for_appetite_and_the_gates_impose_reality():
    """The requested quantity is the strategy ceiling; every reduction from
    there is the gate chain's doing, which is why the budget formula lives in
    exactly one place."""
    outcome = decide_entry(proposal(), brief(contract()), context())
    trimmed = [name for name, verdict in outcome.trace
               if verdict.decision.value == "shrink"]
    assert "risk_budget" in trimmed


def test_a_bigger_account_is_allowed_a_bigger_position():
    rich = context(account=AccountState(equity=1_000_000,
                                        options_buying_power=1_000_000, cash=1_000_000))
    outcome = decide_entry(proposal(), brief(contract()), rich)
    assert outcome.draft.quantity == 25


def test_the_quantity_never_exceeds_the_strategy_ceiling():
    """Even an account that could afford hundreds is capped by max_contracts."""
    huge = context(account=AccountState(equity=100_000_000,
                                        options_buying_power=100_000_000,
                                        cash=100_000_000))
    outcome = decide_entry(proposal(), brief(contract()), huge)
    assert outcome.draft.quantity == SETTINGS.max_contracts


def test_a_declined_proposal_never_reaches_the_gates():
    declined = Proposal("AAPL", Direction.UP, 0.0, "evidence is mixed")
    outcome = decide_entry(declined, brief(contract()), context())
    assert not outcome.approved
    assert outcome.reason == "evidence is mixed"
    assert outcome.trace == ()


def test_no_suitable_contract_is_reported_as_such_not_as_a_gate_refusal():
    outcome = decide_entry(proposal(), brief(contract(delta=0.20)), context())
    assert not outcome.approved
    assert "delta band" in outcome.reason


def test_a_low_confidence_view_is_refused_by_the_confidence_gate():
    outcome = decide_entry(proposal(confidence=0.2), brief(contract()), context())
    assert not outcome.approved
    assert "confidence" in outcome.reason


def test_a_wide_spread_is_refused_after_the_contract_is_chosen():
    wide = contract(bid=14.00, ask=18.00)     # 25% of a 16.00 midpoint
    outcome = decide_entry(proposal(), brief(wide), context())
    assert not outcome.approved
    assert "wide" in outcome.reason


def test_cash_committed_elsewhere_shrinks_the_new_position():
    """Two positions worth $49,000 each leave $2,000 uncommitted of a $100,000
    account -- enough for one contract at $1,580, not the two the per-trade
    budget would otherwise allow."""
    held = tuple(
        OpenPosition(f"X{i}261016C00100000", f"X{i}", 10, 49.0, 49.0, date(2026, 10, 16))
        for i in range(2)
    )
    outcome = decide_entry(proposal(), brief(contract()), context(open_positions=held))
    assert outcome.approved
    assert outcome.draft.quantity == 1


def test_an_approved_entry_prices_the_contract_it_chose():
    outcome = decide_entry(proposal(), brief(contract()), context())
    assert outcome.draft.contract.occ_symbol == "AAPL261016C00310000"
    assert outcome.draft.limit_price == pytest.approx(15.80)


def test_every_outcome_carries_a_reason_a_person_can_read():
    cases = [
        decide_entry(Proposal("AAPL", Direction.UP, 0.0, "mixed"), brief(contract()), context()),
        decide_entry(proposal(), brief(), context()),
        decide_entry(proposal(confidence=0.1), brief(contract()), context()),
        decide_entry(proposal(), brief(contract()), context()),
    ]
    for outcome in cases:
        assert outcome.reason.strip()
