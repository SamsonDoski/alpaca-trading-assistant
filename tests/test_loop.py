"""Tests for the pass loop, the journal and the exit rules.

The loop is sequencing, so these tests assert on *order* and on *what was
called* rather than on returned values. Every collaborator is a fake: no market,
no broker, no model, no webhook. The ordering guarantees -- exits before entries,
journal before Discord -- are the properties that make a partial failure fail
safely, so they get tests of their own rather than being left as intentions in a
docstring.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta

import pytest

from agent.domain import (
    AccountState,
    Direction,
    MarketBrief,
    OpenPosition,
    OptionContract,
    Proposal,
)
from agent.exits import ExitReason, check_exit, exit_limit_price
from agent.executor import ExecutionError, OrderReceipt
from agent.journal import Journal
from agent.loop import run_pass
from agent.market import MarketDataError
from agent.settings import Settings

TODAY = date(2026, 9, 4)
EXPIRY = date(2026, 10, 16)
SETTINGS = Settings(symbols=("AAPL", "MSFT"))


def position(underlying="AAPL", *, quantity=2, entry=15.80, current=15.80,
             expiry=EXPIRY) -> OpenPosition:
    return OpenPosition(f"{underlying}261016C00310000", underlying, quantity,
                        entry, current, expiry)


def contract(underlying="AAPL") -> OptionContract:
    return OptionContract(f"{underlying}261016C00310000", underlying, "call", 310.0,
                          EXPIRY, 15.50, 16.00, 0.65, 0.30, 900)


# --- Exit rules ------------------------------------------------------------

def test_a_position_at_the_stop_is_closed():
    held = position(entry=20.00, current=15.00)      # -25%
    decision = check_exit(held, SETTINGS, TODAY)
    assert decision.reason is ExitReason.STOP_LOSS


def test_a_position_at_the_target_is_closed():
    held = position(entry=10.00, current=15.00)      # +50%
    decision = check_exit(held, SETTINGS, TODAY)
    assert decision.reason is ExitReason.TAKE_PROFIT


def test_a_position_between_the_levels_is_left_alone():
    assert check_exit(position(entry=15.00, current=16.00), SETTINGS, TODAY) is None


def test_expiry_closes_a_position_regardless_of_profit():
    """A contract in its final week stops behaving like the directional bet it
    was opened as, whether it is winning or losing."""
    winning = position(entry=10.00, current=12.00, expiry=TODAY + timedelta(days=5))
    decision = check_exit(winning, SETTINGS, TODAY)
    assert decision.reason is ExitReason.EXPIRY


def test_expiry_outranks_the_stop():
    both = position(entry=20.00, current=15.00, expiry=TODAY + timedelta(days=3))
    assert check_exit(both, SETTINGS, TODAY).reason is ExitReason.EXPIRY


def test_a_stop_is_urgent_and_a_target_is_not():
    """A stop is racing a position moving against us; a target is not racing
    anything."""
    stop = check_exit(position(entry=20.00, current=15.00), SETTINGS, TODAY)
    target = check_exit(position(entry=10.00, current=15.00), SETTINGS, TODAY)
    assert stop.urgent
    assert not target.urgent


def test_full_exit_aggression_hits_the_bid():
    assert exit_limit_price(11.90, 12.10, 1.0) == pytest.approx(11.90)


def test_zero_exit_aggression_asks_the_ask():
    assert exit_limit_price(11.90, 12.10, 0.0) == pytest.approx(12.10)


# --- The journal -----------------------------------------------------------

@pytest.fixture
def journal(tmp_path) -> Journal:
    return Journal(tmp_path / "test.db")


def test_a_stop_loss_close_starts_a_cooldown(journal):
    journal.record("closed", "AAPL", "AAPL261016C00310000",
                   "stop loss — down 26%", pnl=-400)
    cooling = journal.cooling_off(within_days=2)
    assert cooling["AAPL"] == 2


def test_a_take_profit_close_starts_no_cooldown(journal):
    """A win means the reasoning worked, and re-entering after a win is not the
    behaviour the cooldown exists to prevent."""
    journal.record("closed", "AAPL", "AAPL261016C00310000",
                   "take profit — up 51%", pnl=800)
    assert journal.cooling_off(within_days=2) == {}


def test_a_cooldown_expires(journal):
    journal.record("closed", "AAPL", "AAPL261016C00310000", "stop loss — down 26%",
                   pnl=-400)
    later = date.today() + timedelta(days=5)
    assert journal.cooling_off(within_days=2, as_of=later) == {}


def test_the_journal_records_refusals_not_only_trades(journal):
    journal.record_decision("MSFT", approved=False, reason="already holding")
    rows = journal.decisions_for_day()
    assert len(rows) == 1
    assert not rows[0].approved
    assert rows[0].reason == "already holding"


def test_a_decision_keeps_the_models_reasoning(journal):
    proposal = Proposal("AAPL", Direction.UP, 0.72, "trend intact", "weighed the trend")
    journal.record_decision("AAPL", approved=True, reason="approved", proposal=proposal)
    row = journal.decisions_for_day()[0]
    assert row.confidence == pytest.approx(0.72)
    assert row.thinking == "weighed the trend"


def test_realised_and_lifetime_totals_add_up(journal):
    journal.record("closed", "AAPL", "A", "stop loss", pnl=-400)
    journal.record("closed", "MSFT", "M", "take profit", pnl=900)
    assert journal.realized_for_day() == pytest.approx(500)
    assert journal.summary()["win_rate"] == pytest.approx(0.5)


# --- Fakes for the loop ----------------------------------------------------

class FakeReader:
    def __init__(self, *, positions=(), equity=100_000, is_open=True,
                 candidates=None, fail_account=False):
        self._positions = tuple(positions)
        self._equity = equity
        self._is_open = is_open
        self._candidates = candidates if candidates is not None else (contract(),)
        self._fail_account = fail_account
        self.reads = 0

    async def account(self):
        if self._fail_account:
            raise MarketDataError("broker unreachable")
        self.reads += 1
        return AccountState(self._equity, self._equity, self._equity)

    async def positions(self):
        return self._positions

    async def market_open(self):
        return self._is_open

    async def option_quote(self, occ_symbol):
        return (11.90, 12.10)

    async def recent_bars(self, symbol, days=120):
        return []

    async def option_chain(self, underlying, **kwargs):
        return [c for c in self._candidates if c.underlying == underlying]

    async def headlines(self, symbol, limit=6):
        return []


class FakeExecutor:
    is_dry_run = True

    def __init__(self, *, fail_open=False):
        self.actions: list[tuple[str, str]] = []
        self._fail_open = fail_open

    def _receipt(self, symbol, qty):
        return OrderReceipt("ata-x", symbol, qty, None, status="validated", dry_run=True)

    def buy_to_open(self, draft):
        if self._fail_open:
            raise ExecutionError("insufficient buying power")
        self.actions.append(("buy", draft.contract.occ_symbol))
        return self._receipt(draft.contract.occ_symbol, draft.quantity)

    def sell_to_close(self, position, limit_price):
        self.actions.append(("sell_limit", position.occ_symbol))
        return self._receipt(position.occ_symbol, position.quantity)

    def close_at_market(self, position):
        self.actions.append(("sell_market", position.occ_symbol))
        return self._receipt(position.occ_symbol, position.quantity)


class FakeProposer:
    def __init__(self, answers: dict):
        self.answers = answers
        self.asked: list[str] = []

    def propose(self, brief: MarketBrief) -> Proposal:
        self.asked.append(brief.underlying)
        confidence, direction = self.answers.get(brief.underlying, (0.0, Direction.UP))
        return Proposal(brief.underlying, direction, confidence, "test view")


class RecordingNotifier:
    def __init__(self):
        self.calls: list[str] = []

    def opened(self, *a, **k):
        self.calls.append("opened")

    def closed(self, *a, **k):
        self.calls.append("closed")

    def alert(self, *a, **k):
        self.calls.append("alert")

    def pass_summary(self, **k):
        self.calls.append("summary")
        self.summary = k


def run(reader, executor, proposer, journal, notifier, **kwargs):
    return asyncio.run(run_pass(
        reader, settings=SETTINGS, executor=executor, proposer=proposer,
        journal=journal, notifier=notifier, today=TODAY, **kwargs))


# --- Sequencing, which is what this module actually owns -------------------

def test_exits_run_before_entries(journal):
    """A pass that dies partway through must not have opened something new
    while leaving a stopped-out position untouched."""
    stopped = position("AAPL", entry=20.00, current=15.00)
    executor = FakeExecutor()
    run(FakeReader(positions=(stopped,),
                   candidates=(contract("AAPL"), contract("MSFT"))), executor,
        FakeProposer({"MSFT": (0.9, Direction.UP)}), journal, RecordingNotifier())

    kinds = [action for action, _ in executor.actions]
    assert kinds[0].startswith("sell")
    assert "buy" in kinds


def test_a_stopped_position_is_closed_at_market_not_at_a_limit(journal):
    stopped = position("AAPL", entry=20.00, current=15.00)
    executor = FakeExecutor()
    run(FakeReader(positions=(stopped,)), executor, FakeProposer({}), journal,
        RecordingNotifier())
    assert ("sell_market", stopped.occ_symbol) in executor.actions


def test_a_target_is_closed_at_a_limit(journal):
    winner = position("AAPL", entry=10.00, current=15.00)
    executor = FakeExecutor()
    run(FakeReader(positions=(winner,)), executor, FakeProposer({}), journal,
        RecordingNotifier())
    assert ("sell_limit", winner.occ_symbol) in executor.actions


def test_a_held_symbol_never_reaches_the_model(journal):
    """The screen is free; the model call is not."""
    proposer = FakeProposer({"AAPL": (0.9, Direction.UP), "MSFT": (0.9, Direction.UP)})
    run(FakeReader(positions=(position("AAPL"),)), FakeExecutor(), proposer, journal,
        RecordingNotifier())
    assert "AAPL" not in proposer.asked
    assert "MSFT" in proposer.asked


def test_a_closed_market_does_nothing_at_all(journal):
    executor = FakeExecutor()
    proposer = FakeProposer({"AAPL": (0.9, Direction.UP)})
    result = run(FakeReader(is_open=False, positions=(position(),)), executor,
                 proposer, journal, RecordingNotifier())
    assert executor.actions == []
    assert proposer.asked == []
    assert result.opened == 0


def test_the_kill_switch_stops_buying_but_not_selling(journal):
    stopped = position("AAPL", entry=20.00, current=15.00)
    executor = FakeExecutor()
    run(FakeReader(positions=(stopped,)), executor,
        FakeProposer({"MSFT": (0.9, Direction.UP)}), journal, RecordingNotifier(),
        trading_halted=True)

    kinds = [action for action, _ in executor.actions]
    assert any(k.startswith("sell") for k in kinds)
    assert "buy" not in kinds


def test_candidates_are_ranked_by_conviction(journal):
    """When slots or cash run out, the weakest ideas should be the ones that
    miss out -- not whichever happened to be later in the watchlist.

    This test caught a real bug: the position-slot gate runs in the entry
    screen, which happens once per pass before any of that pass's orders
    exist. Nothing re-checked it afterwards, so a pass with more candidates
    than free slots opened every one of them."""
    settings = Settings(symbols=("AAPL", "MSFT"), max_positions=1)
    executor = FakeExecutor()
    reader = FakeReader(candidates=(contract("AAPL"), contract("MSFT")))
    asyncio.run(run_pass(
        reader, settings=settings, executor=executor,
        proposer=FakeProposer({"AAPL": (0.55, Direction.UP),
                               "MSFT": (0.95, Direction.UP)}),
        journal=journal, notifier=RecordingNotifier(), today=TODAY))

    assert len(executor.actions) == 1
    assert "MSFT" in executor.actions[0][1]


def test_an_unreadable_account_aborts_without_trading(journal):
    """Guessing at the account means trading against something we cannot see."""
    executor = FakeExecutor()
    notifier = RecordingNotifier()
    result = run(FakeReader(fail_account=True), executor,
                 FakeProposer({"AAPL": (0.9, Direction.UP)}), journal, notifier)
    assert executor.actions == []
    assert result.errors
    assert "alert" in notifier.calls


def test_a_failed_order_is_recorded_and_the_pass_continues(journal):
    executor = FakeExecutor(fail_open=True)
    notifier = RecordingNotifier()
    result = run(FakeReader(candidates=(contract("AAPL"), contract("MSFT"))),
                 executor, FakeProposer({"AAPL": (0.9, Direction.UP),
                                         "MSFT": (0.9, Direction.UP)}),
                 journal, notifier)
    assert len(result.errors) == 2      # both attempted, both failed, neither fatal
    assert "alert" in notifier.calls


def test_every_symbol_considered_leaves_a_journal_row(journal):
    run(FakeReader(positions=(position("AAPL"),)), FakeExecutor(),
        FakeProposer({"MSFT": (0.1, Direction.UP)}), journal, RecordingNotifier())
    underlyings = {row.underlying for row in journal.decisions_for_day()}
    assert underlyings == {"AAPL", "MSFT"}


def test_the_pass_always_ends_with_a_summary(journal):
    notifier = RecordingNotifier()
    run(FakeReader(), FakeExecutor(), FakeProposer({}), journal, notifier)
    assert notifier.calls[-1] == "summary"


def test_the_summary_reports_refusals(journal):
    notifier = RecordingNotifier()
    run(FakeReader(), FakeExecutor(), FakeProposer({"AAPL": (0.1, Direction.UP)}),
        journal, notifier)
    assert notifier.summary["refusals"]
