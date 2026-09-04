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
    PriceBar,
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
                          EXPIRY, 15.50, 16.00, 0.65, 0.20, 900)


def bars_with_volatility(count: int = 25) -> list[PriceBar]:
    """Alternating closes, so realized volatility is real and known."""
    return [
        PriceBar(day=date(2026, 8, 1), open=310.0, high=314.0, low=309.0,
                 close=(313.0 if i % 2 else 310.0), volume=1_000_000)
        for i in range(count)
    ]


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
                 candidates=None, fail_account=False, spot=313.0):
        self._positions = tuple(positions)
        self._equity = equity
        self._is_open = is_open
        self._candidates = candidates if candidates is not None else (contract(),)
        self._fail_account = fail_account
        self._spot = spot
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

    async def stock_price(self, symbol):
        return self._spot

    async def recent_bars(self, symbol, days=120):
        return bars_with_volatility()

    async def option_chain(self, underlying, **kwargs):
        return [c for c in self._candidates if c.underlying == underlying]

    async def headlines(self, symbol, limit=6):
        return []


class FakeExecutor:
    is_dry_run = True

    def __init__(self, *, fail_open=False, resting=(), fail_orders=False,
                 resting_exits=()):
        self.actions: list[tuple[str, str]] = []
        self._fail_open = fail_open
        self._resting = list(resting)
        self._fail_orders = fail_orders
        self._resting_orders = [
            {"symbol": s, "side": "sell", "id": f"sell-{s}"} for s in resting_exits]

    def open_orders(self):
        if self._fail_orders:
            raise ExecutionError("broker unreachable")
        return list(self._resting_orders) + [
            {"symbol": s, "side": "buy", "id": f"buy-{s}"} for s in self._resting]

    def cancel(self, order_id):
        self.actions.append(("cancel", order_id))
        return True

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


def test_a_resting_order_stops_a_duplicate_on_the_same_name(journal):
    """Found live: four unfilled limit orders, and gates that only counted
    filled positions were still willing to place four more."""
    executor = FakeExecutor(resting=["AAPL261016C00310000"])
    settings = Settings(symbols=("AAPL", "MSFT"))
    asyncio.run(run_pass(
        FakeReader(candidates=(contract("AAPL"), contract("MSFT"))),
        settings=settings, executor=executor,
        proposer=FakeProposer({"AAPL": (0.9, Direction.UP),
                               "MSFT": (0.9, Direction.UP)}),
        journal=journal, notifier=RecordingNotifier(), today=TODAY))

    bought = [symbol for action, symbol in executor.actions if action == "buy"]
    assert not any("AAPL" in s for s in bought)
    assert any("MSFT" in s for s in bought)


def test_an_unreadable_order_book_skips_entries_rather_than_risking_duplicates(journal):
    """Missing a pass costs one opportunity. A second order on a name that
    already has one costs money if the price moves through both."""
    executor = FakeExecutor(fail_orders=True)
    notifier = RecordingNotifier()
    result = run(FakeReader(), executor,
                 FakeProposer({"AAPL": (0.9, Direction.UP)}), journal, notifier)
    assert not any(action == "buy" for action, _ in executor.actions)
    assert result.errors
    assert "alert" in notifier.calls


def test_exits_still_run_when_the_order_book_cannot_be_read(journal):
    """Entries stop; exits must not."""
    stopped = position("AAPL", entry=20.00, current=15.00)
    executor = FakeExecutor(fail_orders=True)
    run(FakeReader(positions=(stopped,)), executor, FakeProposer({}), journal,
        RecordingNotifier())
    assert any(action.startswith("sell") for action, _ in executor.actions)


def test_symbol_work_is_bounded_rather_than_fanned_out(journal):
    """Unlimited fan-out killed the agent: thirty symbols at four reads each is
    120 concurrent calls through one stdio connection, the MCP server closed it,
    and eight consecutive passes died before reaching their own stop checks."""
    peak = 0
    live = 0

    class CountingReader(FakeReader):
        async def option_chain(self, underlying, **kwargs):
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0)
            live -= 1
            return [c for c in self._candidates if c.underlying == underlying]

    symbols = tuple(f"SYM{i}" for i in range(20))
    settings = Settings(symbols=symbols, max_concurrent_symbols=3)
    asyncio.run(run_pass(
        CountingReader(), settings=settings, executor=FakeExecutor(),
        proposer=FakeProposer({}), journal=journal,
        notifier=RecordingNotifier(), today=TODAY))

    # Two chain reads per symbol (calls and puts), three symbols at a time.
    assert peak <= 6, f"{peak} concurrent reads; the ceiling was meant to be 6"


def test_a_higher_ceiling_allows_more_at_once(journal):
    """The bound is a real setting, not a hardcoded serialisation."""
    peak = 0
    live = 0

    class CountingReader(FakeReader):
        async def option_chain(self, underlying, **kwargs):
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0)
            live -= 1
            return []

    settings = Settings(symbols=tuple(f"S{i}" for i in range(20)),
                        max_concurrent_symbols=10)
    asyncio.run(run_pass(
        CountingReader(), settings=settings, executor=FakeExecutor(),
        proposer=FakeProposer({}), journal=journal,
        notifier=RecordingNotifier(), today=TODAY))
    assert peak > 6


# --- Underlying-keyed stops -----------------------------------------------
#
# The change these cover: a 25% fall in a 0.65-delta option is under a 2% move
# in the stock. Keying the stop to premium fires on noise and on volatility
# crushes while the thesis is intact -- and fires hardest on exactly the
# high-volatility names where premium swings most.

def test_opening_a_position_records_where_the_underlying_stood(journal):
    executor = FakeExecutor()
    run(FakeReader(candidates=(contract("AAPL"),)), executor,
        FakeProposer({"AAPL": (0.9, Direction.UP)}), journal, RecordingNotifier())

    held = journal.holding("AAPL261016C00310000")
    assert held is not None
    # Whatever the brief's last close was -- asserted against the fixture rather
    # than a literal, so the test stays true if the bar series changes.
    assert held.entry_spot == pytest.approx(bars_with_volatility()[-1].close)
    assert held.stop_spot < held.entry_spot < held.target_spot


def test_the_stop_sits_below_entry_for_a_call_and_above_for_a_put():
    from agent.exits import stop_and_target
    settings = Settings()
    up_stop, up_target = stop_and_target(100.0, "up", atr_pct=0.02, settings=settings)
    down_stop, down_target = stop_and_target(100.0, "down", atr_pct=0.02,
                                             settings=settings)
    assert up_stop < 100.0 < up_target
    assert down_target < 100.0 < down_stop


def test_a_volatile_stock_gets_a_wider_stop():
    """Distance measured in the stock's own units. A flat 3% is a real move on
    a quiet name and pure noise on a wild one."""
    from agent.exits import stop_and_target
    calm, _ = stop_and_target(100.0, "up", atr_pct=0.01, settings=Settings())
    wild, _ = stop_and_target(100.0, "up", atr_pct=0.05, settings=Settings())
    assert wild < calm


def test_the_underlying_moving_through_the_stop_closes_the_position(journal):
    from agent.exits import ExitReason, check_exit
    from agent.journal import Holding
    holding = Holding("AAPL261016C00310000", "AAPL", "2026-09-04T00:00:00",
                      "up", 313.0, 15.75, stop_spot=300.0, target_spot=340.0)
    # Premium only down 10%, nowhere near the old 25% rule -- but the stock has
    # gone through the level that says the thesis was wrong.
    held = position("AAPL", entry=15.75, current=14.20)
    decision = check_exit(held, SETTINGS, TODAY, holding=holding, spot=298.0)
    assert decision is not None and decision.reason is ExitReason.STOP_LOSS
    assert "against the thesis" in decision.detail


def test_premium_noise_no_longer_stops_out_a_live_thesis(journal):
    """The whole point. Down 30% on premium, but the stock has barely moved and
    has not reached the level that would say we were wrong."""
    from agent.exits import check_exit
    from agent.journal import Holding
    holding = Holding("AAPL261016C00310000", "AAPL", "2026-09-04T00:00:00",
                      "up", 313.0, 15.75, stop_spot=300.0, target_spot=340.0)
    held = position("AAPL", entry=15.75, current=11.00)      # -30% premium
    assert check_exit(held, SETTINGS, TODAY, holding=holding, spot=311.0) is None


def test_a_large_gain_is_banked_even_before_the_underlying_target():
    """Observed live on 2 Sep 2026: a PLTR put stood at +50.7% -- $1,105 -- with
    PLTR at 167.88 against a target of 159.21, and nothing in the system would
    have sold it. The underlying target decides when the THESIS is finished;
    this decides when the POSITION has already paid."""
    from agent.exits import ExitReason, check_exit
    from agent.journal import Holding
    holding = Holding("PLTR261016P00200000", "PLTR", "2026-09-01T13:00:00",
                      "down", 186.38, 21.80, stop_spot=199.97, target_spot=159.21)
    held = position("PLTR", entry=21.80, current=32.85)      # +50.7%
    decision = check_exit(held, SETTINGS, TODAY, holding=holding, spot=167.88)
    assert decision is not None
    assert decision.reason is ExitReason.PREMIUM_TARGET
    assert "still short of the" in decision.detail


def test_a_gain_below_the_backstop_keeps_running():
    """The underlying target still governs anything short of the threshold."""
    from agent.exits import check_exit
    from agent.journal import Holding
    holding = Holding("PLTR261016P00200000", "PLTR", "2026-09-01T13:00:00",
                      "down", 186.38, 21.80, stop_spot=199.97, target_spot=159.21)
    held = position("PLTR", entry=21.80, current=28.00)      # +28%
    assert check_exit(held, SETTINGS, TODAY, holding=holding, spot=172.0) is None


def test_banking_a_gain_is_patient_not_urgent():
    """A winner is not racing anything, so it goes out as a limit rather than
    crossing the spread at market."""
    from agent.exits import check_exit
    from agent.journal import Holding
    holding = Holding("PLTR261016P00200000", "PLTR", "2026-09-01T13:00:00",
                      "down", 186.38, 21.80, stop_spot=199.97, target_spot=159.21)
    held = position("PLTR", entry=21.80, current=32.85)
    assert not check_exit(held, SETTINGS, TODAY, holding=holding, spot=167.88).urgent


def test_banking_a_gain_starts_no_cooldown(journal):
    """A win means the reasoning worked. Nothing to cool off from."""
    journal.record("closed", "PLTR", "PLTR261016P00200000",
                   "premium target -- premium up 50.7%", pnl=1105)
    assert journal.cooling_off(within_days=2) == {}


def test_the_underlying_target_still_wins_when_both_are_reached():
    """When the thesis actually completed, that is the reason recorded -- the
    journal should say the trade worked, not that a backstop caught it."""
    from agent.exits import ExitReason, check_exit
    from agent.journal import Holding
    holding = Holding("PLTR261016P00200000", "PLTR", "2026-09-01T13:00:00",
                      "down", 186.38, 21.80, stop_spot=199.97, target_spot=159.21)
    held = position("PLTR", entry=21.80, current=40.00)
    decision = check_exit(held, SETTINGS, TODAY, holding=holding, spot=155.0)
    assert decision.reason is ExitReason.TAKE_PROFIT


def test_a_collapsed_premium_still_closes_through_the_backstop():
    """The case the underlying cannot see: implied volatility gutting the option
    while the stock does nothing."""
    from agent.exits import ExitReason, check_exit
    from agent.journal import Holding
    holding = Holding("AAPL261016C00310000", "AAPL", "2026-09-04T00:00:00",
                      "up", 313.0, 15.75, stop_spot=300.0, target_spot=340.0)
    held = position("AAPL", entry=15.75, current=7.00)       # -56% premium
    decision = check_exit(held, SETTINGS, TODAY, holding=holding, spot=312.0)
    assert decision is not None
    assert decision.reason is ExitReason.PREMIUM_BACKSTOP
    assert "collapsed, not the thesis" in decision.detail


def test_a_backstop_close_starts_no_cooldown(journal):
    """Nothing was disproved, so there is nothing to cool off from."""
    journal.record("closed", "AAPL", "AAPL261016C00310000",
                   "premium backstop -- the option collapsed", pnl=-800)
    assert journal.cooling_off(within_days=2) == {}


def test_a_position_we_did_not_open_falls_back_to_the_premium_stop():
    """Positions from an earlier system have no recorded entry level. Pretending
    otherwise would leave them silently unmanaged."""
    from agent.exits import ExitReason, check_exit
    held = position("AAPL", entry=20.00, current=15.00)      # -25%
    decision = check_exit(held, SETTINGS, TODAY, holding=None, spot=300.0)
    assert decision is not None and decision.reason is ExitReason.STOP_LOSS
    assert "no entry level recorded" in decision.detail


def test_closing_a_position_clears_its_recorded_levels(journal):
    """A stale row would let a later re-entry inherit levels computed for a
    different trade."""
    journal.open_holding(occ_symbol="X261016C00310000", underlying="AAPL",
                         direction="up", entry_spot=313.0, entry_premium=15.0,
                         stop_spot=300.0, target_spot=340.0)
    assert journal.holding("X261016C00310000") is not None
    journal.close_holding("X261016C00310000")
    assert journal.holding("X261016C00310000") is None


# --- Escalating an exit that did not fill ---------------------------------
#
# Observed live on 2 Sep 2026: a take-profit went out as a patient limit at
# 13:15, sat unfilled, and the next pass would have placed a second sell for the
# same single contract. An unfilled exit is a decision that has not happened --
# for a winner it is a gain not banked, for a stop it is a loss still running.

def test_a_stale_exit_order_is_cancelled_and_escalated_to_market(journal):
    stopped = position("AAPL", entry=20.00, current=15.00)
    executor = FakeExecutor(resting_exits=[stopped.occ_symbol])
    run(FakeReader(positions=(stopped,)), executor, FakeProposer({}), journal,
        RecordingNotifier())

    kinds = [action for action, _ in executor.actions]
    assert "cancel" in kinds
    assert "sell_market" in kinds
    assert "sell_limit" not in kinds


def test_the_escalation_is_recorded_as_such(journal):
    stopped = position("AAPL", entry=20.00, current=15.00)
    run(FakeReader(positions=(stopped,)),
        FakeExecutor(resting_exits=[stopped.occ_symbol]),
        FakeProposer({}), journal, RecordingNotifier())
    detail = journal.recent(limit=5)[0].detail
    assert "did not fill" in detail


def test_only_one_exit_order_exists_per_position_per_pass(journal):
    """The hazard: eight passes, eight sell orders, one contract."""
    stopped = position("AAPL", entry=20.00, current=15.00)
    executor = FakeExecutor(resting_exits=[stopped.occ_symbol])
    run(FakeReader(positions=(stopped,)), executor, FakeProposer({}), journal,
        RecordingNotifier())
    sells = [a for a, _ in executor.actions if a.startswith("sell")]
    assert len(sells) == 1


def test_a_position_with_no_resting_order_still_uses_a_limit(journal):
    """Escalation applies to a retry, not to a first attempt."""
    winner = position("AAPL", entry=10.00, current=15.00)
    executor = FakeExecutor()
    run(FakeReader(positions=(winner,)), executor, FakeProposer({}), journal,
        RecordingNotifier())
    assert ("sell_limit", winner.occ_symbol) in executor.actions


def test_a_resting_buy_order_does_not_trigger_exit_escalation(journal):
    """Only sells count. A resting entry is a different thing entirely."""
    winner = position("AAPL", entry=10.00, current=15.00)
    executor = FakeExecutor(resting=["MSFT261016C00310000"])
    run(FakeReader(positions=(winner,)), executor, FakeProposer({}), journal,
        RecordingNotifier())
    assert ("sell_limit", winner.occ_symbol) in executor.actions


def test_the_summary_reports_refusals(journal):
    notifier = RecordingNotifier()
    run(FakeReader(), FakeExecutor(), FakeProposer({"AAPL": (0.1, Direction.UP)}),
        journal, notifier)
    assert notifier.summary["refusals"]
