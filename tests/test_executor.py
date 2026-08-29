"""Tests for the CLI executor.

No CLI is installed for these and no order is ever placed. The subprocess runner
is injected, so the tests assert on the exact command that *would* have run --
which is the right thing to check here, because the command is the whole
behaviour of this module.

The paper-account tests matter most. Everything else in this system can be wrong
and cost a missed opportunity; this is the file where being wrong costs money.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date

import pytest

from agent.domain import Direction, OpenPosition, OptionContract, OrderDraft, Proposal
from agent.executor import (
    CliExecutor,
    ExecutionError,
    NotAPaperAccount,
    new_client_order_id,
)

PAPER = {"account_number": "PA3SUQU0C4MY", "equity": "93196.02"}
LIVE = {"account_number": "928374651", "equity": "5000.00"}


@dataclass
class FakeResult:
    returncode: int = 0
    stdout: str = "{}"
    stderr: str = ""


class FakeRunner:
    """Records every command and replays canned responses in order."""

    def __init__(self, *responses, account=PAPER):
        import json
        self.commands: list[list[str]] = []
        self.account_json = json.dumps(account)
        self.responses = list(responses)

    def __call__(self, args, timeout):
        self.commands.append(list(args))
        if "account" in args:
            return FakeResult(stdout=self.account_json)
        if self.responses:
            nxt = self.responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return FakeResult(stdout='{"id": "order-123", "status": "accepted"}')

    @property
    def last(self) -> list[str]:
        return self.commands[-1]

    def flag(self, name: str) -> str | None:
        """The value following a flag in the last command, or None."""
        cmd = self.last
        return cmd[cmd.index(name) + 1] if name in cmd else None


def draft(quantity: int = 2, limit: float = 15.80) -> OrderDraft:
    contract = OptionContract("AAPL261016C00310000", "AAPL", "call", 310.0,
                              date(2026, 10, 16), 15.50, 16.00, 0.65, 0.30, 900)
    proposal = Proposal("AAPL", Direction.UP, 0.75, "trend intact")
    return OrderDraft(proposal, contract, quantity, limit)


def position(quantity: int = 2) -> OpenPosition:
    return OpenPosition("AAPL261016C00310000", "AAPL", quantity, 15.80, 12.00,
                        date(2026, 10, 16))


# --- The paper-account guard ----------------------------------------------

def test_a_live_account_is_refused():
    runner = FakeRunner(account=LIVE)
    with pytest.raises(NotAPaperAccount):
        CliExecutor(dry_run=False, runner=runner).buy_to_open(draft())


def test_a_live_account_is_refused_before_any_order_command_runs():
    """The refusal must happen before the submit, not alongside it."""
    runner = FakeRunner(account=LIVE)
    with pytest.raises(NotAPaperAccount):
        CliExecutor(dry_run=False, runner=runner).buy_to_open(draft())
    assert not any("submit" in cmd for cmd in runner.commands)


def test_the_paper_check_runs_before_every_submit_not_once():
    """An environment can change under a long-running process. The cost of
    re-checking is one fast request."""
    runner = FakeRunner()
    executor = CliExecutor(runner=runner)
    executor.buy_to_open(draft())
    executor.buy_to_open(draft())
    account_calls = [c for c in runner.commands if "account" in c]
    assert len(account_calls) == 2


def test_selling_is_also_guarded():
    runner = FakeRunner(account=LIVE)
    with pytest.raises(NotAPaperAccount):
        CliExecutor(dry_run=False, runner=runner).sell_to_close(position(), 12.0)


def test_market_close_is_also_guarded():
    runner = FakeRunner(account=LIVE)
    with pytest.raises(NotAPaperAccount):
        CliExecutor(dry_run=False, runner=runner).close_at_market(position())


# --- Dry run is the default ------------------------------------------------

def test_dry_run_is_the_default():
    """Placing real orders must be something a caller asks for, never what
    happens because a flag went missing."""
    assert CliExecutor(runner=FakeRunner()).is_dry_run


def test_a_dry_run_passes_the_flag_to_the_cli():
    runner = FakeRunner()
    CliExecutor(dry_run=True, runner=runner).buy_to_open(draft())
    assert "--dry-run" in runner.last


def test_a_live_run_does_not_pass_the_dry_run_flag():
    runner = FakeRunner()
    CliExecutor(dry_run=False, runner=runner).buy_to_open(draft())
    assert "--dry-run" not in runner.last


# --- The command we build --------------------------------------------------

def test_a_buy_names_the_contract_quantity_side_and_limit():
    runner = FakeRunner()
    CliExecutor(runner=runner).buy_to_open(draft(quantity=3, limit=15.80))

    assert runner.flag("--symbol") == "AAPL261016C00310000"
    assert runner.flag("--qty") == "3"
    assert runner.flag("--side") == "buy"
    assert runner.flag("--type") == "limit"
    assert runner.flag("--limit-price") == "15.80"
    assert runner.flag("--time-in-force") == "day"


def test_a_buy_states_its_position_intent_rather_than_leaving_it_inferred():
    """Without this the broker infers intent from current holdings, and
    inference is not what you want on an account holding other things."""
    runner = FakeRunner()
    CliExecutor(runner=runner).buy_to_open(draft())
    assert runner.flag("--position-intent") == "buy_to_open"


def test_a_sell_closes_rather_than_opening_a_short():
    runner = FakeRunner()
    CliExecutor(runner=runner).sell_to_close(position(), 12.00)
    assert runner.flag("--side") == "sell"
    assert runner.flag("--position-intent") == "sell_to_close"
    assert runner.flag("--limit-price") == "12.00"


def test_the_escalation_path_uses_a_market_order():
    """A missed exit is not harmless: the loss keeps growing while a patient
    limit sits unfilled."""
    runner = FakeRunner()
    CliExecutor(runner=runner).close_at_market(position())
    assert runner.flag("--type") == "market"
    assert "--limit-price" not in runner.last


def test_the_limit_price_is_always_formatted_to_two_decimals():
    runner = FakeRunner()
    CliExecutor(runner=runner).buy_to_open(draft(limit=15.8))
    assert runner.flag("--limit-price") == "15.80"


def test_the_command_is_a_list_never_a_shell_string():
    """A command re-parsed by a shell is a command whose meaning can be
    changed by its own arguments."""
    runner = FakeRunner()
    CliExecutor(runner=runner).buy_to_open(draft())
    assert isinstance(runner.last, list)
    assert all(isinstance(part, str) for part in runner.last)


def test_no_credential_ever_appears_in_the_command():
    """Credentials travel in the environment because argv is visible to every
    other process on the machine."""
    runner = FakeRunner()
    CliExecutor(runner=runner).buy_to_open(draft())
    joined = " ".join(runner.last).lower()
    for secret in ("api_key", "secret", "apca", "password", "token"):
        assert secret not in joined


# --- Client order ids ------------------------------------------------------

def test_every_order_carries_a_client_order_id():
    runner = FakeRunner()
    CliExecutor(runner=runner).buy_to_open(draft())
    assert runner.flag("--client-order-id")


def test_client_order_ids_are_unique_across_orders_in_the_same_second():
    ids = {new_client_order_id("AAPL261016C00310000") for _ in range(200)}
    assert len(ids) == 200


def test_a_client_order_id_carries_the_symbol_and_our_prefix():
    generated = new_client_order_id("AAPL261016C00310000")
    assert generated.startswith("ata-")
    assert "AAPL261016C00310000" in generated


def test_a_client_order_id_fits_alpacas_limit():
    assert len(new_client_order_id("AAPL261016C00310000")) <= 128


# --- Receipts --------------------------------------------------------------

def test_a_live_receipt_carries_the_brokers_order_id():
    runner = FakeRunner(FakeResult(stdout='{"id": "abc-123", "status": "accepted"}'))
    receipt = CliExecutor(dry_run=False, runner=runner).buy_to_open(draft())
    assert receipt.order_id == "abc-123"
    assert receipt.status == "accepted"
    assert not receipt.dry_run


def test_a_dry_run_receipt_says_so_and_has_no_order_id():
    """A dry run echoes the request body rather than a created order, so there
    is nothing to record as an id -- but a receipt still comes back, because
    the caller's job is the same either way."""
    runner = FakeRunner(FakeResult(
        stdout='{"client_order_id": "ata-x", "qty": "2", "type": "limit"}'))
    receipt = CliExecutor(dry_run=True, runner=runner).buy_to_open(draft())
    assert receipt.dry_run
    assert receipt.order_id == ""
    assert receipt.status == "validated"


def test_a_receipt_reports_what_was_asked_for():
    receipt = CliExecutor(runner=FakeRunner()).buy_to_open(draft(quantity=4, limit=9.25))
    assert receipt.quantity == 4
    assert receipt.limit_price == 9.25
    assert receipt.symbol == "AAPL261016C00310000"


# --- Failures --------------------------------------------------------------

def test_a_rejection_surfaces_the_brokers_message():
    """stderr carries the reason, which is the most useful thing to journal
    when an order fails."""
    runner = FakeRunner(FakeResult(returncode=1, stderr="insufficient buying power"))
    with pytest.raises(ExecutionError, match="insufficient buying power"):
        CliExecutor(runner=runner).buy_to_open(draft())


def test_a_timeout_becomes_an_execution_error():
    runner = FakeRunner(subprocess.TimeoutExpired(cmd="alpaca", timeout=45))
    with pytest.raises(ExecutionError, match="timed out"):
        CliExecutor(runner=runner).buy_to_open(draft())


def test_a_missing_binary_says_how_to_install_it():
    runner = FakeRunner(FileNotFoundError())
    with pytest.raises(ExecutionError, match="not on PATH"):
        CliExecutor(runner=runner).buy_to_open(draft())


def test_unparseable_output_becomes_an_execution_error():
    runner = FakeRunner(FakeResult(stdout="not json at all"))
    with pytest.raises(ExecutionError, match="could not parse"):
        CliExecutor(runner=runner).buy_to_open(draft())


def test_cancelling_an_already_filled_order_is_not_an_error():
    """The normal race between a scheduled pass and a moving market."""
    runner = FakeRunner(FakeResult(returncode=1, stderr="order not cancelable"))
    assert CliExecutor(runner=runner).cancel("abc-123") is False


def test_cancelling_an_open_order_reports_success():
    assert CliExecutor(runner=FakeRunner()).cancel("abc-123") is True


# --- What this module cannot do -------------------------------------------

def test_the_executor_cannot_originate_a_trade():
    """It takes an OrderDraft, which only the entry bridge can produce and only
    the gate chain can approve. There is no method that builds one."""
    public = [name for name in dir(CliExecutor) if not name.startswith("_")]
    for name in public:
        assert "decide" not in name and "propose" not in name and "size" not in name
