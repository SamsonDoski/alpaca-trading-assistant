"""Placing orders through the Alpaca CLI.

This is the only module in the system that can spend money, and it is written to
be the least clever one. It takes an OrderDraft that the gates have already
approved and turns it into a command. It has no access to the settings, the
model, the gates or the market reader, and it contains no branch that decides
whether a trade is a good idea. Handed a draft, it submits it; handed nothing, it
does nothing.

**Why a subprocess rather than the SDK.** The CLI is a separate program with its
own credentials, invoked with an argument list this code builds explicitly. That
makes the write path physically distinct from the read path: the MCP server the
model reads through and the binary that places orders are different processes,
and no amount of confusion in the reasoning layer can turn one into the other.
It also means every order this agent has ever placed can be reconstructed exactly,
because the command is logged verbatim before it runs.

**Four safety properties, each enforced rather than intended:**

  * The account is confirmed to be a paper account before *every* submit, not
    once at startup.
  * Arguments are passed as a list, never a shell string. Nothing is interpolated
    into a line a shell will parse.
  * Credentials travel in the environment, never in argv, because argv is
    visible to every other process on the machine.
  * Every submit carries a client order id we generated, so a fill can always be
    traced back to the proposal and the gate verdicts that produced it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agent.domain import OpenPosition, OrderDraft

logger = logging.getLogger(__name__)

BINARY = "alpaca"

# A paper account number always begins with PA. This is the same check the
# earlier system used, and it is the last line of defence: if the environment
# ever points at a funded account, nothing here should be willing to trade it.
PAPER_PREFIX = "PA"

DEFAULT_TIMEOUT = 45


class ExecutionError(RuntimeError):
    """An order could not be placed, or could not be confirmed as placed."""


class NotAPaperAccount(ExecutionError):
    """The configured credentials are not a paper account. Refuse everything."""


@dataclass(frozen=True, slots=True)
class OrderReceipt:
    """What came back from a submitted order."""

    client_order_id: str
    symbol: str
    quantity: int
    limit_price: float | None
    order_id: str = ""
    status: str = ""
    dry_run: bool = False
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        prefix = "DRY RUN " if self.dry_run else ""
        price = f" at {self.limit_price:.2f}" if self.limit_price else " at market"
        return (f"{prefix}{self.symbol} x{self.quantity}{price} "
                f"[{self.status or 'submitted'}] {self.client_order_id}")


def new_client_order_id(symbol: str) -> str:
    """A unique, greppable identifier for one order.

    Carries the symbol and a timestamp so a human reading the broker's order
    list can recognise it, plus a short random suffix so two passes firing in
    the same second cannot collide. Alpaca allows 128 characters; this uses
    about forty.

    The prefix marks orders this agent placed, which matters on an account that
    also holds positions opened by something else.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"ata-{symbol}-{stamp}-{uuid.uuid4().hex[:6]}"


def _run_command(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Run the CLI once. The default runner; tests substitute their own.

    `shell=False` is the default and is left that way deliberately. Building a
    command as a string and letting a shell re-parse it is how a symbol or a
    price ends up changing the meaning of the line, and an argument list simply
    cannot be re-interpreted.

    The environment is inherited so the CLI reads ALPACA_API_KEY and
    ALPACA_SECRET_KEY from it. Credentials never appear in `args`, which is what
    keeps them out of `ps` output and out of the log line below.
    """
    logger.info("executing: %s", " ".join(args))
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )


class CliExecutor:
    """Submits option orders through the Alpaca CLI.

    Every method here takes something already decided. There is no path from a
    market observation to an order that does not pass through the gates first,
    because this class cannot see the market.
    """

    def __init__(self, *, dry_run: bool = True, runner=_run_command,
                 timeout: int = DEFAULT_TIMEOUT, binary: str = BINARY) -> None:
        # Dry run defaults to TRUE. Placing real orders has to be something a
        # caller asks for explicitly; it must never be what happens because a
        # flag went missing.
        self._dry_run = dry_run
        self._run = runner
        self._timeout = timeout
        self._binary = binary

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run

    # -- the plumbing ------------------------------------------------------

    def _call(self, *args: str) -> dict | list:
        """Run one CLI command and return its parsed JSON output."""
        command = [self._binary, *args, "--quiet", "--timeout", str(self._timeout)]

        try:
            result = self._run(command, self._timeout + 15)
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(f"command timed out after {self._timeout}s") from exc
        except FileNotFoundError as exc:
            raise ExecutionError(
                f"the '{self._binary}' CLI is not on PATH -- install it from "
                f"github.com/alpacahq/cli") from exc

        if result.returncode != 0:
            # stderr carries the broker's rejection message, which is the single
            # most useful thing to have in the journal when an order fails.
            detail = (result.stderr or result.stdout or "").strip()
            raise ExecutionError(f"CLI exited {result.returncode}: {detail[:400]}")

        import json
        try:
            return json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise ExecutionError(
                f"could not parse CLI output: {(result.stdout or '')[:200]}") from exc

    # -- safety ------------------------------------------------------------

    def account_number(self) -> str:
        payload = self._call("account", "get")
        if not isinstance(payload, dict):
            raise ExecutionError("unexpected account response")
        return str(payload.get("account_number", ""))

    def verify_paper(self) -> str:
        """Confirm with the broker that these credentials are a paper account.

        Called before every submit rather than once at construction. The cost is
        one fast request; the thing it protects against is an environment that
        changed underneath a long-running process, which is exactly the failure
        that would otherwise be discovered by placing a real trade.
        """
        number = self.account_number()
        if not number.upper().startswith(PAPER_PREFIX):
            raise NotAPaperAccount(
                f"account {number} is not a paper account. Refusing to trade.")
        return number

    # -- orders ------------------------------------------------------------

    def buy_to_open(self, draft: OrderDraft) -> OrderReceipt:
        """Open the position described by an approved draft.

        Takes an OrderDraft rather than loose arguments on purpose. A draft can
        only be produced by the entry bridge and can only be marked approved by
        the gate chain, so the type itself carries the evidence that this trade
        was permitted.
        """
        self.verify_paper()

        client_order_id = new_client_order_id(draft.contract.occ_symbol)
        args = [
            "order", "submit",
            "--symbol", draft.contract.occ_symbol,
            "--qty", str(draft.quantity),
            "--side", "buy",
            "--type", "limit",
            "--limit-price", f"{draft.limit_price:.2f}",
            "--time-in-force", "day",
            # Says explicitly that this opens a new long position rather than
            # closing a short one. Without it the broker infers intent from the
            # current holdings, and inference is not what you want on an account
            # that may hold other things.
            "--position-intent", "buy_to_open",
            "--client-order-id", client_order_id,
        ]
        if self._dry_run:
            args.append("--dry-run")

        payload = self._call(*args)
        return self._receipt(payload, client_order_id, draft.contract.occ_symbol,
                             draft.quantity, draft.limit_price)

    def sell_to_close(self, position: OpenPosition, limit_price: float) -> OrderReceipt:
        """Close a position at a limit.

        Options carry no broker-side trailing stop -- Alpaca supports those for
        stocks only -- so every exit in this system is an order this method
        places, decided by code that re-checks the position on each pass.
        """
        self.verify_paper()

        client_order_id = new_client_order_id(position.occ_symbol)
        args = [
            "order", "submit",
            "--symbol", position.occ_symbol,
            "--qty", str(position.quantity),
            "--side", "sell",
            "--type", "limit",
            "--limit-price", f"{limit_price:.2f}",
            "--time-in-force", "day",
            "--position-intent", "sell_to_close",
            "--client-order-id", client_order_id,
        ]
        if self._dry_run:
            args.append("--dry-run")

        payload = self._call(*args)
        return self._receipt(payload, client_order_id, position.occ_symbol,
                             position.quantity, limit_price)

    def close_at_market(self, position: OpenPosition) -> OrderReceipt:
        """Close a position immediately, accepting whatever the market gives.

        The escalation path, and it exists because of a measured failure in the
        earlier system: exits placed as patient limits repeatedly timed out
        while the position kept falling, and stops meant for -25% filled between
        -27% and -33%. A missed entry is harmless because the opportunity
        returns; a missed exit is not, because the loss keeps growing while the
        order sits unfilled.
        """
        self.verify_paper()

        client_order_id = new_client_order_id(position.occ_symbol)
        args = [
            "order", "submit",
            "--symbol", position.occ_symbol,
            "--qty", str(position.quantity),
            "--side", "sell",
            "--type", "market",
            "--time-in-force", "day",
            "--position-intent", "sell_to_close",
            "--client-order-id", client_order_id,
        ]
        if self._dry_run:
            args.append("--dry-run")

        payload = self._call(*args)
        return self._receipt(payload, client_order_id, position.occ_symbol,
                             position.quantity, None)

    def cancel(self, order_id: str) -> bool:
        """Cancel an open order. Returns False rather than raising if it is gone.

        An order that cannot be cancelled because it already filled or already
        expired is not an error -- it is the normal race between a scheduled
        pass and a moving market.
        """
        try:
            self._call("order", "cancel", "--order-id", order_id)
            return True
        except ExecutionError as exc:
            logger.info("could not cancel %s: %s", order_id, exc)
            return False

    def open_orders(self) -> list[dict]:
        """Orders that have not yet filled or been cancelled."""
        payload = self._call("order", "list", "--status", "open")
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            rows = payload.get("orders") or payload.get("result") or []
            return rows if isinstance(rows, list) else []
        return []

    # -- shared ------------------------------------------------------------

    def _receipt(self, payload, client_order_id: str, symbol: str,
                 quantity: int, limit_price: float | None) -> OrderReceipt:
        """Build a receipt from whatever the CLI returned.

        A dry run echoes the request body rather than a created order, so there
        is no order id to record. Both shapes produce a receipt, because the
        caller's job -- write this to the journal, announce it -- is the same
        either way.
        """
        body = payload if isinstance(payload, dict) else {}
        return OrderReceipt(
            client_order_id=str(body.get("client_order_id") or client_order_id),
            symbol=symbol,
            quantity=quantity,
            limit_price=limit_price,
            order_id=str(body.get("id") or ""),
            status=str(body.get("status") or ("validated" if self._dry_run else "")),
            dry_run=self._dry_run,
            raw=body,
        )
