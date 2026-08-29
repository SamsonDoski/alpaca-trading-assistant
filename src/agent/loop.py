"""One pass of the agent: manage what is open, then look for what is not.

This module owns the *order* things happen in, and almost nothing else. Every
decision it reaches for lives somewhere else -- exits in `exits.py`, screening
and sizing in `gates.py` and `entry.py`, judgement in `proposer.py`, orders in
`executor.py`. What is left here is sequencing, and sequencing turns out to be
where the important safety properties live.

**Exits run first, unconditionally.** Before a single symbol is screened, before
any model call, every open position is checked and closed if it needs closing.
The reason is not efficiency. If entries ran first, a pass that failed partway
through -- a rate limit, a timeout, a crash -- would have spent its budget opening
something new while leaving a stopped-out position sitting untouched until the
next pass. Doing exits first means a partial failure fails in the safe direction.

**Reads and reasoning fan out; decisions and orders do not.** Gathering nine
briefs and asking for nine opinions is nine independent waits, so those run
concurrently. Gate checks and order submission run strictly one at a time, in a
fixed order, because they share state: two threads both seeing "four of five
slots used" would both open a position. Concurrency where it is free, sequence
where correctness lives.

**Everything is written down before it is announced.** The journal is the record
of what happened; Discord is a courtesy. A failed webhook must never cost an
audit row, so the write always comes first.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date

from agent.domain import MarketBrief, OpenPosition, Proposal
from agent.entry import decide_entry
from agent.executor import CliExecutor, ExecutionError
from agent.exits import ExitDecision, check_exit, exit_limit_price
from agent.gates import GateContext, screen
from agent.journal import Journal
from agent.market import MarketDataError, build_brief
from agent.notify import Notifier
from agent.proposer import Proposer
from agent.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class PassResult:
    """What one pass did, for the caller to print and the summary to announce."""

    considered: int = 0
    opened: int = 0
    closed: int = 0
    skipped_before_model: int = 0
    refusals: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    equity: float = 0.0
    unrealized: float = 0.0

    def note(self, symbol: str, reason: str) -> None:
        self.refusals.append((symbol, reason))


async def run_pass(
    reader,
    *,
    settings: Settings,
    executor: CliExecutor,
    proposer: Proposer,
    journal: Journal,
    notifier: Notifier,
    today: date | None = None,
    ignore_clock: bool = False,
    trading_halted: bool = False,
) -> PassResult:
    """Run one complete pass and return what it did.

    Every collaborator is passed in rather than constructed here. That is what
    makes the whole pass testable: a test supplies a fake reader, a fake
    executor and a proposer wired to a canned answer, and can then assert on
    the sequence of things that happened without a network, an account, or a
    single token spent.
    """
    today = today or date.today()
    result = PassResult()

    # --- 1. See the world -------------------------------------------------
    # These three must succeed. Guessing at any of them means trading against
    # an account we cannot actually see.
    try:
        account, positions, is_open = await asyncio.gather(
            reader.account(), reader.positions(), reader.market_open())
    except MarketDataError as exc:
        message = f"ABORT: cannot read the account ({exc}). No trades this pass."
        logger.error(message)
        journal.record("alert", "", "", message)
        notifier.alert(message)
        result.errors.append(message)
        return result

    result.equity = account.equity
    result.unrealized = sum(p.unrealized_pnl for p in positions)
    market_open = is_open or ignore_clock

    if not market_open:
        logger.info("market is closed; nothing to do")
        return result

    # --- 2. Exits, before anything else ----------------------------------
    for position in positions:
        decision = check_exit(position, settings, today)
        if decision is None:
            continue
        try:
            await _close(decision, reader, executor, journal, notifier, settings)
            result.closed += 1
        except ExecutionError as exc:
            message = f"failed to close {position.occ_symbol}: {exc}"
            logger.error(message)
            journal.record("alert", position.underlying, position.occ_symbol, message)
            notifier.alert(message)
            result.errors.append(message)

    # Re-read the book. Positions just closed have freed both a slot and the
    # cash behind it, and the entry gates below must judge against the account
    # as it is now rather than as it was at the top of the pass.
    if result.closed:
        try:
            account, positions = await asyncio.gather(reader.account(), reader.positions())
        except MarketDataError as exc:
            logger.warning("could not re-read after closing: %s", exc)

    if trading_halted:
        # Announced on every pass it is on. A silent freeze looks exactly like a
        # quiet market, and that is how a halt gets left on for a week.
        message = ("KILL SWITCH ON -- no new positions will be opened. "
                   "Exits and stop checks are still running normally.")
        logger.warning(message)
        notifier.alert(message)

    ctx = GateContext(
        today=today,
        market_open=market_open,
        trading_halted=trading_halted,
        account=account,
        open_positions=tuple(positions),
        cooling_off=journal.cooling_off(within_days=settings.cooldown_days, as_of=today),
        settings=settings,
    )

    # --- 3. Screen, free, before spending anything -----------------------
    candidates = []
    for symbol in settings.symbols:
        outcome = screen(symbol, ctx)
        if outcome.approved:
            candidates.append(symbol)
        else:
            result.skipped_before_model += 1
            journal.record_decision(symbol, approved=False, reason=outcome.reason,
                                    trace=outcome.trace)

    if not candidates:
        await _finish(result, ctx, positions, journal, notifier, executor)
        return result

    # --- 4. Gather and reason, concurrently ------------------------------
    briefs = await asyncio.gather(
        *(build_brief(reader, symbol, settings, today=today) for symbol in candidates),
        return_exceptions=True,
    )

    usable: list[MarketBrief] = []
    for symbol, brief in zip(candidates, briefs, strict=True):
        if isinstance(brief, BaseException):
            logger.warning("brief failed for %s: %s", symbol, brief)
            journal.record_decision(symbol, approved=False,
                                    reason=f"market data unavailable ({brief})")
            result.note(symbol, "market data unavailable")
        else:
            usable.append(brief)

    # The model calls are independent of one another, so they wait together.
    # `to_thread` because the Anthropic client is synchronous; this keeps the
    # event loop free rather than making nine calls in series.
    proposals: list[Proposal] = await asyncio.gather(
        *(asyncio.to_thread(proposer.propose, brief) for brief in usable))
    result.considered = len(proposals)

    # --- 5. Decide and order, strictly one at a time ---------------------
    # Highest conviction first, so that when slots or cash run out it is the
    # weakest ideas that miss out rather than whichever happened to be later in
    # the watchlist.
    ranked = sorted(zip(usable, proposals, strict=True),
                    key=lambda pair: pair[1].confidence, reverse=True)

    for brief, proposal in ranked:
        # Screen again, against the context as it now stands. The first screen
        # ran before any of this pass's orders existed, and opening a position
        # consumes a slot and the cash behind it -- so a candidate that was
        # eligible at the top of the pass may not be eligible by the time its
        # turn arrives. Without this, a pass with nine candidates and five free
        # slots would open nine positions. The gates are free, so re-running
        # them costs nothing but the correctness is not optional.
        rescreen = screen(brief.underlying, ctx)
        if not rescreen.approved:
            journal.record_decision(brief.underlying, approved=False,
                                    reason=rescreen.reason, proposal=proposal,
                                    trace=rescreen.trace)
            result.note(brief.underlying, rescreen.reason)
            continue

        outcome = decide_entry(proposal, brief, ctx)
        journal.record_decision(brief.underlying, approved=outcome.approved,
                                reason=outcome.reason, proposal=proposal,
                                trace=outcome.trace)

        if not outcome.approved:
            result.note(brief.underlying, outcome.reason)
            continue

        try:
            receipt = executor.buy_to_open(outcome.draft)
        except ExecutionError as exc:
            message = f"failed to open {brief.underlying}: {exc}"
            logger.error(message)
            journal.record("alert", brief.underlying, outcome.draft.contract.occ_symbol,
                           message)
            notifier.alert(message)
            result.errors.append(message)
            continue

        detail = (f"{outcome.draft} — {proposal.confidence:.0%} confidence. "
                  f"{proposal.rationale}")
        journal.record("opened", brief.underlying, outcome.draft.contract.occ_symbol,
                       detail)
        notifier.opened(outcome.draft.contract.occ_symbol, detail,
                        reasoning=proposal.thinking_summary)
        result.opened += 1

        # The account has changed. Rebuilding the context rather than mutating
        # it keeps GateContext frozen, and means the next candidate is judged
        # against the position we just took.
        ctx = _with_new_position(ctx, outcome.draft)

    await _finish(result, ctx, ctx.open_positions, journal, notifier, executor)
    return result


async def _close(decision: ExitDecision, reader, executor: CliExecutor,
                 journal: Journal, notifier: Notifier, settings: Settings) -> None:
    """Close one position, patiently or immediately depending on why."""
    position = decision.position
    quote = await reader.option_quote(position.occ_symbol)

    if decision.urgent or quote is None:
        # Either the position is running away from us, or nobody is quoting it
        # and there is no limit price worth naming. Both mean: take what the
        # market gives rather than sit on an unfillable order.
        receipt = executor.close_at_market(position)
        how = "at market"
    else:
        bid, ask = quote
        price = exit_limit_price(bid, ask, settings.exit_aggression)
        receipt = executor.sell_to_close(position, price)
        how = f"limit {price:.2f}"

    pnl = position.unrealized_pnl
    detail = (f"{_detail_prefix(decision)} — {decision.detail}, {how}. "
              f"P&L ${pnl:+,.0f} ({position.return_pct:+.1%}) [{receipt.client_order_id}]")

    journal.record("closed", position.underlying, position.occ_symbol, detail, pnl=pnl)
    notifier.closed(position.occ_symbol, detail, won=pnl > 0)


def _detail_prefix(decision: ExitDecision) -> str:
    """The first words of a close, which the cooldown rule reads back.

    A stop-loss close must begin with exactly the string `journal.cooling_off`
    matches on, because that is what distinguishes "this thesis failed" from
    "this thesis worked and we banked it".
    """
    from agent.exits import ExitReason
    return "stop loss" if decision.reason is ExitReason.STOP_LOSS else decision.reason.value


def _with_new_position(ctx: GateContext, draft) -> GateContext:
    """A context that knows about a position we just opened.

    Frozen dataclasses are replaced, never mutated. The cost is one small
    object; the benefit is that no earlier reference to the context can be
    changed underneath its holder.
    """
    from dataclasses import replace

    added = OpenPosition(
        occ_symbol=draft.contract.occ_symbol,
        underlying=draft.contract.underlying,
        quantity=draft.quantity,
        entry_price=draft.limit_price,
        current_price=draft.limit_price,
        expiry=draft.contract.expiry,
    )
    return replace(ctx, open_positions=ctx.open_positions + (added,))


async def _finish(result: PassResult, ctx: GateContext,
                  positions: tuple[OpenPosition, ...], journal: Journal,
                  notifier: Notifier, executor: CliExecutor) -> None:
    """Announce what the pass did, refusals included."""
    book = [(p.occ_symbol, p.unrealized_pnl) for p in positions]
    notifier.pass_summary(
        equity=result.equity,
        available=ctx.account.available - ctx.committed,
        open_count=len(positions),
        opened=result.opened,
        closed=result.closed,
        considered=result.considered,
        realized=journal.realized_for_day(),
        unrealized=sum(p.unrealized_pnl for p in positions),
        book=book,
        refusals=result.refusals,
        dry_run=executor.is_dry_run,
    )
