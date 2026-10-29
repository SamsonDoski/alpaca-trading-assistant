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
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

from agent.domain import MarketBrief, OpenPosition, Proposal
from agent.entry import decide_entry
from agent.executor import CliExecutor, ExecutionError
from agent.exits import (
    ExitDecision,
    ExitReason,
    check_exit,
    exit_limit_price,
    stop_and_target,
)
from agent.gates import GateContext, screen
from agent.journal import Journal
from agent.market import MarketDataError, build_brief
from agent.notify import Notifier
from agent.pricing import average_true_range_pct, daily_decay_pct, premium_richness
from agent.proposer import Proposer
from agent.selector import Candidate, select
from agent.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class PassResult:
    """What one pass did, for the caller to print and the summary to announce."""

    considered: int = 0
    ordered: int = 0      # sent to the broker this pass
    opened: int = 0       # confirmed filled this pass, whenever they were ordered
    closed: int = 0
    skipped_before_model: int = 0
    refusals: list[tuple[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    equity: float = 0.0
    unrealized: float = 0.0

    def note(self, symbol: str, reason: str) -> None:
        self.refusals.append((symbol, reason))


_NEW_YORK = ZoneInfo("America/New_York")
_SESSION_OPEN = time(9, 30)
_SESSION_CLOSE = time(16, 0)


def _inside_regular_session(now: datetime) -> bool:
    """Whether `now` falls inside regular US options trading hours.

    Only a fallback for when the broker's clock cannot be read, and deliberately
    blunt: weekdays, 9:30 to 16:00 New York time. It knows nothing about
    holidays or early closes, which is why the pass that uses it runs exits and
    nothing else. On a holiday the worst it can do is submit a close the broker
    refuses; against the alternative -- a real session with no stop checks, which
    is what actually happened -- that is the cheap mistake.
    """
    local = now.astimezone(_NEW_YORK)
    if local.weekday() >= 5:
        return False
    return _SESSION_OPEN <= local.time() < _SESSION_CLOSE


def _reconcile_holdings(positions: tuple[OpenPosition, ...] | list[OpenPosition],
                        journal: Journal, notifier: Notifier,
                        executor: CliExecutor) -> int:
    """Bring the agent's picture of the book back in line with the broker's.

    The `holdings` table is written when an order is *submitted*, because that
    is the only moment the underlying's price at the decision is still known --
    and a stop keyed to the underlying is impossible without it. The cost of
    writing it then is that a row exists before the position does, and on this
    feed a limit priced off an indicative quote can rest unfilled until it
    expires. Nothing outside this module ever removed those rows.

    So a row with no matching position means one of two quite different things,
    and treating them alike is worse than not reconciling at all:

      **The order never filled.** There was no position, nothing was decided,
      and nothing happened worth reporting. Drop the row and move on.

      **It filled and then left the book** -- closed by hand, assigned, expired,
      or liquidated. That row is actively dangerous, because it carries
      `stop_spot` and `target_spot` for a trade that is over, and a re-entry
      into the same contract would inherit them. It also means no `closed` event
      was written, so no cooldown started, which is how a name stopped out by
      hand gets re-bought on the next pass. Observed live: an F put closed by
      hand on 4 Sep 2026 left exactly that behind.

    The broker's fill history is what separates them. A resting buy order is
    left alone entirely -- it may still fill, and clearing its row would strand
    the position that arrives without the levels its stop needs.

    Returns how many rows were reconciled, so a caller can tell "nothing to do"
    from "did nothing".
    """
    stale = [h for h in journal.open_holdings()
             if h.occ_symbol not in {p.occ_symbol for p in positions}]
    if not stale:
        return 0

    # If either read fails, do nothing. Reconciling on a partial view would
    # either invent a cooldown or drop levels a live position still needs, and
    # both are worse than leaving the table alone for one pass.
    try:
        pending = {str(o.get("symbol", "")) for o in executor.open_orders()
                   if str(o.get("side", "")).lower() == "buy"}
        filled = executor.bought_contracts()
    except ExecutionError as exc:
        logger.warning("could not reconcile holdings against the broker: %s", exc)
        return 0

    reconciled = 0
    for holding in stale:
        if holding.occ_symbol in pending:
            continue

        if holding.occ_symbol not in filled:
            # Written down, but not announced: nothing happened to the account.
            # It is still worth a row, because an order that never fills is a
            # cost of the feed worth being able to count.
            journal.record("unfilled", holding.underlying, holding.occ_symbol,
                           "order expired or was cancelled without filling")
            journal.close_holding(holding.occ_symbol)
            logger.info("%s never filled; dropping its recorded levels",
                        holding.occ_symbol)
            reconciled += 1
            continue

        journal.record_external_close(
            holding, note="the broker no longer reports this position")
        message = (f"{holding.occ_symbol} left the book without the agent "
                   f"closing it -- clearing its recorded levels and cooling "
                   f"off {holding.underlying}")
        logger.warning(message)
        notifier.alert(message)
        reconciled += 1

    return reconciled


def _record_entry_metrics(journal: Journal, draft, brief: MarketBrief, proposal,
                          settings: Settings, *, today: date,
                          stop_spot: float, target_spot: float,
                          atr_pct: float | None) -> None:
    """Write down every number that was weighed, at the moment it was weighed.

    None of this is read back today. It exists because the one question this
    system could answer about itself -- does the model's confidence predict
    anything? -- was answerable only because confidence happened to be stored,
    and the answer was worth about $10,000. The same question about spread,
    implied volatility, delta or time to expiry could not be asked, because
    every one of those numbers was computed, used by a gate, and discarded.

    Deliberately recorded at the ORDER, not the fill. A limit that never fills
    still reflects a judgement that was made, and whether unfilled orders differ
    from filled ones is itself a question worth being able to ask.

    Never raises. This is measurement, and measurement must not be able to stop
    a trade that the gates have already approved.
    """
    contract = draft.contract
    try:
        journal.record_entry_metrics(
            occ_symbol=contract.occ_symbol,
            underlying=brief.underlying,
            direction=proposal.direction.value,
            confidence=proposal.confidence,
            setup_quality=proposal.setup_quality,
            quantity=draft.quantity,
            limit_price=draft.limit_price,
            cost=draft.total_cost,
            spot=brief.spot,
            strike=contract.strike,
            days_to_expiry=contract.days_to_expiry(today),
            delta=contract.abs_delta,
            implied_vol=contract.implied_volatility,
            realized_vol=draft.realized_vol,
            iv_to_realized=premium_richness(contract.implied_volatility,
                                            draft.realized_vol),
            spread_pct=contract.spread_pct,
            daily_decay=daily_decay_pct(contract, draft.spot, today),
            open_interest=contract.open_interest,
            atr_pct=atr_pct,
            range_position=brief.range_position,
            change_5d=brief.change_pct(5),
            change_20d=brief.change_pct(20),
            change_60d=brief.change_pct(60),
            stop_spot=stop_spot,
            target_spot=target_spot,
        )
    except Exception as exc:           # noqa: BLE001 -- see the docstring
        logger.warning("could not record entry metrics for %s: %s",
                       contract.occ_symbol, exc)


def _book_completed_sales(positions: tuple[OpenPosition, ...] | list[OpenPosition],
                          journal: Journal) -> int:
    """Book every sale the broker now agrees actually happened.

    `_close` sends the order and records it as `sold`. This is where the trade
    becomes `closed`: once, when the position is genuinely gone, however many
    orders it took to get there.

    The separation exists because conflating the two double-counted a trade. On
    22 Oct a GLD put hit its premium target at 19:40 and went out as a patient
    limit; the 19:45 pass found it still held with that order resting, escalated
    to market, and wrote a second `closed` row. One +$3,020 trade was reported
    as +$6,040 of realised P&L.

    Booking from the sale rather than the holdings row matters for positions the
    agent sold but never opened -- an older system's, or one whose holdings row
    was already cleared. Those have no row to reconcile, so keying off the row
    alone would leave the trade unbooked for ever.

    Returns how many trades were booked.
    """
    held = {p.occ_symbol for p in positions}
    booked = 0

    for sale in journal.unbooked_sales():
        if sale["symbol"] in held:
            continue                      # the order has not filled yet
        journal.record("closed", sale["underlying"], sale["symbol"],
                       sale["detail"], pnl=sale["pnl"])
        journal.close_holding(sale["symbol"])
        logger.info("%s confirmed closed by the broker", sale["symbol"])
        booked += 1

    return booked


def _announce_fills(positions: tuple[OpenPosition, ...] | list[OpenPosition],
                    journal: Journal, notifier: Notifier) -> int:
    """Record and announce each ordered position the broker now shows as held.

    Uses the broker's own average entry price and quantity, so the announcement
    says what was actually paid rather than the limit that was asked for -- on
    a marketable limit those differ, and the difference is the real cost of the
    entry.

    Returns how many fills were announced.
    """
    by_symbol = {p.occ_symbol: p for p in positions}
    fills = journal.unannounced_fills(set(by_symbol))

    for holding in fills:
        held = by_symbol[holding.occ_symbol]
        cost = held.entry_price * held.quantity * 100
        ordered, reasoning = journal.order_story(holding)
        detail = (f"filled {held.quantity} at {held.entry_price:.2f} "
                  f"(${cost:,.0f})")
        if ordered:
            detail += f" — ordered as {ordered}"
        journal.record("opened", holding.underlying, holding.occ_symbol, detail)
        notifier.opened(holding.occ_symbol, detail, reasoning=reasoning)

    return len(fills)


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
    exits_only: bool = False,
    selector_backend=None,
) -> PassResult:
    """Run one complete pass and return what it did.

    Every collaborator is passed in rather than constructed here. That is what
    makes the whole pass testable: a test supplies a fake reader, a fake
    executor and a proposer wired to a canned answer, and can then assert on
    the sequence of things that happened without a network, an account, or a
    single token spent.

    **`exits_only` runs the managing half and stops.** Entries are the expensive
    part -- a model call per candidate -- while exits are a handful of broker
    reads and some arithmetic. That asymmetry means the stop can be checked far
    more often than new positions are looked for, and it should be: the gap
    between passes is how far a position travels unwatched, and measured over
    81 round trips, losses that overshot the stop while the market was open cost
    $3,847 beyond where the rule said to exit.
    """
    today = today or date.today()
    result = PassResult()

    # --- 1. See the world -------------------------------------------------
    # These two must succeed. Guessing at either means trading against an
    # account we cannot actually see.
    try:
        account, positions = await asyncio.gather(reader.account(), reader.positions())
    except MarketDataError as exc:
        message = (f"ABORT: cannot read the account or positions ({exc}). "
                   f"No trades this pass.")
        logger.error(message)
        journal.record("alert", "", "", message)
        notifier.alert(message)
        result.errors.append(message)
        return result

    # The clock is read on its own, because it only decides whether to OPEN
    # anything. It used to be bundled into the read above, so a failure on it
    # aborted the whole pass -- stop checks included. On 11 Sep 2026 the broker's
    # clock endpoint returned HTTP 500 for three passes in a row and every open
    # position went 45 minutes unwatched, over a clock the exits never needed.
    #
    # Without it, regular session hours decide instead, and entries are skipped:
    # closing a position that has hit its stop is always right, while opening one
    # against a session we could not confirm is not.
    clock_known = True
    try:
        is_open = await reader.market_open()
    except MarketDataError as exc:
        clock_known = False
        is_open = _inside_regular_session(datetime.now(UTC))
        message = (f"broker clock unavailable ({exc}). Judging the session from "
                   f"regular hours instead ({'open' if is_open else 'closed'}): "
                   f"exits run, entries are skipped this pass.")
        logger.warning(message)
        journal.record("alert", "", "", message)
        notifier.alert(message)

    result.equity = account.equity
    result.unrealized = sum(p.unrealized_pnl for p in positions)
    market_open = is_open or ignore_clock

    if not market_open:
        logger.info("market is closed; nothing to do")
        return result

    # --- 2. Reconcile our picture of the book against the broker's --------
    # Before any exit is evaluated, because `check_exit` reads the holdings
    # table and a stale row there is worse than a missing one: it supplies stop
    # and target levels computed for a trade that is already over.
    # Sales that have filled since the pass that sent them. Booked before
    # reconciliation, so a position the agent sold is recognised as its own
    # trade rather than mistaken for one someone closed by hand.
    result.closed += _book_completed_sales(positions, journal)

    _reconcile_holdings(positions, journal, notifier, executor)

    # Orders from earlier passes that have filled since. Announced here rather
    # than when they were sent, so "opened" means the contract is in the book.
    result.opened += _announce_fills(positions, journal, notifier)

    # Exit orders already resting at the broker, by contract. Read before the
    # exit loop because it decides between placing a fresh order and escalating
    # a stale one -- and because without it the same position is sold once per
    # pass until something fills.
    try:
        resting_exits = await _resting_exit_orders(executor)
    except ExecutionError as exc:
        logger.warning("could not read resting exit orders: %s", exc)
        resting_exits = {}

    # --- 3. Exits, before anything else ----------------------------------
    for position in positions:
        # The stop is keyed to the underlying, so the underlying's price is
        # needed to evaluate it. Held names never reach the brief stage, so this
        # is the exit path's only source for it. A failed read falls back to the
        # premium rule rather than skipping the check -- an unmanaged position is
        # worse than a bluntly managed one.
        spot = await reader.stock_price(position.underlying)

        # The high-water mark moves up here and nowhere else. It has to happen
        # on every pass, before the check and whether or not anything closes,
        # because a peak that is not written while the position is at it is
        # gone -- the next pass reads a price, not a history.
        holding = journal.holding(position.occ_symbol)
        if holding is not None:
            holding = journal.raise_peak(holding, position.current_price)

        decision = check_exit(position, settings, today,
                              holding=holding, spot=spot)
        if decision is None:
            continue
        try:
            await _close(decision, reader, executor, journal, notifier, settings,
                         stale_order=resting_exits.get(position.occ_symbol),
                         spot=spot)
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

    if exits_only:
        # Everything that manages open risk has now run. Stop before the entry
        # machinery: no gate context, no screening, no model calls -- and no
        # decision rows, which is what keeps these extra passes from counting as
        # extra trading sessions when a cooldown measures them.
        logger.info("exits-only pass: %d closed", result.closed)
        return result

    if trading_halted:
        # Announced on every pass it is on. A silent freeze looks exactly like a
        # quiet market, and that is how a halt gets left on for a week.
        message = ("KILL SWITCH ON -- no new positions will be opened. "
                   "Exits and stop checks are still running normally.")
        logger.warning(message)
        notifier.alert(message)

    # Orders resting at the broker that have not filled. On a data feed where a
    # limit can sit unexecuted for hours, this is not an edge case -- it is the
    # normal state between placing an order and owning a position, and the gates
    # are blind to it unless it is handed to them.
    #
    # A failed read stops entries rather than assuming none are resting. Missing
    # a pass costs one opportunity; placing a second order on a name that
    # already has one costs real money if the price later moves through both.
    try:
        pending = _pending_underlyings(executor)
    except ExecutionError as exc:
        message = (f"could not read resting orders ({exc}). Exits ran; entries are "
                   f"skipped this pass rather than risk duplicating an open order.")
        logger.error(message)
        journal.record("alert", "", "", message)
        notifier.alert(message)
        result.errors.append(message)
        return result

    ctx = GateContext(
        today=today,
        market_open=market_open,
        trading_halted=trading_halted,
        account=account,
        open_positions=tuple(positions),
        cooling_off=journal.cooling_off(within_days=settings.cooldown_days, as_of=today),
        pending=pending,
        settings=settings,
    )

    # Exits have run; nothing is opened against a session the broker did not
    # confirm. Stopping before the screen matters for more than tokens: screening
    # writes decision rows, and decision rows are how cooldowns count trading
    # sessions -- a holiday mistaken for a session would quietly shorten one.
    if not clock_known:
        await _finish(result, ctx, positions, journal, notifier, executor)
        return result

    # --- 4. Screen, free, before spending anything -----------------------
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

    # --- 5. Gather and reason, concurrently but bounded ------------------
    # The bound is the important word. Unlimited fan-out is what killed the
    # agent once the watchlist grew: every symbol's four reads went out at
    # once, the MCP server closed the connection under the load, and the pass
    # died before it could do anything -- including check its own stops.
    gate = asyncio.Semaphore(max(1, settings.max_concurrent_symbols))

    async def gather_one(symbol: str):
        async with gate:
            return await build_brief(reader, symbol, settings, today=today)

    briefs = await asyncio.gather(
        *(gather_one(symbol) for symbol in candidates),
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

    # The model calls are independent of one another, so they wait together --
    # under the same ceiling, because a reasoning model thinking for thirty
    # symbols at once is its own way to hit a rate limit.
    async def ask_one(brief: MarketBrief) -> Proposal:
        async with gate:
            return await asyncio.to_thread(proposer.propose, brief)

    proposals: list[Proposal] = await asyncio.gather(
        *(ask_one(brief) for brief in usable))
    result.considered = len(proposals)

    # --- 6. Decide and order, strictly one at a time ---------------------
    # Highest conviction first, so that when slots or cash run out it is the
    # weakest ideas that miss out rather than whichever happened to be later in
    # the watchlist.
    ranked = sorted(zip(usable, proposals, strict=True),
                    key=lambda pair: pair[1].confidence, reverse=True)

    # --- 6a. Authorise every candidate individually ----------------------
    # Each is judged as if it were the only trade, which deliberately
    # over-approves: slots, cash and concentration are cumulative, and cannot be
    # applied until an order is chosen. That is what the second pass is for.
    slate: list[Candidate] = []
    for brief, proposal in ranked:
        outcome = decide_entry(proposal, brief, ctx)
        if not outcome.approved:
            journal.record_decision(brief.underlying, approved=False,
                                    reason=outcome.reason, proposal=proposal,
                                    trace=outcome.trace)
            result.note(brief.underlying, outcome.reason)
            continue
        slate.append(Candidate(
            draft=outcome.draft,
            group=settings.correlation_groups.get(brief.underlying, brief.underlying),
        ))

    # --- 6b. Order the slate ---------------------------------------------
    free_slots = max(0, settings.max_positions - ctx.committed_slots)
    slate, selection_note = select(slate, backend=selector_backend,
                                   settings=settings, slots=free_slots, today=today)
    if slate:
        logger.info("selection: %s", selection_note)
        journal.record("selection", "", "", selection_note)

    briefs_by_symbol = {b.underlying: b for b in usable}

    # --- 6c. Walk the order and open, re-authorising as the book fills ---
    for candidate in slate:
        brief = briefs_by_symbol[candidate.underlying]
        proposal = candidate.draft.proposal

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

        # The binding authorisation. The first pass judged this candidate alone;
        # this one judges it against the book including anything opened earlier
        # in this very walk, so slots, cash and concentration actually apply.
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

        # Write down where the underlying stood, and the levels that will decide
        # this position's fate. Without this the stop has nothing to measure
        # against and silently falls back to the premium rule it replaced.
        atr_pct = average_true_range_pct(brief.bars)
        stop_spot, target_spot = stop_and_target(
            brief.spot, proposal.direction.value,
            atr_pct=atr_pct,
            settings=settings)
        journal.open_holding(
            occ_symbol=outcome.draft.contract.occ_symbol,
            underlying=brief.underlying,
            direction=proposal.direction.value,
            entry_spot=brief.spot,
            entry_premium=outcome.draft.limit_price,
            stop_spot=stop_spot,
            target_spot=target_spot,
        )

        # Recorded as ORDERED, not opened. A limit priced off an indicative quote
        # can rest unfilled until it expires, and announcing it as a position the
        # moment it is sent is how the log came to report trades that never
        # happened. `_announce_fills` says "opened" once the broker shows it.
        detail = (f"{outcome.draft} — {proposal.confidence:.0%} confidence, "
                  f"stop {stop_spot:,.2f} / target {target_spot:,.2f} on "
                  f"{brief.underlying} at {brief.spot:,.2f}. {proposal.rationale}")
        journal.record("ordered", brief.underlying, outcome.draft.contract.occ_symbol,
                       detail)
        _record_entry_metrics(journal, outcome.draft, brief, proposal, settings,
                              today=today, stop_spot=stop_spot,
                              target_spot=target_spot, atr_pct=atr_pct)
        result.ordered += 1

        # The account has changed. Rebuilding the context rather than mutating
        # it keeps GateContext frozen, and means the next candidate is judged
        # against the position we just took.
        ctx = _with_new_position(ctx, outcome.draft)

    # The walk above treats every order as a position so the gates judge the next
    # candidate against it. The summary must not: `ctx.open_positions` includes
    # orders that may never fill, which is how an unfilled limit came to appear in
    # the open book at +$0. Read the real book once more, and announce anything
    # that filled on the spot -- most marketable limits do.
    book = positions
    if result.ordered:
        try:
            book = await reader.positions()
            result.opened += _announce_fills(book, journal, notifier)
        except MarketDataError as exc:
            logger.warning("could not re-read the book after ordering: %s", exc)

    await _finish(result, ctx, book, journal, notifier, executor)
    return result


async def _resting_exit_orders(executor: CliExecutor) -> dict[str, str]:
    """Contract symbol -> order id, for sell orders not yet filled.

    An exit that has not filled is the exit path's version of the resting-order
    problem the entry gates already handle. Without this the same position is
    sold once per pass -- eight orders for one contract over an afternoon.
    """
    resting = {}
    for order in executor.open_orders():
        if str(order.get("side", "")).lower() != "sell":
            continue
        symbol, order_id = str(order.get("symbol", "")), str(order.get("id", ""))
        if symbol and order_id:
            resting[symbol] = order_id
    return resting


def _protective_floor(position: OpenPosition, quote, spot: float | None) -> float | None:
    """A limit to use instead of a market order, when the quote cannot be real.

    An American option is worth at least its intrinsic value -- anyone could buy
    it, exercise, and bank the difference -- so a bid below that is a broken
    quote rather than a price. This feed produces them: on 16 Sep 2026 a PFE 29
    put was bid 0.72 with PFE at 27.40, worth 1.60 on exercise alone. The stop
    fired on that mark and a market order sold into the same bid, booking -48.9%
    on a position that was ahead.

    Returns a price only in that case. The two cents below intrinsic is slack
    for a real buyer to meet; against a broken bid it simply will not fill,
    which is the point. The position stays, the next pass looks again, and
    nothing was given away.

    Returns None whenever the quote is sane, or the contract is out of the money
    and has no intrinsic value to check against. Then the normal rules apply --
    a stop still goes out at market, because a stop that sits unfilled while the
    price falls is the failure this system has already paid for once.
    """
    intrinsic = position.intrinsic(spot)
    if intrinsic is None or intrinsic <= 0:
        return None

    bid = quote[0] if quote else None
    if bid is not None and bid >= intrinsic:
        return None

    return round(intrinsic - 0.02, 2)


async def _close(decision: ExitDecision, reader, executor: CliExecutor,
                 journal: Journal, notifier: Notifier, settings: Settings,
                 *, stale_order: str | None = None,
                 spot: float | None = None) -> None:
    """Close one position, patiently or immediately depending on why.

    **A resting exit order means last pass's limit did not fill**, and the
    correct response is to escalate rather than to wait again or to stack a
    second order on top. The limit is cancelled and replaced with one that is
    marketable now.

    That is the same lesson the stop path already learned the expensive way: a
    missed exit is not harmless. It applies to winners too -- an unfilled
    take-profit is a gain that has not been banked, and on this data feed a
    patient limit can sit unfilled for the rest of the session.

    What neither path does any more is name no price at all. See `_floor_price`.
    """
    position = decision.position
    quote = await reader.option_quote(position.occ_symbol)
    floor = _protective_floor(position, quote, spot)

    if floor is not None:
        # The quote is below what the contract is worth on exercise. Whatever
        # the reason for closing, refuse to hand it over at that price.
        if stale_order:
            executor.cancel(stale_order)
        receipt = executor.sell_to_close(position, floor)
        how = (f"limit {floor:.2f}, holding out for intrinsic value -- the "
               f"quote is below what exercise alone is worth")
    elif stale_order:
        logger.info("escalating %s: previous exit limit did not fill",
                    position.occ_symbol)
        executor.cancel(stale_order)
        receipt = executor.close_at_market(position)
        how = "escalated to market after the limit did not fill"
    elif decision.urgent or quote is None:
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

    # Recorded as SOLD, not closed, for the same reason an entry is recorded as
    # ordered rather than opened: sending a sell order and no longer holding the
    # position are different events, and on this feed a patient limit routinely
    # sits unfilled between them.
    #
    # Conflating them double-counted a trade. On 22 Oct a GLD put hit its
    # premium target at 19:40 and went out as a limit; the 19:45 pass found it
    # still held with the order resting, escalated to market, and wrote a second
    # `closed` row. The day's realised P&L read +$6,040 for one +$3,020 trade.
    #
    # The holdings row stays too. Deleting it here cost the escalating pass its
    # stop and target levels, which is why the second notification called a
    # premium target a take profit -- harmless on a winner, but the same path
    # handles stops.
    journal.record("sold", position.underlying, position.occ_symbol, detail, pnl=pnl)
    notifier.closed(position.occ_symbol, detail, won=pnl > 0)


def _detail_prefix(decision: ExitDecision) -> str:
    """The first words of a close, which the cooldown rule reads back.

    A stop-loss close must begin with exactly the string `journal.cooling_off`
    matches on, because that is what distinguishes "this thesis failed" from
    "this thesis worked and we banked it".

    A premium backstop deliberately does NOT get that prefix, and so starts no
    cooldown. It fires when the option collapsed while the underlying thesis
    held -- nothing was disproved, so there is nothing to cool off from.
    """
    return "stop loss" if decision.reason is ExitReason.STOP_LOSS else decision.reason.value


def _pending_underlyings(executor: CliExecutor) -> frozenset[str]:
    """Which underlyings have an unfilled order resting at the broker.

    Reduced to underlyings rather than contract symbols because that is the
    granularity the gates reason at: one position per underlying, one slot per
    underlying. A resting call on MSFT should block a put on MSFT too.
    """
    from agent.market import parse_occ_symbol

    found = set()
    for order in executor.open_orders():
        symbol = str(order.get("symbol", ""))
        try:
            found.add(parse_occ_symbol(symbol).underlying)
        except ValueError:
            # Not an option order -- something else placed it. Not ours to
            # reason about, but not a reason to fail either.
            continue
    return frozenset(found)


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
        ordered=result.ordered,
        opened=result.opened,
        closed=result.closed,
        considered=result.considered,
        realized=journal.realized_for_day(),
        unrealized=sum(p.unrealized_pnl for p in positions),
        book=book,
        refusals=result.refusals,
        dry_run=executor.is_dry_run,
    )
