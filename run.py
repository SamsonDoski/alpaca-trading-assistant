"""Command line entry point.

    python run.py tools               list what the MCP server can do
    python run.py account             equity and buying power
    python run.py positions           what we currently hold
    python run.py chain AAPL          the tradable slice of one option chain
    python run.py trade               one full pass (dry run unless --live)
    python run.py journal             what it did and declined today
    python run.py report              render it all as one HTML file
    python run.py notify-test         check the Discord webhook
    python run.py raw get_clock       call any read tool and dump its JSON

The last one exists for a specific reason. The MCP server builds its tool list
from Alpaca's OpenAPI specs, so parameter and field names come from Alpaca rather
than from any documentation we control. When a decoder in `market.py` needs to be
checked against reality, `raw` is how you see the reality.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402

from agent.market import MarketDataError, open_reader  # noqa: E402
from agent.settings import load_settings  # noqa: E402


async def cmd_tools(args) -> int:
    """Print the server's live tool menu, marking the ones this agent uses."""
    from agent.market import (
        TOOL_ACCOUNT,
        TOOL_CLOCK,
        TOOL_OPTION_CHAIN,
        TOOL_OPTION_SNAPSHOT,
        TOOL_POSITIONS,
        TOOL_STOCK_QUOTE,
    )

    ours = {TOOL_ACCOUNT, TOOL_POSITIONS, TOOL_CLOCK, TOOL_OPTION_CHAIN,
            TOOL_OPTION_SNAPSHOT, TOOL_STOCK_QUOTE}

    async with open_reader() as reader:
        tools = await reader.describe_tools()

    print(f"\n{len(tools)} tools available. Marked ones are what this agent calls.\n")
    for name, description in sorted(tools):
        mark = "  *" if name in ours else "   "
        print(f"{mark} {name:<34} {description[:70]}")

    missing = ours - {name for name, _ in tools}
    if missing:
        print(f"\n  WARNING: this agent expects tools the server does not offer: "
              f"{', '.join(sorted(missing))}")
        print("  The names in market.py need updating to match the server.")
        return 1

    print(f"\n  All {len(ours)} tools this agent needs are present.")
    return 0


async def cmd_account(args) -> int:
    async with open_reader() as reader:
        account = await reader.account()
        is_open = await reader.market_open()

    print(f"\n  equity              ${account.equity:>12,.2f}")
    print(f"  options buying power ${account.options_buying_power:>12,.2f}")
    print(f"  cash                 ${account.cash:>12,.2f}")
    print(f"  available to commit  ${account.available:>12,.2f}")
    print(f"\n  market is {'OPEN' if is_open else 'closed'}")
    return 0


async def cmd_positions(args) -> int:
    async with open_reader() as reader:
        positions = await reader.positions()

    if not positions:
        print("\n  No open option positions.")
        return 0

    print(f"\n  {len(positions)} open position(s):\n")
    total = 0.0
    for p in positions:
        total += p.unrealized_pnl
        days = p.days_to_expiry(date.today())
        print(f"    {p.occ_symbol:<24} x{p.quantity:<3} "
              f"entry ${p.entry_price:>6.2f}  now ${p.current_price:>6.2f}  "
              f"{p.return_pct:>+7.1%}  ${p.unrealized_pnl:>+9,.0f}  "
              f"{days}d to expiry")
    print(f"\n    unrealised total: ${total:+,.0f}")
    return 0


async def cmd_chain(args) -> int:
    """Show the part of a chain the gates would actually consider.

    Printing the whole chain is useless -- a liquid name lists thousands of
    contracts. This filters to the delta band and expiry window from settings,
    which is the same slice the agent works from, so what you see here is what
    it sees.
    """
    settings = load_settings(args.config)
    today = date.today()
    right = "put" if args.puts else "call"

    async with open_reader() as reader:
        contracts = await reader.option_chain(
            args.symbol.upper(),
            right=right,
            dte_min=settings.dte_min,
            dte_max=settings.dte_max,
        )

    if not contracts:
        print(f"\n  No chain data returned for {args.symbol.upper()}.")
        return 1

    tradable = [c for c in contracts
                if settings.delta_min <= c.abs_delta <= settings.delta_max]

    print(f"\n  {len(contracts)} {right}s listed for {args.symbol.upper()} between "
          f"{settings.dte_min} and {settings.dte_max} days out; "
          f"{len(tradable)} inside the "
          f"{settings.delta_min:.2f}-{settings.delta_max:.2f} delta band.\n")

    if not tradable:
        print("    Nothing in range. Widen delta_min/delta_max or dte_min/dte_max "
              "in config.yaml to see more.")
        return 0

    print(f"    {'contract':<24}{'strike':>8}{'delta':>7}{'spread':>8}"
          f"{'mid':>8}{'cost':>9}{'dte':>5}")
    print("    " + "-" * 69)
    for c in sorted(tradable, key=lambda c: c.strike):
        flag = " " if c.spread_pct <= settings.max_spread_pct else "!"
        print(f"    {c.occ_symbol:<24}{c.strike:>8.1f}{c.abs_delta:>7.2f}"
              f"{c.spread_pct:>7.1%}{flag}${c.mid:>7.2f}"
              f"${c.cost_per_contract:>8,.0f}{c.days_to_expiry(today):>5}")

    wide = sum(1 for c in tradable if c.spread_pct > settings.max_spread_pct)
    if wide:
        print(f"\n    ! {wide} of these quote wider than the "
              f"{settings.max_spread_pct:.0%} limit and would be refused.")
    return 0


async def cmd_propose(args) -> int:
    """Run the full read-and-reason pipeline for one symbol.

    This is the whole agent minus the parts that spend money: it gathers a
    brief, asks for a view, and prints what came back. Nothing is sized and
    nothing is ordered.
    """
    import anthropic

    from agent.market import build_brief
    from agent.proposer import Proposer, render_brief

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n  [stop] ANTHROPIC_API_KEY is not set in .env")
        return 1

    from agent.entry import decide_entry
    from agent.gates import GateContext, screen

    settings = load_settings(args.config)
    symbol = args.symbol.upper()

    # Exactly the order the live loop uses: assemble the world, run the free
    # entry screen, and only then spend a model call.
    async with open_reader() as reader:
        account, positions, is_open = await asyncio.gather(
            reader.account(), reader.positions(), reader.market_open())

        ctx = GateContext(
            today=date.today(),
            market_open=is_open or args.ignore_clock,
            trading_halted=os.getenv("TRADING_HALTED", "").lower() in ("true", "1", "yes"),
            account=account,
            open_positions=positions,
            cooling_off={},
            settings=settings,
        )

        screening = screen(symbol, ctx)
        print(f"\n  ENTRY SCREEN ({len(screening.trace)} gates run)")
        for name, verdict in screening.trace:
            mark = "pass" if verdict.decision.value == "allow" else "REFUSE"
            print(f"    {mark:>6}  {name}"
                  + (f" -- {verdict.reason}" if verdict.reason else ""))

        if not screening.approved:
            print(f"\n  STOPPED before the model call. No tokens spent.")
            return 0

        brief = await build_brief(reader, symbol, settings)

    if args.show_brief:
        print("\n" + "-" * 72)
        print(render_brief(brief))
        print("-" * 72)

    print(f"\n  {len(brief.bars)} daily bars, {len(brief.candidates)} contracts in range, "
          f"{len(brief.headlines)} headlines")
    print("  asking Claude...\n")

    proposal = Proposer(anthropic.Anthropic()).propose(brief)

    if proposal.thinking_summary:
        print("  REASONING")
        for line in proposal.thinking_summary.splitlines():
            print(f"    {line}")
        print()

    if proposal.is_actionable:
        print(f"  PROPOSAL: {proposal.direction.value.upper()} on {proposal.underlying} "
              f"(confidence {proposal.confidence:.2f})")
        print(f"  {proposal.rationale}\n")

    outcome = decide_entry(proposal, brief, ctx)

    if outcome.trace:
        print("  ORDER GATES")
        for name, verdict in outcome.trace:
            decision = verdict.decision.value
            mark = {"allow": "pass", "shrink": "TRIM", "deny": "REFUSE"}[decision]
            print(f"    {mark:>6}  {name}"
                  + (f" -- {verdict.reason}" if verdict.reason else ""))
        print()

    if not outcome.approved:
        print(f"  NO TRADE -- {outcome.reason}")
        return 0

    print(f"  APPROVED: {outcome.draft}")

    if not args.execute:
        print("  (not submitted -- pass --execute to send it)")
        return 0

    from agent.executor import CliExecutor, ExecutionError

    executor = CliExecutor(dry_run=not args.live)
    try:
        receipt = executor.buy_to_open(outcome.draft)
    except ExecutionError as exc:
        print(f"\n  ORDER FAILED -- {exc}")
        return 1

    print(f"\n  {receipt}")
    if receipt.dry_run:
        print("  (validated by Alpaca but not placed -- pass --live to place it)")
    return 0


async def cmd_trade(args) -> int:
    """One full pass: manage exits, then look for entries.

    This is what the schedule runs. Everything it needs is constructed here and
    handed to `run_pass`, which owns only the order things happen in.
    """
    import anthropic

    from agent.executor import CliExecutor
    from agent.journal import Journal
    from agent.loop import run_pass
    from agent.notify import Notifier
    from agent.proposer import Proposer

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  [stop] ANTHROPIC_API_KEY is not set in .env")
        return 1

    settings = load_settings(args.config)
    executor = CliExecutor(dry_run=not args.live)
    notifier = Notifier(os.getenv("DISCORD_WEBHOOK_URL"))
    journal = Journal()

    mode = "LIVE on the paper account" if args.live else "DRY RUN -- nothing will be sent"
    print(f"\n  {mode}")
    print(f"  {len(settings.symbols)} symbols, "
          f"{settings.stop_loss_pct:.0%} stop / {settings.take_profit_pct:.0%} target "
          f"({settings.break_even_win_rate:.0%} break-even win rate)")

    async with open_reader() as reader:
        result = await run_pass(
            reader,
            settings=settings,
            executor=executor,
            proposer=Proposer(anthropic.Anthropic()),
            journal=journal,
            notifier=notifier,
            ignore_clock=args.ignore_clock,
            trading_halted=os.getenv("TRADING_HALTED", "").lower() in ("true", "1", "yes"),
        )

    print(f"\n  {result.skipped_before_model} screened out before any model call")
    print(f"  {result.considered} considered · {result.opened} opened · "
          f"{result.closed} closed")

    if result.refusals:
        print("\n  CONSIDERED AND DECLINED")
        for symbol, reason in result.refusals:
            print(f"    {symbol:<8} {reason}")

    if result.errors:
        print("\n  ERRORS")
        for error in result.errors:
            print(f"    {error}")
        return 1
    return 0


async def cmd_notify_test(args) -> int:
    """Send a probe to the Discord webhook and report the real result."""
    from agent.notify import Notifier

    notifier = Notifier(os.getenv("DISCORD_WEBHOOK_URL"))
    if not notifier.enabled:
        print("\n  [fail] DISCORD_WEBHOOK_URL is not set in .env")
        return 1

    ok, detail = notifier.verify()
    print(f"\n  {'[ok]' if ok else '[fail]'} {detail}")
    return 0 if ok else 1


async def cmd_journal(args) -> int:
    """What the agent did and declined today."""
    from agent.journal import Journal

    journal = Journal()
    decisions = journal.decisions_for_day()
    events = journal.recent(limit=args.limit)

    print(f"\n  {len(decisions)} decision(s) today\n")
    for row in decisions[:args.limit]:
        mark = "OPEN " if row.approved else "skip "
        conviction = f"{row.confidence:.2f}" if row.confidence is not None else "  -- "
        print(f"    {row.at[11:16]}  {mark} {row.underlying:<7} {conviction}  "
              f"{row.reason[:70]}")

    if events:
        print(f"\n  Recent events\n")
        for event in events:
            pnl = f"  ${event.pnl:+,.0f}" if event.pnl is not None else ""
            print(f"    {event.at[:16]}  {event.action:<7} {event.symbol:<24} "
                  f"{event.detail[:60]}{pnl}")

    totals = journal.summary()
    print(f"\n  Lifetime: {totals['closed']} closed, "
          f"{totals['win_rate']:.0%} won, ${totals['total_pnl']:+,.0f}")
    return 0


async def cmd_report(args) -> int:
    """Render the journal as a single self-contained HTML file."""
    from agent.journal import Journal
    from agent.report import write

    account, positions = None, ()
    if not args.offline:
        try:
            async with open_reader() as reader:
                account, positions = await asyncio.gather(
                    reader.account(), reader.positions())
        except MarketDataError as exc:
            print(f"  [note] could not reach the broker ({exc}); "
                  f"reporting from the journal alone")

    path = write(Journal(), args.output, account=account, positions=positions,
                 dry_run=not args.live)
    print(f"\n  wrote {path.resolve()}")
    return 0


async def cmd_raw(args) -> int:
    """Call one read tool and print exactly what came back."""
    arguments = json.loads(args.arguments) if args.arguments else {}

    async with open_reader() as reader:
        payload = await reader.call(args.tool, arguments)

    print(json.dumps(payload, indent=2, default=str)[:args.limit])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="list the MCP server's tools").set_defaults(func=cmd_tools)
    sub.add_parser("account", help="equity, buying power, market status").set_defaults(
        func=cmd_account)
    sub.add_parser("positions", help="open option positions").set_defaults(func=cmd_positions)

    chain = sub.add_parser("chain", help="the tradable slice of one option chain")
    chain.add_argument("symbol")
    chain.add_argument("--puts", action="store_true", help="show puts instead of calls")
    chain.set_defaults(func=cmd_chain)

    propose = sub.add_parser("propose", help="gather a brief and ask Claude for a view")
    propose.add_argument("symbol")
    propose.add_argument("--show-brief", action="store_true",
                         help="print exactly what the model was shown")
    propose.add_argument("--ignore-clock", action="store_true",
                         help="run the market-open gate as if the market were open")
    propose.add_argument("--execute", action="store_true",
                         help="submit an approved draft (validated only, unless --live)")
    propose.add_argument("--live", action="store_true",
                         help="with --execute, actually place the order")
    propose.set_defaults(func=cmd_propose)

    trade = sub.add_parser("trade", help="one full pass: exits, then entries")
    trade.add_argument("--live", action="store_true",
                       help="actually place orders (default is a dry run)")
    trade.add_argument("--ignore-clock", action="store_true",
                       help="run as if the market were open")
    trade.set_defaults(func=cmd_trade)

    sub.add_parser("notify-test", help="check the Discord webhook works").set_defaults(
        func=cmd_notify_test)

    jrnl = sub.add_parser("journal", help="what the agent did and declined today")
    jrnl.add_argument("--limit", type=int, default=30)
    jrnl.set_defaults(func=cmd_journal)

    report = sub.add_parser("report", help="render the journal as one HTML file")
    report.add_argument("--output", default="state/report.html")
    report.add_argument("--offline", action="store_true",
                        help="skip the broker read and use the journal only")
    report.add_argument("--live", action="store_true",
                        help="label the report as live rather than dry run")
    report.set_defaults(func=cmd_report)

    raw = sub.add_parser("raw", help="call any read tool and dump its JSON")
    raw.add_argument("tool")
    raw.add_argument("arguments", nargs="?", help='JSON object, e.g. \'{"symbol": "AAPL"}\'')
    raw.add_argument("--limit", type=int, default=4000, help="characters to print")
    raw.set_defaults(func=cmd_raw)

    args = parser.parse_args()
    load_dotenv()

    try:
        return asyncio.run(args.func(args))
    except MarketDataError as exc:
        print(f"\n  [stop] {exc}")
        return 1
    except FileNotFoundError:
        print("\n  [stop] could not start the MCP server. Is uv installed?")
        print("         Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh")
        return 1


if __name__ == "__main__":
    sys.exit(main())
