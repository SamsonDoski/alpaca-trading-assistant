"""Command line entry point.

    python run.py tools               list what the MCP server can do
    python run.py account             equity and buying power
    python run.py positions           what we currently hold
    python run.py chain AAPL          the tradable slice of one option chain
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
