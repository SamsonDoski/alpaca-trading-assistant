"""Reading the market through Alpaca's MCP server.

The only place that knows what Alpaca's JSON looks like -- an *anti-corruption
layer*. Alpaca calls a bid price `bp` and spells implied volatility
`impliedVolatility`; if those names leaked outward, a renamed field would surface
in ten modules instead of one. **JSON comes in, domain objects go out.**

Asynchronous because talking to an MCP server is mostly waiting, and a pass has
many symbols to look up. The gates and sizing stay synchronous -- they compute
rather than wait, so concurrency would cost clarity and buy nothing.

Failing loudly versus softly is a real distinction here. Account and positions
must succeed: guessing "probably flat" lets the agent re-buy what it owns and
skip an exit it needed. One symbol's chain may fail -- that is one missed
opportunity, and the next pass is fifteen minutes away.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from agent.domain import (
    AccountState,
    MarketBrief,
    OpenPosition,
    OptionContract,
    PriceBar,
)

logger = logging.getLogger(__name__)


# The tools we call, named once. `run.py tools` prints the server's live list.
#
# Note what is deliberately absent: place_option_market_order, close_position and
# close_all_positions all exist on that server. This module never names them, so
# no code path here can reach them even by accident.
TOOL_ACCOUNT = "get_account_info"
TOOL_POSITIONS = "get_all_positions"
TOOL_CLOCK = "get_clock"
TOOL_OPTION_CHAIN = "get_option_chain"
TOOL_OPTION_SNAPSHOT = "get_option_snapshot"
TOOL_STOCK_QUOTE = "get_stock_latest_quote"
TOOL_STOCK_BARS = "get_stock_bars"
TOOL_NEWS = "get_news"
TOOL_OPTION_QUOTE = "get_option_latest_quote"


class MarketDataError(RuntimeError):
    """A read that the pass cannot safely continue without."""


# --------------------------------------------------------------------------
# Decoding Alpaca's representations. Plain dependency-free functions: the easiest
# part to test, and the likeliest to be wrong, since they encode assumptions
# about someone else's data format.
# --------------------------------------------------------------------------

# An OCC option symbol packs four facts into one fixed-width string:
#
#     AAPL  260918  C  00230000
#     ^     ^       ^  ^
#     |     |       |  strike x 1000, 8 digits, zero padded
#     |     |       call or put
#     |     expiry as YYMMDD
#     underlying, 1-6 characters
_OCC_PATTERN = re.compile(r"^(?P<root>[A-Z]{1,6})(?P<expiry>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True, slots=True)
class OccSymbol:
    """The four facts decoded out of an OCC symbol."""

    underlying: str
    expiry: date
    right: str
    strike: float


def parse_occ_symbol(symbol: str) -> OccSymbol:
    """Decode an OCC option symbol into its parts.

    Parsed rather than read off whatever fields the API sends alongside: the
    symbol is the one thing guaranteed present and self-consistent, and if it
    ever disagreed with a separate `strike` field, the exchange honours the
    symbol.
    """
    match = _OCC_PATTERN.match(symbol.strip().upper())
    if not match:
        raise ValueError(f"not a valid OCC option symbol: {symbol!r}")

    parts = match.groupdict()
    expiry = datetime.strptime(parts["expiry"], "%y%m%d").date()

    return OccSymbol(
        underlying=parts["root"],
        expiry=expiry,
        right="call" if parts["right"] == "C" else "put",
        # The last eight digits are the strike in thousandths of a dollar, so
        # 00230000 is $230.000 rather than $230,000.
        strike=int(parts["strike"]) / 1000.0,
    )


def _first(mapping: dict, *names, default=None):
    """The first of several possible key names that is actually present.

    Alpaca returns camelCase over REST while its Python SDK exposes snake_case,
    and the MCP server sits between the two. Accepting either is a small
    tolerance in one place, and cheaper than a pass that dies on Monday because
    a field arrived under its other name.
    """
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _to_float(value, default: float = 0.0) -> float:
    """Coerce a JSON number that might be a string, None, or missing."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def contract_from_snapshot(occ_symbol: str, snapshot: dict) -> OptionContract:
    """Build an OptionContract from one entry of an option chain response.

    A snapshot bundles quote, Greeks and implied volatility for one contract --
    everything the gates need, which is why the chain call is the only market
    read an entry decision makes.
    """
    parsed = parse_occ_symbol(occ_symbol)

    quote = _first(snapshot, "latestQuote", "latest_quote", default={}) or {}
    greeks = _first(snapshot, "greeks", default={}) or {}

    return OptionContract(
        occ_symbol=occ_symbol,
        underlying=parsed.underlying,
        right=parsed.right,
        strike=parsed.strike,
        expiry=parsed.expiry,
        # `bp` and `ap` are Alpaca's short names for bid price and ask price.
        bid=_to_float(_first(quote, "bp", "bid_price", "bidPrice")),
        ask=_to_float(_first(quote, "ap", "ask_price", "askPrice")),
        # Missing rather than defaulted: a contract with no delta must fail the
        # delta gate, and a cheerful 0.0 would read as a real "no sensitivity".
        delta=_first(greeks, "delta"),
        implied_volatility=_first(snapshot, "impliedVolatility", "implied_volatility"),
        open_interest=_first(snapshot, "openInterest", "open_interest"),
    )


def account_from_payload(payload: dict) -> AccountState:
    """Build an AccountState from the account tool's response."""
    return AccountState(
        equity=_to_float(_first(payload, "equity", "portfolio_value")),
        options_buying_power=_to_float(
            _first(payload, "options_buying_power", "optionsBuyingPower",
                   "buying_power", "buyingPower")),
        cash=_to_float(_first(payload, "cash")),
    )


def position_from_payload(payload: dict) -> OpenPosition | None:
    """Build an OpenPosition, or None if this row is not a long option.

    The account may hold things this agent did not open. Returning None keeps
    those visible to the broker and invisible to the exit logic, which is the
    right handling of something we do not own the reasoning for.
    """
    symbol = str(_first(payload, "symbol", default=""))
    try:
        parsed = parse_occ_symbol(symbol)
    except ValueError:
        return None

    quantity = int(_to_float(_first(payload, "qty", "quantity")))
    if quantity <= 0:
        return None

    entry = _to_float(_first(payload, "avg_entry_price", "avgEntryPrice"))
    current = _to_float(_first(payload, "current_price", "currentPrice"), entry)

    return OpenPosition(
        occ_symbol=symbol,
        underlying=parsed.underlying,
        quantity=quantity,
        entry_price=entry,
        current_price=current,
        expiry=parsed.expiry,
        # Decoded from the symbol, never defaulted. Omitting it let every
        # position fall back to the "call" default, so a book of seven puts
        # reported itself as seven calls and the directional cap capped the
        # wrong side -- exactly the concentration that gate exists to prevent.
        right=parsed.right,
    )


# --------------------------------------------------------------------------
# The connection itself.
# --------------------------------------------------------------------------

def unwrap_envelope(payload):
    """Strip the wrapper the MCP server puts around every response.

    Alpaca wraps every payload as `{"_alpaca_mcp_security": {...}, "data": {...}}`.
    That block is a prompt-injection guard: anything from a market API -- a news
    headline especially -- is text written by someone else, so the server labels
    its own output as data rather than instructions.

    We get that protection and then some, because everything downstream of here
    is a domain object with typed fields. No free text reaches the model as part
    of a prompt without passing a parser first.

    Some tools nest once more as `{"result": [...]}`. That is unwrapped too, but
    only when `result` is the sole key, so a genuine `result` field survives.
    """
    if isinstance(payload, dict) and "_alpaca_mcp_security" in payload and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return payload


def _payload(result) -> dict | list:
    """Pull usable data out of whatever an MCP tool call returned.

    An MCP result is a list of content blocks, since a tool may return text,
    images or several pieces at once. Newer servers also attach a parsed
    `structuredContent`, preferred here, with the text block as the fallback --
    so nothing downstream has to know which arrived.
    """
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Prose instead of JSON means we are calling the tool wrongly. The
            # text is usually the server's error message, so include it.
            raise MarketDataError(f"expected JSON from the tool, got: {text[:200]}") from None

    raise MarketDataError("tool returned no usable content")


class MarketReader:
    """Everything the agent is allowed to learn about the market.

    Read-only by construction: no method places, cancels or closes anything, and
    the constants above name no write tool, so this class could not send an order
    even if asked.
    """

    def __init__(self, session) -> None:
        # Injected rather than constructed, so a test can pass a fake with the
        # same three methods and exercise every decoding path with no server.
        self._session = session

    async def call(self, tool: str, arguments: dict | None = None) -> dict | list:
        """Call one MCP tool and return its decoded payload.

        Every read funnels through here, so logging, error wrapping and payload
        extraction are written once.
        """
        logger.debug("mcp call %s %s", tool, arguments or {})
        try:
            result = await self._session.call_tool(tool, arguments or {})
        except Exception as exc:
            raise MarketDataError(f"{tool} failed: {exc}") from exc
        return unwrap_envelope(_payload(result))

    async def describe_tools(self) -> list[tuple[str, str]]:
        """The server's live tool menu, as (name, description) pairs.

        `run.py tools` prints it, and it is the honest way to confirm the names
        above still match the server.
        """
        listing = await self._session.list_tools()
        return [(t.name, (t.description or "").strip().splitlines()[0] if t.description else "")
                for t in listing.tools]

    async def account(self) -> AccountState:
        """Equity and buying power. A failure here stops the pass."""
        return account_from_payload(_as_dict(await self.call(TOOL_ACCOUNT)))

    async def positions(self) -> tuple[OpenPosition, ...]:
        """Every long option position we currently hold.

        Deliberately does not swallow errors: an empty list and a failed read
        look identical to the caller, and one of them means "sell nothing today"
        when the truth was "we could not see what we own".
        """
        payload = await self.call(TOOL_POSITIONS)
        rows = payload if isinstance(payload, list) else _first(
            payload, "positions", "result", default=[])

        found = []
        for row in rows or []:
            if isinstance(row, dict):
                position = position_from_payload(row)
                if position is not None:
                    found.append(position)
        return tuple(found)

    async def market_open(self) -> bool:
        """Whether the US equity market is open right now."""
        payload = _as_dict(await self.call(TOOL_CLOCK))
        return bool(_first(payload, "is_open", "isOpen", default=False))

    async def option_chain(
        self,
        underlying: str,
        *,
        right: str | None = None,
        dte_min: int | None = None,
        dte_max: int | None = None,
        limit: int = 1000,
        today: date | None = None,
    ) -> list[OptionContract]:
        """The contracts for one underlying, priced with Greeks, already narrowed.

        **The filtering happens on the server, and that is the point.** A liquid
        name lists thousands of contracts; the tool returns 100 by default,
        paginated from the nearest expiry outward. So an unfiltered call returns
        only contracts expiring within days -- the part of the chain this strategy
        never trades, and often carrying no Greeks at all. Fetching everything and
        filtering afterwards looks right and quietly returns junk.

        Pushing the expiry window and call/put choice into the request brings back
        one round trip of contracts the gates might actually accept.

        Fails softly: losing one symbol costs one opportunity, and the next pass
        is fifteen minutes away.
        """
        today = today or date.today()
        arguments: dict[str, object] = {"underlying_symbol": underlying, "limit": limit}

        if right is not None:
            arguments["type"] = right
        if dte_min is not None:
            arguments["expiration_date_gte"] = (today + timedelta(days=dte_min)).isoformat()
        if dte_max is not None:
            arguments["expiration_date_lte"] = (today + timedelta(days=dte_max)).isoformat()

        try:
            payload = _as_dict(await self.call(TOOL_OPTION_CHAIN, arguments))
        except MarketDataError as exc:
            logger.warning("chain unavailable for %s: %s", underlying, exc)
            return []

        snapshots = _first(payload, "snapshots", default=payload) or {}
        if not isinstance(snapshots, dict):
            return []

        contracts = []
        for occ_symbol, snapshot in snapshots.items():
            if not isinstance(snapshot, dict):
                continue
            try:
                contracts.append(contract_from_snapshot(occ_symbol, snapshot))
            except ValueError:
                # The chain response carries metadata keys alongside the
                # contracts; a non-symbol key is not worth stopping for.
                continue
        return contracts


    async def option_quote(self, occ_symbol: str) -> tuple[float, float] | None:
        """The latest bid and ask for one contract, or None if unquoted.

        Used when closing. The broker reports a mark on the position, but a mark
        is a valuation rather than something anyone has offered to pay, and an
        exit has to be priced against a real bid.
        """
        # `symbols`, not `symbol_or_symbols`: the tool is built from Alpaca's
        # OpenAPI spec, so the name comes from the REST query string rather than
        # the SDK signature. Guessing wrong produced a 400 on every exit check,
        # which failed softly and demoted every stop back to the premium rule.
        try:
            payload = _as_dict(await self.call(TOOL_OPTION_QUOTE,
                                               {"symbols": occ_symbol}))
        except MarketDataError as exc:
            logger.warning("no quote for %s: %s", occ_symbol, exc)
            return None

        quotes = _first(payload, "quotes", default=payload) or {}
        quote = quotes.get(occ_symbol) if isinstance(quotes, dict) else None
        if not isinstance(quote, dict):
            return None

        bid = _to_float(_first(quote, "bp", "bid_price", "bidPrice"))
        ask = _to_float(_first(quote, "ap", "ask_price", "askPrice"))
        return (bid, ask) if bid > 0 or ask > 0 else None

    async def stock_price(self, symbol: str) -> float | None:
        """The underlying's current price.

        Needed at EXIT time, and held symbols are screened out before any brief
        is built, so the exit path has no other source. Uses the quote midpoint
        rather than the last trade, which can be stale by minutes on a quiet name.
        """
        try:
            payload = _as_dict(await self.call(TOOL_STOCK_QUOTE, {"symbols": symbol}))
        except MarketDataError as exc:
            logger.warning("no stock quote for %s: %s", symbol, exc)
            return None

        quotes = _first(payload, "quotes", default=payload) or {}
        quote = quotes.get(symbol) if isinstance(quotes, dict) else None
        if not isinstance(quote, dict):
            return None

        bid = _to_float(_first(quote, "bp", "bid_price", "bidPrice"))
        ask = _to_float(_first(quote, "ap", "ask_price", "askPrice"))
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return bid or ask or None

    async def recent_bars(self, symbol: str, days: int = 120) -> list[PriceBar]:
        """Daily bars for the underlying, oldest first.

        `days` is CALENDAR days, not sessions -- 120 of them yields roughly 82
        trading days. The brief reports a 60-session trend, and asking for 60
        calendar days returned only 42, which was not enough to compute it.

        Fails softly like the chain: a symbol with no readable history simply
        gets no opinion this pass.
        """
        try:
            payload = _as_dict(await self.call(TOOL_STOCK_BARS, {
                "symbols": symbol,
                "timeframe": "1Day",
                "days": days,
                "limit": days,
                "sort": "asc",
            }))
        except MarketDataError as exc:
            logger.warning("bars unavailable for %s: %s", symbol, exc)
            return []

        # The response is keyed by symbol because the tool accepts several.
        series = _first(payload, "bars", default={}) or {}
        rows = series.get(symbol, []) if isinstance(series, dict) else series

        bars = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            stamp = str(_first(row, "t", "timestamp", default=""))[:10]
            try:
                day = datetime.strptime(stamp, "%Y-%m-%d").date()
            except ValueError:
                continue
            bars.append(PriceBar(
                day=day,
                open=_to_float(_first(row, "o", "open")),
                high=_to_float(_first(row, "h", "high")),
                low=_to_float(_first(row, "l", "low")),
                close=_to_float(_first(row, "c", "close")),
                volume=_to_float(_first(row, "v", "volume")),
            ))
        return bars

    async def headlines(self, symbol: str, limit: int = 6) -> list[str]:
        """Recent news headlines for the underlying.

        The one piece of free text in this system, and it is written by
        strangers. Carried as plain strings and handed to the model inside a
        fenced block, with the system prompt instructing it to treat them as
        reported facts rather than instructions. Anything that cannot be reduced
        to a headline string never travels further.
        """
        try:
            payload = await self.call(TOOL_NEWS, {
                "symbols": symbol,
                "limit": limit,
                "exclude_contentless": True,
                "sort": "desc",
            })
        except MarketDataError as exc:
            logger.warning("news unavailable for %s: %s", symbol, exc)
            return []

        rows = payload if isinstance(payload, list) else _first(
            _as_dict(payload), "news", "result", default=[])

        found = []
        for row in rows or []:
            if isinstance(row, dict):
                text = _first(row, "headline", "title")
                if text:
                    found.append(str(text).strip())
        return found[:limit]


async def build_brief(reader: MarketReader, underlying: str, settings,
                      *, today: date | None = None) -> MarketBrief:
    """Gather everything the model will see about one symbol.

    The reads are independent, so they run concurrently; multiplied across the
    watchlist that is the difference between a pass finishing inside its slot and
    one that does not.

    `return_exceptions=True` is deliberate: one failed read yields an empty
    section rather than taking the whole brief down. Each read already fails
    softly on its own, and this guarantees it at the assembly point too.
    """
    today = today or date.today()

    bars, calls, puts, news = await asyncio.gather(
        reader.recent_bars(underlying),
        reader.option_chain(underlying, right="call",
                            dte_min=settings.dte_min, dte_max=settings.dte_max, today=today),
        reader.option_chain(underlying, right="put",
                            dte_min=settings.dte_min, dte_max=settings.dte_max, today=today),
        reader.headlines(underlying),
        return_exceptions=True,
    )

    def usable(result):
        return result if isinstance(result, list) else []

    # Only the delta band is shown. The model is choosing a direction, not
    # shopping a chain, and a thousand strikes would cost tokens without
    # improving that judgement.
    candidates = [
        c for c in usable(calls) + usable(puts)
        if settings.delta_min <= c.abs_delta <= settings.delta_max
    ]

    return MarketBrief(
        underlying=underlying,
        as_of=today,
        bars=tuple(usable(bars)),
        candidates=tuple(sorted(candidates, key=lambda c: (c.right, c.strike))),
        headlines=tuple(usable(news)),
    )


def _as_dict(payload) -> dict:
    """Narrow a payload to a dict, with a clear error if it is not one."""
    if isinstance(payload, dict):
        return payload
    raise MarketDataError(f"expected an object, got {type(payload).__name__}")


@asynccontextmanager
async def open_reader(*, paper: bool = True):
    """Start the Alpaca MCP server and yield a MarketReader over it.

    The server is a child process, started on entry and shut down on exit over
    stdin and stdout. No port is opened and nothing needs to be running
    beforehand, so a scheduled pass has no long-lived service that might have
    died overnight.

    Credentials go through the environment, never the command line, which is
    visible to every other process via `ps`.
    """
    # Imported here, not at module scope, so the decoding functions above and
    # their tests do not require the mcp package to be installed.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise MarketDataError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")

    # Pinned, and the pinning is the point. `uvx` re-resolves dependencies on
    # EVERY invocation, which is convenient until an upstream release lands
    # mid-session: FastMCP 4.0.0 was published during market hours on 31 Aug
    # 2026, alpaca-mcp-server 2.3.0 could not import against it, and every pass
    # from 14:30 ET died on startup. Nothing here changed -- the floor did.
    #
    # A scheduled agent that re-resolves its dependencies every fifteen minutes
    # is one upstream publish away from dead, and it will die during market hours
    # because that is when it runs. `fastmcp<4` rather than an exact pin because
    # 3.4.7 is what it ran on all morning. Both are overridable from the
    # environment, so an upstream fix needs no redeploy.
    server_spec = os.getenv("ALPACA_MCP_SPEC", "alpaca-mcp-server==2.3.0")
    fastmcp_spec = os.getenv("FASTMCP_SPEC", "fastmcp<4")

    parameters = StdioServerParameters(
        command="uvx",
        args=["--from", server_spec, "--with", fastmcp_spec, "alpaca-mcp-server"],
        env={
            **os.environ,
            "ALPACA_API_KEY": api_key,
            "ALPACA_SECRET_KEY": secret_key,
            "ALPACA_PAPER_TRADE": "true" if paper else "false",
        },
    )

    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield MarketReader(session)
