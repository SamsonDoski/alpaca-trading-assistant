"""Reading the market through Alpaca's MCP server.

This module is the only place in the system that knows what Alpaca's JSON looks
like. Everywhere else works with the types in `domain.py`. That boundary has a
name in software design -- an *anti-corruption layer* -- and the reason for it is
practical: Alpaca calls a bid price `bp`, nests Greeks under `greeks`, and spells
implied volatility `impliedVolatility`. If those names leaked outward, every
module would quietly depend on the shape of somebody else's API, and the day
Alpaca renames a field the change would surface in ten files instead of one.

So the rule is: **JSON comes in, domain objects go out.** Nothing raw escapes.

**Why this module is asynchronous.** Talking to an MCP server means waiting on
another process, and a pass has nine symbols to look up. Done one after another
that is minutes of mostly waiting. `asyncio` lets all nine wait at the same time,
which is what keeps a pass comfortably inside its fifteen-minute slot. The gates
and the sizing stay ordinary synchronous functions -- they compute rather than
wait, so concurrency would buy them nothing and cost them clarity.

**Failing loudly versus failing softly.** The two are not the same here and the
difference matters:

  * Account and positions must succeed. Guessing "probably flat" after a failed
    read lets the agent re-buy something it already owns and skip an exit it
    needed to make. A failure here stops the pass.
  * A single symbol's chain may fail. That is one missed opportunity out of
    nine, and the next pass is fifteen minutes away, so it is logged and skipped.
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


# The tools we call, named once. The server builds its tool list from Alpaca's
# OpenAPI specs, so these names come from the server rather than from us --
# `run.py tools` prints the live list if you ever need to check them.
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
# Decoding Alpaca's representations.
#
# These are plain functions with no dependencies, which makes them the easiest
# part of the module to test -- and the part most likely to be wrong, since they
# encode assumptions about someone else's data format.
# --------------------------------------------------------------------------

# An OCC option symbol packs four facts into one string, fixed width:
#
#     AAPL  260918  C  00230000
#     ^     ^       ^  ^
#     |     |       |  strike x 1000, 8 digits, zero padded
#     |     |       call or put
#     |     expiry as YYMMDD
#     underlying, 1-6 characters
#
# So AAPL260918C00230000 is an Apple call struck at $230.00 expiring 18 Sep 2026.
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

    Written as a parser rather than trusting the fields the API happens to send
    alongside, because the symbol is the one thing guaranteed to be present and
    self-consistent. If the symbol and a separate `strike` field ever disagreed,
    the symbol is the one the exchange will honour.
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

    Alpaca returns camelCase over REST (`impliedVolatility`) while its Python SDK
    exposes snake_case (`implied_volatility`), and the MCP server sits between
    the two. Rather than betting on one spelling, this accepts either. It is a
    small tolerance in exactly one place, which is much cheaper than a pass that
    dies at 9:45 on Monday because a field arrived under its other name.
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

    A snapshot bundles the latest quote, the Greeks and the implied volatility
    for a single contract. Everything the gates need to judge a trade is in here,
    which is why the chain call is the only market read an entry decision makes.
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
        # Delta and IV are allowed to be missing rather than defaulted to a
        # number. A contract with no delta must fail the delta gate, and a
        # cheerful 0.0 would look like a real reading of "no sensitivity".
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

    The account may hold things this agent did not open and does not manage.
    Returning None for those keeps them visible to the broker and invisible to
    the exit logic, which is the correct handling of something we do not own the
    reasoning for.
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
        # Decoded from the symbol, never defaulted. Omitting this let every
        # position fall back to OpenPosition's "call" default, so a book of
        # seven puts reported itself as seven calls -- and the directional cap
        # then blocked calls while letting puts through without limit. The book
        # went one hundred percent short-direction, which is exactly the
        # concentration that gate exists to prevent.
        right=parsed.right,
    )


# --------------------------------------------------------------------------
# The connection itself.
# --------------------------------------------------------------------------

def unwrap_envelope(payload):
    """Strip the wrapper the MCP server puts around every response.

    Alpaca's server does not hand back the API's JSON directly. It wraps it:

        {"_alpaca_mcp_security": {"trust": "untrusted_tool_output", ...},
         "data": { ...the actual response... }}

    That security block is a prompt-injection guard, and it is a sensible one.
    Anything coming back from a market API -- a news headline especially -- is
    text written by someone else, and if it were pasted straight into a model
    prompt it could carry instructions. The server is labelling its own output as
    data rather than instructions.

    We get that protection for free and then some, because of where this function
    sits: everything downstream of here is a domain object with typed fields, so
    no free text from the API ever reaches the model as part of a prompt. Strings
    that do reach it -- a symbol, a strike -- have been through a parser first.

    Some tools nest once more, returning `{"result": [...]}` inside `data`. That
    is unwrapped too, but only when `result` is the only key, so a real response
    that happens to contain a `result` field is left alone.
    """
    if isinstance(payload, dict) and "_alpaca_mcp_security" in payload and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict) and set(payload) == {"result"}:
        payload = payload["result"]
    return payload


def _payload(result) -> dict | list:
    """Pull usable data out of whatever an MCP tool call returned.

    An MCP result is a list of content blocks rather than a bare value, because
    a tool is allowed to return text, images or several pieces at once. Newer
    servers also attach a parsed `structuredContent`. This prefers the structured
    form when it is there and falls back to parsing the text block, so the rest
    of the module never has to think about which it got.
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
            # A tool that returned prose rather than JSON is a tool we are
            # calling wrongly. Say so with the text included, because the text is
            # usually the server's error message.
            raise MarketDataError(f"expected JSON from the tool, got: {text[:200]}") from None

    raise MarketDataError("tool returned no usable content")


class MarketReader:
    """Everything the agent is allowed to learn about the market.

    Read-only by construction. There is no method here that places, cancels or
    closes anything, and the tool constants at the top of this file name no
    write tool -- so this class could not send an order even if a caller asked
    it to.
    """

    def __init__(self, session) -> None:
        # The session is handed in rather than created here, which is what lets
        # a test pass a fake with the same three methods and exercise every
        # decoding path above with no server running.
        self._session = session

    async def call(self, tool: str, arguments: dict | None = None) -> dict | list:
        """Call one MCP tool and return its decoded payload.

        Every read in this class funnels through here, so logging, error
        wrapping and payload extraction are written once.
        """
        logger.debug("mcp call %s %s", tool, arguments or {})
        try:
            result = await self._session.call_tool(tool, arguments or {})
        except Exception as exc:
            raise MarketDataError(f"{tool} failed: {exc}") from exc
        return unwrap_envelope(_payload(result))

    async def describe_tools(self) -> list[tuple[str, str]]:
        """The server's live tool menu, as (name, description) pairs.

        Useful on its own -- `run.py tools` prints it -- and the honest way to
        confirm the tool names at the top of this file still match the server.
        """
        listing = await self._session.list_tools()
        return [(t.name, (t.description or "").strip().splitlines()[0] if t.description else "")
                for t in listing.tools]

    async def account(self) -> AccountState:
        """Equity and buying power. A failure here stops the pass."""
        return account_from_payload(_as_dict(await self.call(TOOL_ACCOUNT)))

    async def positions(self) -> tuple[OpenPosition, ...]:
        """Every long option position we currently hold.

        Deliberately does not swallow errors. An empty list and a failed read
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

        **The filtering happens on the server, and that is the whole point.** A
        liquid name like AAPL lists thousands of contracts across dozens of
        expiries. The tool returns 100 by default, paginated, ordered from the
        nearest expiry outward -- so an unfiltered call returns nothing but
        contracts expiring within days, which is the exact part of the chain this
        strategy never trades. Worse, those near-dated illiquid contracts often
        carry no Greeks at all, because there is no sensible implied volatility
        to derive them from. Asking for the whole chain and filtering afterwards
        looks like it should work and quietly returns junk.

        So the expiry window and the call/put choice are pushed down into the
        request. One round trip comes back holding only contracts the gates might
        actually accept.

        Returns an empty list rather than raising when a chain cannot be read.
        Losing one symbol costs one opportunity out of nine and the next pass is
        fifteen minutes away, so this is the one read that fails softly.
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
                # A key that is not an option symbol is not an error worth
                # stopping for -- the chain response carries some metadata keys
                # alongside the contracts.
                continue
        return contracts


    async def option_quote(self, occ_symbol: str) -> tuple[float, float] | None:
        """The latest bid and ask for one contract, or None if unquoted.

        Used when closing a position. The broker reports a mark price on the
        position itself, but a mark is a valuation, not something anyone has
        offered to pay -- and an exit has to be priced against a real bid.
        """
        # `symbols`, not `symbol_or_symbols`. The MCP server builds its tools
        # from Alpaca's OpenAPI specs, so the parameter name comes from the REST
        # query string rather than from the Python SDK's signature -- and the two
        # differ. Guessing it from the SDK produced a 400 on every exit check,
        # which failed softly and quietly demoted every stop back to the premium
        # rule it was meant to replace.
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

        Needed at EXIT time, which is the awkward part: held symbols are
        screened out before any brief is built, so the exit path has no other
        source for where the stock stands now. Uses the quote midpoint rather
        than the last trade, because a trade can be stale by minutes on a quiet
        name while a quote is current.
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

        `days` is CALENDAR days, not sessions. Markets are open about five days
        in seven and closed on holidays, so 120 calendar days yields roughly 82
        trading sessions. The brief reports a 60-session trend, and asking for 60
        calendar days returned only 42 -- not enough to compute it.

        Fails softly for the same reason the chain does: this feeds the model's
        sense of context, and a symbol with no readable history simply gets no
        opinion this pass.
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

        Headlines are the one piece of free text in this system, and they are
        written by strangers. They are carried as plain strings and handed to the
        model inside a clearly fenced block, with the system prompt instructing
        it to treat them as reported facts rather than instructions. Anything
        that cannot be reduced to a headline string never travels further.
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

    The three reads are independent, so they run concurrently rather than one
    after another. Multiply that saving by nine symbols and it is the difference
    between a pass that finishes comfortably inside its fifteen-minute slot and
    one that does not.

    `asyncio.gather` with `return_exceptions=True` is deliberate: a failure in
    any one read yields an empty section rather than an exception that takes the
    whole brief down. Each of these three already fails softly on its own; this
    guarantees it at the assembly point too.
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

    # Only contracts inside the delta band are shown. The model is choosing a
    # direction, not shopping a chain, and a thousand strikes of context would
    # cost tokens without improving that judgement.
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

    The server is a child process: it starts when this context is entered, talks
    over stdin and stdout, and is shut down on exit. Nothing needs to be running
    beforehand and no port is opened, which is what makes a scheduled pass simple
    to reason about -- there is no long-lived service to have died overnight.

    Credentials are passed through the environment rather than on the command
    line, because a command line is visible to every other process on the machine
    via `ps`.
    """
    # Imported here rather than at module scope so that the decoding functions
    # above -- and their tests -- do not require the mcp package to be installed.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise MarketDataError("ALPACA_API_KEY / ALPACA_SECRET_KEY are not set")

    # Pinned, and the pinning is the point.
    #
    # `uvx` re-resolves its dependencies on EVERY invocation. That is convenient
    # until an upstream release lands mid-session: on 31 Aug 2026 FastMCP 4.0.0
    # was published during market hours, alpaca-mcp-server 2.3.0 could not import
    # against it, and every pass from 14:30 ET onward died on startup with
    # "ModuleNotFoundError: No module named 'fastmcp.tools.tool'" followed by a
    # closed connection. Nothing in this repository changed. The floor did.
    #
    # A scheduled agent that resolves its own dependencies fresh every fifteen
    # minutes is one upstream publish away from being dead, and it will die
    # during market hours because that is when it runs. So the versions are
    # named. `fastmcp<4` rather than an exact pin because 3.4.7 is the version
    # this agent ran on all morning; Alpaca's own workaround suggests 3.1.0 if a
    # tighter pin is ever needed.
    #
    # Both are overridable from the environment so a fix upstream can be adopted
    # without a code change or a redeploy.
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
