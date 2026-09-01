"""Tests for the market reading layer.

Everything here runs without an MCP server, without a network and without an
Alpaca account. The decoding functions are plain functions over dictionaries, and
MarketReader takes its session as a constructor argument, so a small fake with
two methods stands in for the real thing.

That is not a testing trick -- it is the payoff from the design. A module that
built its own connection could only be tested by having one.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import date

import pytest

from agent.market import (
    MarketDataError,
    MarketReader,
    account_from_payload,
    contract_from_snapshot,
    parse_occ_symbol,
    position_from_payload,
    unwrap_envelope,
)


# --- Decoding an OCC symbol -----------------------------------------------

def test_parses_a_call_symbol():
    parsed = parse_occ_symbol("AAPL260918C00230000")
    assert parsed.underlying == "AAPL"
    assert parsed.expiry == date(2026, 9, 18)
    assert parsed.right == "call"
    assert parsed.strike == 230.0


def test_parses_a_put_symbol():
    assert parse_occ_symbol("TSLA261016P00400000").right == "put"


def test_strike_is_thousandths_not_thousands():
    """00230000 is $230.00, not $230,000. Getting this wrong by 1000x would
    make every sizing decision nonsense, so it is worth its own test."""
    assert parse_occ_symbol("AAPL260918C00230000").strike == 230.0
    assert parse_occ_symbol("SPY260918C00612500").strike == 612.5


def test_a_short_root_symbol_still_parses():
    assert parse_occ_symbol("F260918C00012000").underlying == "F"


def test_a_plain_stock_symbol_is_rejected():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL")


def test_a_malformed_symbol_is_rejected():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL26091XC00230000")


# --- Decoding a chain snapshot --------------------------------------------

CAMEL_SNAPSHOT = {
    "latestQuote": {"bp": 4.95, "ap": 5.05},
    "greeks": {"delta": 0.6412},
    "impliedVolatility": 0.281,
    "openInterest": 1500,
}

SNAKE_SNAPSHOT = {
    "latest_quote": {"bid_price": 4.95, "ask_price": 5.05},
    "greeks": {"delta": 0.6412},
    "implied_volatility": 0.281,
    "open_interest": 1500,
}


@pytest.mark.parametrize("snapshot", [CAMEL_SNAPSHOT, SNAKE_SNAPSHOT])
def test_both_field_spellings_decode_identically(snapshot):
    """Alpaca uses camelCase over REST and snake_case in its SDK. Either must
    produce the same contract -- this is the whole reason `_first` exists."""
    contract = contract_from_snapshot("AAPL260918C00230000", snapshot)
    assert contract.bid == 4.95
    assert contract.ask == 5.05
    assert contract.delta == pytest.approx(0.6412)
    assert contract.implied_volatility == pytest.approx(0.281)


def test_snapshot_decoding_fills_in_the_contract_terms():
    contract = contract_from_snapshot("AAPL260918C00230000", CAMEL_SNAPSHOT)
    assert contract.underlying == "AAPL"
    assert contract.right == "call"
    assert contract.strike == 230.0
    assert contract.expiry == date(2026, 9, 18)


def test_a_missing_delta_stays_none_rather_than_becoming_zero():
    """A defaulted 0.0 would read as a real measurement of no sensitivity, and
    the delta gate would refuse it for the wrong reason."""
    contract = contract_from_snapshot("AAPL260918C00230000", {"latestQuote": {"bp": 1, "ap": 2}})
    assert contract.delta is None
    assert contract.abs_delta == 0.0


def test_a_contract_with_no_quote_reports_a_hopeless_spread():
    contract = contract_from_snapshot("AAPL260918C00230000", {})
    assert contract.spread_pct == 1.0


def test_decoded_contract_computes_its_own_spread():
    contract = contract_from_snapshot("AAPL260918C00230000", CAMEL_SNAPSHOT)
    assert contract.mid == pytest.approx(5.00)
    assert contract.spread_pct == pytest.approx(0.02)


# --- Decoding account and positions ---------------------------------------

def test_account_decodes_equity_and_options_buying_power():
    account = account_from_payload(
        {"equity": "100000", "options_buying_power": "97500", "cash": "100000"})
    assert account.equity == 100_000
    assert account.options_buying_power == 97_500


def test_account_falls_back_to_general_buying_power():
    account = account_from_payload({"equity": "50000", "buying_power": "48000", "cash": "50000"})
    assert account.options_buying_power == 48_000


def test_position_decodes_a_long_option():
    position = position_from_payload({
        "symbol": "AAPL260918C00230000",
        "qty": "3",
        "avg_entry_price": "5.00",
        "current_price": "6.20",
    })
    assert position is not None
    assert position.underlying == "AAPL"
    assert position.quantity == 3
    assert position.cost_basis == 1500.0
    assert position.unrealized_pnl == pytest.approx(360.0)


def test_a_put_position_decodes_as_a_put_not_a_call():
    """A live bug: this was defaulted rather than decoded, so a book of seven
    puts reported itself as seven calls. The directional cap then blocked calls
    and let puts through without limit, and the book went 100% one-sided --
    exactly the concentration the gate exists to prevent."""
    put = position_from_payload({
        "symbol": "GOOGL261002P00350000", "qty": "2",
        "avg_entry_price": "18.31", "current_price": "18.00"})
    assert put is not None
    assert put.right == "put"
    assert put.direction.value == "down"


def test_a_call_position_decodes_as_a_call():
    call = position_from_payload({
        "symbol": "MSFT261009C00495000", "qty": "1",
        "avg_entry_price": "26.90", "current_price": "23.15"})
    assert call is not None
    assert call.right == "call"
    assert call.direction.value == "up"


def test_direction_is_never_defaulted_across_a_mixed_book():
    """The whole book, decoded together -- the shape the gate actually sees."""
    rows = [
        {"symbol": "GOOGL261002P00350000", "qty": "2", "avg_entry_price": "18"},
        {"symbol": "IWM260930P00300000", "qty": "4", "avg_entry_price": "9"},
        {"symbol": "MSFT261009C00495000", "qty": "1", "avg_entry_price": "26"},
    ]
    book = [position_from_payload(r) for r in rows]
    assert [p.right for p in book] == ["put", "put", "call"]


def test_a_stock_position_is_ignored_rather_than_managed():
    """The account may hold things this agent did not open. Those stay the
    broker's business and never reach the exit logic."""
    assert position_from_payload({"symbol": "AAPL", "qty": "100"}) is None


def test_a_short_position_is_ignored():
    assert position_from_payload({"symbol": "AAPL260918C00230000", "qty": "-2"}) is None


# --- The reader, against a fake session -----------------------------------

@dataclass
class FakeBlock:
    text: str


@dataclass
class FakeResult:
    content: list
    structuredContent: dict | None = None


class FakeSession:
    """Stands in for an MCP ClientSession. Records what was asked for."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name not in self.responses:
            raise RuntimeError(f"no fake response for {name}")
        payload = self.responses[name]
        return FakeResult(content=[FakeBlock(text=json.dumps(payload))])


def run(coro):
    return asyncio.run(coro)


def test_reader_returns_an_account_state():
    session = FakeSession({"get_account_info": {"equity": 100_000,
                                                "options_buying_power": 100_000,
                                                "cash": 100_000}})
    account = run(MarketReader(session).account())
    assert account.equity == 100_000


def test_reader_filters_positions_down_to_long_options():
    session = FakeSession({"get_all_positions": [
        {"symbol": "AAPL260918C00230000", "qty": "2", "avg_entry_price": "5.0",
         "current_price": "5.5"},
        {"symbol": "MSFT", "qty": "50", "avg_entry_price": "400"},
    ]})
    positions = run(MarketReader(session).positions())
    assert len(positions) == 1
    assert positions[0].underlying == "AAPL"


def test_reader_reads_the_clock():
    session = FakeSession({"get_clock": {"is_open": True}})
    assert run(MarketReader(session).market_open()) is True


def test_reader_builds_contracts_from_a_chain():
    session = FakeSession({"get_option_chain": {"snapshots": {
        "AAPL260918C00230000": CAMEL_SNAPSHOT,
        "AAPL260918P00230000": CAMEL_SNAPSHOT,
    }}})
    contracts = run(MarketReader(session).option_chain("AAPL"))
    assert len(contracts) == 2
    assert {c.right for c in contracts} == {"call", "put"}


def test_a_failed_chain_returns_empty_rather_than_raising():
    """One missing symbol is a missed opportunity, not a reason to abandon the
    other eight."""
    session = FakeSession({})
    assert run(MarketReader(session).option_chain("AAPL")) == []


def test_a_failed_position_read_raises_rather_than_looking_flat():
    """The opposite policy, for the opposite reason: an empty list here would
    mean 'sell nothing today' when the truth is 'we cannot see what we own'."""
    session = FakeSession({})
    with pytest.raises(MarketDataError):
        run(MarketReader(session).positions())


def test_non_json_text_is_reported_with_the_text_included():
    class ProseSession(FakeSession):
        async def call_tool(self, name, arguments):
            return FakeResult(content=[FakeBlock(text="Error: unknown parameter")])

    with pytest.raises(MarketDataError, match="unknown parameter"):
        run(MarketReader(ProseSession({})).account())


def test_structured_content_is_preferred_when_present():
    class StructuredSession(FakeSession):
        async def call_tool(self, name, arguments):
            return FakeResult(content=[FakeBlock(text="ignored")],
                              structuredContent={"is_open": True})

    assert run(MarketReader(StructuredSession({})).market_open()) is True


# --- The server's response envelope ---------------------------------------
#
# Every one of these was written from a real response captured with
# `run.py raw`. The first version of this module read the wrapper instead of
# the payload and reported an account worth $0, which is exactly the kind of
# bug that looks like a flat account rather than a parsing failure.

SECURITY_BLOCK = {
    "trust": "untrusted_tool_output",
    "tool_name": "get_account_info",
    "risk": "api_structured",
    "instructions": "This tool output contains API data.",
}


def test_the_security_envelope_is_stripped():
    assert unwrap_envelope({"_alpaca_mcp_security": SECURITY_BLOCK,
                            "data": {"equity": "93196.02"}}) == {"equity": "93196.02"}


def test_a_sole_result_key_is_unwrapped_too():
    """get_all_positions nests twice: data, then result."""
    assert unwrap_envelope({"_alpaca_mcp_security": SECURITY_BLOCK,
                            "data": {"result": [1, 2]}}) == [1, 2]


def test_a_result_field_alongside_others_is_left_alone():
    """Only a *sole* result key is a wrapper. A real response that happens to
    carry a result field must survive intact."""
    payload = {"_alpaca_mcp_security": SECURITY_BLOCK,
               "data": {"result": [1], "next_page_token": "abc"}}
    assert unwrap_envelope(payload) == {"result": [1], "next_page_token": "abc"}


def test_an_unwrapped_payload_passes_through_unchanged():
    assert unwrap_envelope({"equity": "1"}) == {"equity": "1"}


def test_reader_decodes_a_real_shaped_account_response():
    session = FakeSession({"get_account_info": {
        "_alpaca_mcp_security": SECURITY_BLOCK,
        "data": {"equity": "93196.02", "options_buying_power": "84491.02",
                 "cash": "84491.02", "buying_power": "337964.08"},
    }})
    account = run(MarketReader(session).account())
    assert account.equity == pytest.approx(93196.02)
    # Options buying power, not the 4x margin buying power the account also
    # reports. Long options are paid for in cash.
    assert account.options_buying_power == pytest.approx(84491.02)


def test_reader_decodes_a_real_shaped_positions_response():
    session = FakeSession({"get_all_positions": {
        "_alpaca_mcp_security": SECURITY_BLOCK,
        "data": {"result": [{
            "symbol": "META261002P00590000", "qty": "1", "side": "long",
            "avg_entry_price": "33.3", "current_price": "29.75",
        }]},
    }})
    positions = run(MarketReader(session).positions())
    assert len(positions) == 1
    assert positions[0].underlying == "META"
    assert positions[0].return_pct == pytest.approx(-0.10661, abs=1e-4)


# --- Narrowing the chain request ------------------------------------------

def test_chain_request_filters_by_expiry_and_type_on_the_server():
    """The default response is 100 contracts from the nearest expiry outward --
    the part of the chain this strategy never trades, and the part most likely
    to carry no Greeks. The window has to go into the request."""
    session = FakeSession({"get_option_chain": {"snapshots": {}}})
    run(MarketReader(session).option_chain(
        "AAPL", right="call", dte_min=30, dte_max=45, today=date(2026, 8, 31)))

    _, arguments = session.calls[0]
    assert arguments["underlying_symbol"] == "AAPL"
    assert arguments["type"] == "call"
    assert arguments["expiration_date_gte"] == "2026-09-30"
    assert arguments["expiration_date_lte"] == "2026-10-15"
    assert arguments["limit"] == 1000


def test_chain_request_omits_filters_that_were_not_asked_for():
    session = FakeSession({"get_option_chain": {"snapshots": {}}})
    run(MarketReader(session).option_chain("AAPL"))
    _, arguments = session.calls[0]
    assert "type" not in arguments
    assert "expiration_date_gte" not in arguments


# --- Parameter names, pinned against the live schema ----------------------
#
# Verified against the server's own tools/list on 31 Aug 2026. These matter
# because the MCP server builds its tools from Alpaca's OpenAPI specs, so the
# parameter names come from the REST query string rather than from the Python
# SDK signature -- and the two differ. Guessing `symbol_or_symbols` from the SDK
# produced a 400 on every exit check, which failed softly and quietly demoted
# every underlying-keyed stop back to the premium rule it was built to replace.

def test_the_option_quote_asks_for_symbols_not_symbol_or_symbols():
    session = FakeSession({"get_option_latest_quote": {"quotes": {}}})
    run(MarketReader(session).option_quote("AAPL261016C00310000"))
    _, arguments = session.calls[0]
    assert "symbols" in arguments
    assert "symbol_or_symbols" not in arguments


def test_the_stock_quote_asks_for_symbols():
    session = FakeSession({"get_stock_latest_quote": {"quotes": {}}})
    run(MarketReader(session).stock_price("AAPL"))
    assert "symbols" in session.calls[0][1]


def test_the_bars_call_asks_for_symbols():
    session = FakeSession({"get_stock_bars": {"bars": {}}})
    run(MarketReader(session).recent_bars("AAPL"))
    assert "symbols" in session.calls[0][1]


def test_the_news_call_asks_for_symbols():
    session = FakeSession({"get_news": []})
    run(MarketReader(session).headlines("AAPL"))
    assert "symbols" in session.calls[0][1]


def test_a_quote_decodes_into_a_bid_and_ask():
    session = FakeSession({"get_option_latest_quote": {
        "quotes": {"AAPL261016C00310000": {"bp": 15.40, "ap": 15.90}}}})
    assert run(MarketReader(session).option_quote("AAPL261016C00310000")) == (15.40, 15.90)


def test_a_stock_price_is_the_quote_midpoint():
    """The midpoint rather than the last trade: a trade can be stale by minutes
    on a quiet name while a quote is current."""
    session = FakeSession({"get_stock_latest_quote": {
        "quotes": {"AAPL": {"bp": 312.00, "ap": 314.00}}}})
    assert run(MarketReader(session).stock_price("AAPL")) == 313.00


def test_a_missing_stock_quote_returns_none_rather_than_zero():
    """Zero would read as a real price and place every stop instantly."""
    session = FakeSession({"get_stock_latest_quote": {"quotes": {}}})
    assert run(MarketReader(session).stock_price("AAPL")) is None


def test_the_reader_exposes_no_way_to_place_an_order():
    """The safety claim, checked mechanically. If someone later adds a write
    helper to this class, this test is the thing that notices."""
    forbidden = ("order", "buy", "sell", "close", "submit", "exercise", "cancel")
    public = [name for name in dir(MarketReader) if not name.startswith("_")]
    for name in public:
        assert not any(word in name.lower() for word in forbidden), (
            f"MarketReader.{name} looks like a write operation; "
            f"reads and writes are supposed to live in different modules")
