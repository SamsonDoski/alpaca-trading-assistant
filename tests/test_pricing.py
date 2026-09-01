"""Tests for the option pricing arithmetic.

Pure functions over numbers, so these are exact rather than approximate, and
they encode the two claims the strategy now depends on: that we can tell an
expensive premium from a cheap one, and that we can see the clock.
"""

from __future__ import annotations

from datetime import date

import pytest

from agent.domain import OptionContract, PriceBar
from agent.pricing import (
    average_true_range_pct,
    black_scholes_theta,
    daily_decay_pct,
    decay_to_target_pct,
    premium_richness,
    realized_volatility,
    underlying_move_for,
)

TODAY = date(2026, 9, 4)
EXPIRY = date(2026, 10, 16)      # 42 days out


def bars(*closes: float) -> tuple[PriceBar, ...]:
    return tuple(
        PriceBar(day=date(2026, 8, 1), open=c, high=c + 2, low=c - 2, close=c,
                 volume=1_000_000)
        for c in closes
    )


def alternating(low: float = 310.0, high: float = 313.0, count: int = 25):
    return bars(*[(high if i % 2 else low) for i in range(count)])


def contract(**overrides) -> OptionContract:
    defaults = dict(
        occ_symbol="AAPL261016C00310000", underlying="AAPL", right="call",
        strike=310.0, expiry=EXPIRY, bid=15.50, ask=16.00, delta=0.65,
        implied_volatility=0.20, open_interest=900,
    )
    return OptionContract(**{**defaults, **overrides})


# --- Realized volatility ---------------------------------------------------

def test_a_steady_alternating_series_has_a_measurable_volatility():
    rv = realized_volatility(alternating(), lookback=20)
    assert rv is not None
    assert 0.14 < rv < 0.18


def test_a_flat_series_has_no_volatility():
    rv = realized_volatility(bars(*[100.0] * 25), lookback=20)
    assert rv == pytest.approx(0.0)


def test_a_wilder_series_measures_higher():
    calm = realized_volatility(alternating(310.0, 311.0), lookback=20)
    wild = realized_volatility(alternating(310.0, 330.0), lookback=20)
    assert wild > calm * 3


def test_too_little_history_is_none_not_zero():
    """Same rule the trend calculations follow: a value meaning 'we could not
    measure this' must not be spelled like a measurement."""
    assert realized_volatility(alternating(count=5), lookback=20) is None


def test_a_zero_close_is_refused_rather_than_logged():
    assert realized_volatility(bars(*([100.0] * 20 + [0.0])), lookback=20) is None


# --- Premium richness ------------------------------------------------------

def test_implied_equal_to_realized_is_fairly_priced():
    assert premium_richness(0.25, 0.25) == pytest.approx(1.0)


def test_implied_above_realized_is_rich():
    assert premium_richness(0.40, 0.20) == pytest.approx(2.0)


def test_implied_below_realized_is_cheap():
    """The favourable side of the variance risk premium for a buyer."""
    assert premium_richness(0.15, 0.30) == pytest.approx(0.5)


def test_missing_either_volatility_gives_no_answer():
    assert premium_richness(None, 0.2) is None
    assert premium_richness(0.2, None) is None
    assert premium_richness(0.2, 0.0) is None


# --- Theta -----------------------------------------------------------------

def test_theta_is_negative_because_long_options_bleed():
    theta = black_scholes_theta(313.0, 310.0, 0.20, 42, "call")
    assert theta is not None and theta < 0


def test_decay_accelerates_as_expiry_approaches():
    """The reason close_before_expiry exists at all."""
    far = black_scholes_theta(313.0, 310.0, 0.20, 60, "call")
    near = black_scholes_theta(313.0, 310.0, 0.20, 7, "call")
    assert abs(near) > abs(far)


def test_higher_implied_volatility_means_faster_decay():
    calm = black_scholes_theta(313.0, 310.0, 0.15, 42, "call")
    wild = black_scholes_theta(313.0, 310.0, 0.60, 42, "call")
    assert abs(wild) > abs(calm)


def test_an_expired_contract_has_no_theta():
    assert black_scholes_theta(313.0, 310.0, 0.20, 0, "call") is None


def test_nonsense_inputs_give_no_answer():
    assert black_scholes_theta(0.0, 310.0, 0.20, 42, "call") is None
    assert black_scholes_theta(313.0, 310.0, 0.0, 42, "call") is None
    assert black_scholes_theta(313.0, 310.0, 0.20, 42, "future") is None


def test_daily_decay_is_a_positive_fraction_of_the_premium():
    decay = daily_decay_pct(contract(), 313.0, TODAY)
    assert decay is not None
    assert 0.0 < decay < 0.02


def test_a_contract_with_no_implied_volatility_cannot_be_assessed():
    assert daily_decay_pct(contract(implied_volatility=None), 313.0, TODAY) is None


def test_the_two_week_projection_is_ten_days_of_decay():
    daily = daily_decay_pct(contract(), 313.0, TODAY)
    projected = decay_to_target_pct(contract(), 313.0, TODAY, holding_days=10)
    assert projected == pytest.approx(daily * 10)


# --- Translating premium moves into underlying moves -----------------------

def test_a_premium_stop_is_a_much_smaller_move_in_the_stock():
    """The whole reason to key stops to the underlying. A 25% fall in a $15.75
    option at 0.65 delta is under two percent of a $313 stock -- comfortably
    inside a normal day's noise."""
    move = underlying_move_for(contract(), 313.0, -0.25)
    assert move is not None
    assert -0.03 < move < -0.015


def test_a_put_gains_when_the_underlying_falls():
    call_move = underlying_move_for(contract(right="call", delta=0.65), 313.0, 0.25)
    put_move = underlying_move_for(
        contract(right="put", delta=-0.65, occ_symbol="AAPL261016P00310000"),
        313.0, 0.25)
    assert call_move > 0 and put_move < 0


def test_a_higher_delta_needs_a_smaller_move():
    low = underlying_move_for(contract(delta=0.30), 313.0, -0.25)
    high = underlying_move_for(contract(delta=0.90), 313.0, -0.25)
    assert abs(high) < abs(low)


def test_a_contract_with_no_delta_cannot_be_translated():
    assert underlying_move_for(contract(delta=None), 313.0, -0.25) is None


# --- Average true range ----------------------------------------------------

def test_average_true_range_measures_a_typical_day():
    atr = average_true_range_pct(alternating(), lookback=14)
    assert atr is not None and atr > 0


def test_a_quiet_stock_has_a_smaller_range_than_a_wild_one():
    calm = average_true_range_pct(alternating(310.0, 310.5), lookback=14)
    wild = average_true_range_pct(alternating(310.0, 340.0), lookback=14)
    assert wild > calm


def test_too_little_history_gives_no_range():
    assert average_true_range_pct(alternating(count=5), lookback=14) is None
