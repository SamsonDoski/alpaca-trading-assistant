"""Tests for marking the model's views against what the underlying then did.

The resolver decides win rates for the whole evaluation, so the cases that
matter are the ambiguous ones: a day that touches both levels, a stop touched
intraday but recovered by the close, and a view too recent to have finished.
"""

from __future__ import annotations

from datetime import date, timedelta

from agent.domain import PriceBar
from agent.evaluation import build, resolve, summarise
from agent.settings import Settings

SETTINGS = Settings(stop_atr_multiple=2.0, reward_to_risk=2.0, fallback_stop_pct=0.03)
START = date(2026, 9, 1)


def bar(day_offset, *, high, low, close, open_=None):
    return PriceBar(day=START + timedelta(days=day_offset),
                    open=open_ if open_ is not None else close,
                    high=high, low=low, close=close, volume=1_000_000)


def history(close=100.0, count=30):
    """Flat-ish history so the ATR, and therefore the stop, is predictable."""
    return [bar(i, high=close + 1, low=close - 1, close=close) for i in range(count)]


def a_trade(direction="up", **kwargs):
    return build(day="2026-09-30", underlying="TEST", direction=direction,
                 confidence=0.65, traded=False, reason="test",
                 history=history(**kwargs), settings=SETTINGS)


def test_levels_come_from_what_was_knowable_on_the_day():
    trade = a_trade()
    assert trade.entry_spot == 100.0
    assert trade.stop_spot < 100.0 < trade.target_spot
    # reward-to-risk 2.0: the target must sit twice as far away as the stop
    assert (trade.target_spot - 100.0) == (100.0 - trade.stop_spot) * 2


def test_a_view_with_no_history_cannot_be_marked():
    assert build(day="2026-09-30", underlying="TEST", direction="up",
                 confidence=0.65, traded=False, reason="test",
                 history=[], settings=SETTINGS) is None


def test_reaching_the_target_first_is_a_win():
    trade = a_trade()
    forward = [bar(31, high=trade.target_spot + 1, low=99.0, close=trade.target_spot)]
    result = resolve(trade, forward, max_days=45)
    assert result.outcome == "target" and result.won and result.days_held == 1


def test_touching_the_stop_first_is_a_loss():
    trade = a_trade()
    forward = [bar(31, high=101.0, low=trade.stop_spot - 1, close=99.0)]
    result = resolve(trade, forward, max_days=45)
    assert result.outcome == "stop" and not result.won


def test_a_stop_touched_intraday_counts_even_if_the_close_recovers():
    """A stop is a level the price touches, not one it finishes at. Judging on
    closes would quietly turn every intraday stop-out into a position still
    running."""
    trade = a_trade()
    forward = [bar(31, high=101.0, low=trade.stop_spot - 0.5, close=100.5)]
    assert resolve(trade, forward, max_days=45).outcome == "stop"


def test_a_day_that_touches_both_levels_counts_as_a_stop():
    """Daily bars cannot say which came first, and assuming the good outcome
    would flatter every volatile name in the sample."""
    trade = a_trade()
    forward = [bar(31, high=trade.target_spot + 1, low=trade.stop_spot - 1,
                   close=100.0)]
    assert resolve(trade, forward, max_days=45).outcome == "stop"


def test_a_view_that_reaches_neither_level_runs_out():
    trade = a_trade()
    forward = [bar(31 + i, high=100.5, low=99.5, close=100.0) for i in range(45)]
    result = resolve(trade, forward, max_days=45)
    assert result.outcome == "expired" and not result.won and result.decided


def test_a_view_too_recent_to_have_finished_is_still_open():
    trade = a_trade()
    forward = [bar(31 + i, high=100.5, low=99.5, close=100.0) for i in range(3)]
    result = resolve(trade, forward, max_days=45)
    assert result.outcome == "open" and not result.decided


def test_a_down_view_wins_when_the_stock_falls():
    trade = a_trade(direction="down")
    assert trade.target_spot < 100.0 < trade.stop_spot
    forward = [bar(31, high=100.5, low=trade.target_spot - 1, close=trade.target_spot)]
    result = resolve(trade, forward, max_days=45)
    assert result.outcome == "target" and result.move_pct > 0


def test_a_down_view_loses_when_the_stock_rises():
    trade = a_trade(direction="down")
    forward = [bar(31, high=trade.stop_spot + 1, low=99.0, close=trade.stop_spot)]
    result = resolve(trade, forward, max_days=45)
    assert result.outcome == "stop" and result.move_pct < 0


def test_unfinished_views_are_left_out_of_the_win_rate():
    """An open view is not a loss, and counting it as one would make every
    recent day look worse than it was."""
    trade = a_trade()
    settled = resolve(trade, [bar(31, high=trade.target_spot + 1, low=99.0,
                                  close=trade.target_spot)], max_days=45)
    running = resolve(trade, [bar(31, high=100.5, low=99.5, close=100.0)], max_days=45)

    stats = summarise([settled, running])
    assert stats["views"] == 2
    assert stats["decided"] == 1
    assert stats["win_rate"] == 1.0
    assert stats["still_open"] == 1
