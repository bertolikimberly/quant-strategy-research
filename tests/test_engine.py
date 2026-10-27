"""Unit tests for src.engine.backtest.

These tests build synthetic strategies + price series so we can predict
the engine's output exactly and pin it. They are the most important
tests in the framework: the engine is the one piece of shared infra
every strategy goes through, so a bug here corrupts every bake-off
row simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import run_backtest
from src.engine.costs import CostModel
from src.strategies.base import Strategy


class _StaticStrategy(Strategy):
    """Test helper: hand it prices and positions, get them back through the engine.

    Both `data` and `positions` are provided up-front so the test
    knows EXACTLY what should come out of the engine without having
    to mock yfinance or compute moving averages.
    """

    def __init__(self, prices: pd.Series, positions: pd.Series):
        self._data = pd.DataFrame({"close": prices})
        self._positions = positions.copy()
        self.name = "static_test"

    def prepare_data(self, start: str, end: str) -> pd.DataFrame:
        return self._data

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        return self._positions


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_lookahead_protection_no_perfect_foresight():
    """A strategy with PERFECT foresight should still earn nothing if
    the engine shifts positions properly.

    Construction: positions[T] = sign(asset_return[T]). With the engine's
    .shift(1), position USED on day T is sign(asset_return[T-1]) — i.e.
    yesterday's known return. For iid mean-zero returns, the realised
    PnL has expectation zero.

    If the engine accidentally DIDN'T shift, this same setup would
    earn ~E[|r|] per day (the strategy always knows today's direction),
    which for vol=0.01 is ~0.008. So this is a direct regression test
    on the .shift(1): we should be near zero, NOT near 0.008.

    Two important details:
      - We use enough samples (N=3000) that the sample mean's standard
        error is small: SE ≈ vol/sqrt(N) ≈ 0.01/55 ≈ 1.8e-4.
      - We compare against the no-shift baseline (~0.008) so the test
        fails loudly if .shift(1) is ever removed, but doesn't trip on
        finite-sample noise. The bound is 0.001 (~6x larger than SE,
        ~8x smaller than the no-shift outcome).
    """
    n = 3000
    rng = np.random.default_rng(42)
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=_idx(n)
    )
    asset_returns = prices.pct_change().fillna(0)
    positions = pd.Series(np.sign(asset_returns), index=_idx(n))

    strat = _StaticStrategy(prices, positions)
    # Use a zero cost model so we measure ONLY the timing effect.
    zero_costs = CostModel(0, 0, 0, 0)
    result = run_backtest(strat, "x", "y", zero_costs)

    # If the shift is in place, mean net return should be small.
    # The no-shift baseline would be ~0.008, so 0.001 cleanly
    # distinguishes "shift works" from "shift is missing".
    assert abs(result.net_returns.mean()) < 0.001

    # Sanity: if there were NO shift, the mean would be ~|return| ≈
    # 0.008 — well above 0.001. So this test cleanly catches a missing
    # shift.


def test_engine_charges_initial_trade():
    """Establishing a +1.0 position from zero must incur turnover cost.

    Build a flat-price series with position = +1 always. Asset return
    is exactly 0 every day, so gross return is 0 → any net loss must
    come from costs. With per_unit_trade_cost = 10 bp and a one-time
    full position (turnover[T0] = 1.0), total cost should be 10 bp.
    """
    n = 50
    prices = pd.Series(100.0, index=_idx(n))  # flat
    positions = pd.Series(1.0, index=_idx(n))

    strat = _StaticStrategy(prices, positions)
    # 10 bp / unit traded total. Zero financing for a long-only book.
    costs = CostModel(
        half_spread_bps=5, commission_bps=3, slippage_bps=2,
        financing_rate_annual=0.0,
    )
    result = run_backtest(strat, "x", "y", costs)

    # Exactly one trade of size 1.0 → 10 bp total cost, paid on day 1
    # (since engine shifts trade cost by one day to align with PnL).
    total_cost = result.trade_costs.sum()
    assert total_cost == pytest.approx(0.0010, abs=1e-9)
    # And gross return must be zero everywhere (flat asset).
    assert (result.gross_returns == 0).all()


def test_engine_charges_financing_on_leverage():
    """A 2x-long position pays daily financing on the 1x leveraged portion."""
    n = 100
    prices = pd.Series(100.0, index=_idx(n))  # flat asset
    positions = pd.Series(2.0, index=_idx(n))  # 2x leveraged long

    strat = _StaticStrategy(prices, positions)
    costs = CostModel(
        half_spread_bps=0, commission_bps=0, slippage_bps=0,
        financing_rate_annual=0.0252,  # exactly 1bp/day @ 252 days
    )
    result = run_backtest(strat, "x", "y", costs)

    # After the initial day's lag, the applied (lagged) position is 2.0
    # → leveraged_notional = (2-1) + 0 (short part = 0) = 1.0
    # → daily financing = 1.0 * 1bp = 0.0001 per day.
    # Day 0 has applied_position=0 (because of the shift fillna(0)),
    # so financing on day 0 is 0. Days 1..n-1 each pay 0.0001.
    expected = 0.0001 * (n - 1)
    assert result.financing_costs.sum() == pytest.approx(expected, abs=1e-9)


def test_engine_charges_financing_on_short():
    """A -1.0 (full short) pays daily borrow on the |1| short notional."""
    n = 60
    prices = pd.Series(100.0, index=_idx(n))
    positions = pd.Series(-1.0, index=_idx(n))

    strat = _StaticStrategy(prices, positions)
    costs = CostModel(
        half_spread_bps=0, commission_bps=0, slippage_bps=0,
        financing_rate_annual=0.0252,
    )
    result = run_backtest(strat, "x", "y", costs)

    # After lag, applied_position = -1, leveraged_notional = 0 + 1 = 1
    # → 1bp/day * (n-1) days = expected.
    expected = 0.0001 * (n - 1)
    assert result.financing_costs.sum() == pytest.approx(expected, abs=1e-9)


def test_unlevered_long_pays_no_financing():
    """A +1.0 long pays zero financing — we own the asset outright."""
    n = 50
    prices = pd.Series(100.0, index=_idx(n))
    positions = pd.Series(1.0, index=_idx(n))

    strat = _StaticStrategy(prices, positions)
    costs = CostModel(0, 0, 0, financing_rate_annual=0.10)
    result = run_backtest(strat, "x", "y", costs)

    assert (result.financing_costs == 0).all()


def test_position_index_mismatch_raises():
    """Engine must refuse to run if positions and data indices disagree."""

    class BadStrategy(Strategy):
        name = "bad"
        def prepare_data(self, start, end):
            return pd.DataFrame({"close": [100, 101, 102]}, index=_idx(3))
        def generate_positions(self, data):
            # Wrong index length on purpose.
            return pd.Series([1, 1], index=_idx(2))

    with pytest.raises(ValueError, match="index"):
        run_backtest(BadStrategy(), "x", "y")


def test_missing_close_column_raises():
    """Engine must refuse data without a 'close' column."""

    class BadStrategy(Strategy):
        name = "bad"
        def prepare_data(self, start, end):
            return pd.DataFrame({"open": [100, 101]}, index=_idx(2))
        def generate_positions(self, data):
            return pd.Series([1, 1], index=_idx(2), dtype=float)

    with pytest.raises(ValueError, match="close"):
        run_backtest(BadStrategy(), "x", "y")


def test_equity_curve_starts_at_initial():
    n = 10
    prices = pd.Series(100.0, index=_idx(n))
    positions = pd.Series(0.0, index=_idx(n))  # never trade
    strat = _StaticStrategy(prices, positions)
    result = run_backtest(strat, "x", "y", CostModel(0, 0, 0, 0))
    eq = result.equity_curve(initial=1.0)
    # Zero positions + zero costs → equity stays at 1.0.
    assert (eq == 1.0).all()
