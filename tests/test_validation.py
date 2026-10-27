"""Unit tests for src.validation.split and src.validation.permutation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.engine.backtest import run_backtest
from src.engine.costs import CostModel
from src.strategies.base import Strategy
from src.validation.permutation import permutation_test
from src.validation.split import train_test_split


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


def test_split_partitions_perfectly():
    s = pd.Series(range(100), index=_idx(100))
    sp = train_test_split(s, train_ratio=0.7)
    assert len(sp.in_sample) == 70
    assert len(sp.out_of_sample) == 30
    # Concatenating the two should reconstruct the original.
    rebuilt = pd.concat([sp.in_sample, sp.out_of_sample])
    pd.testing.assert_series_equal(rebuilt, s)


def test_split_preserves_chronology():
    s = pd.Series(range(50), index=_idx(50))
    sp = train_test_split(s, train_ratio=0.5)
    # Every IS index must be strictly earlier than every OOS index.
    assert sp.in_sample.index.max() < sp.out_of_sample.index.min()


def test_split_rejects_silly_ratios():
    s = pd.Series(range(10), index=_idx(10))
    with pytest.raises(ValueError):
        train_test_split(s, train_ratio=0.99)
    with pytest.raises(ValueError):
        train_test_split(s, train_ratio=0.05)


def test_split_handles_empty_series():
    s = pd.Series([], dtype=float)
    sp = train_test_split(s, train_ratio=0.7)
    assert sp.in_sample.empty and sp.out_of_sample.empty


# --------------------------------------------------------------------------
# Permutation test
# --------------------------------------------------------------------------


class _StaticStrategy(Strategy):
    """Same helper as test_engine.py — re-declared here so tests are self-contained."""

    def __init__(self, prices: pd.Series, positions: pd.Series, name: str = "static"):
        self._data = pd.DataFrame({"close": prices})
        self._positions = positions.copy()
        self.name = name

    def prepare_data(self, start, end):
        return self._data

    def generate_positions(self, data):
        return self._positions


def test_permutation_real_skill_gets_low_pvalue():
    """A strategy that genuinely knows yesterday's return direction
    should beat its own shuffled null often enough that the p-value
    is small.

    Construction: build prices with POSITIVE serial correlation
    (lag-1 autocorr ≈ 0.3). A strategy that holds position = sign of
    yesterday's return then has true edge. After the engine's
    .shift(1), its effective signal is sign(return at T-2) applied
    against return at T — still positively correlated thanks to the
    autocorrelation. Permuting positions destroys that correlation.
    """
    rng = np.random.default_rng(0)
    n = 1500
    # AR(1) returns with strong positive autocorrelation
    rho = 0.3
    eps = rng.normal(0, 0.01, n)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = rho * returns[i - 1] + eps[i]
    prices = pd.Series(100 * np.cumprod(1 + returns), index=_idx(n))

    # Strategy: position[T] = sign of return at T-1 (uses past info only)
    asset_returns = prices.pct_change().fillna(0)
    positions = pd.Series(
        np.sign(asset_returns.shift(1).fillna(0)),
        index=_idx(n),
    )

    strat = _StaticStrategy(prices, positions, name="ar1_lagged_sign")
    perm = permutation_test(
        strat, "x", "y",
        n_permutations=500,
        cost_model=CostModel(0, 0, 0, 0),  # ignore costs in this test
        seed=1,
    )

    # Real strategy beats the shuffled null comfortably.
    assert perm.observed_sharpe > 0.5
    # Null distribution should center near zero — the shuffled signal
    # has no edge. Mean exactly zero would be too strong; just sanity.
    assert abs(perm.null_sharpes.mean()) < 0.5
    # p-value below 0.10 is enough; with rho=0.3 and N=500 perms it's
    # typically much smaller.
    assert perm.p_value < 0.10


def test_permutation_no_skill_yields_high_pvalue():
    """A genuinely random strategy should NOT beat its shuffled null
    systematically — its p-value should be roughly uniform.

    We don't assert p > 0.5 because that's not statistically
    guaranteed in one draw; we assert p > 0.05, which holds for
    ~95% of random draws under the null.
    """
    rng = np.random.default_rng(7)
    n = 1000
    prices = pd.Series(
        100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=_idx(n)
    )
    # Random positions in {-1, 0, +1}, uncorrelated with anything.
    positions = pd.Series(
        rng.choice([-1, 0, 1], size=n, p=[0.3, 0.4, 0.3]),
        index=_idx(n),
        dtype=float,
    )

    strat = _StaticStrategy(prices, positions, name="random_positions")
    perm = permutation_test(
        strat, "x", "y",
        n_permutations=500,
        cost_model=CostModel(0, 0, 0, 0),
        seed=11,
    )

    # A random strategy's observed Sharpe is itself a draw from
    # roughly the same null distribution we then build. p-value
    # should be > 0.05 about 95% of the time. With our fixed RNG
    # seeds this is deterministic — adjust the seed if a future
    # numpy version changes RNG output and this assertion flips.
    assert perm.p_value > 0.05
