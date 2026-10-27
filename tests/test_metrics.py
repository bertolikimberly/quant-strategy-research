"""Unit tests for src.metrics.scorecard.

We feed in carefully constructed synthetic series so each metric's
expected value is hand-computable. If a future refactor breaks the
formula, the test pins the number; if it changes the API, the test
breaks loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics.scorecard import (
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    hit_rate,
    max_drawdown,
    newey_west_tstat,
    scorecard,
    sharpe,
    sortino,
    turnover,
)


def _daily_series(values: list[float]) -> pd.Series:
    """Helper: build a Series with a daily business-day index."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_annualized_return_constant():
    # 1bp per day, 252 days/yr → 252 bp annualised = 0.0252.
    r = _daily_series([0.0001] * 500)
    assert annualized_return(r) == pytest.approx(0.0001 * TRADING_DAYS_PER_YEAR)


def test_annualized_vol_constant_is_zero():
    # Constant return series has zero stdev → zero annualised vol.
    r = _daily_series([0.001] * 100)
    assert annualized_volatility(r) == pytest.approx(0.0)


def test_sharpe_on_known_series():
    # Construct returns alternating +x / -x. Mean = 0, so Sharpe = 0.
    r = _daily_series([0.01, -0.01] * 100)
    assert sharpe(r) == pytest.approx(0.0)


def test_sharpe_positive_for_positive_drift():
    # All-positive returns with small noise → positive Sharpe.
    rng = np.random.default_rng(0)
    r = _daily_series((0.001 + rng.normal(0, 0.0005, 500)).tolist())
    assert sharpe(r) > 1.0  # very stable → high Sharpe


def test_sortino_only_penalises_downside():
    # Symmetric returns: Sortino ≈ Sharpe up to a constant factor
    # (downside dev ≈ stdev / sqrt(2) for a zero-mean Gaussian, so
    # Sortino ≈ sqrt(2) * Sharpe for symmetric data with mean 0).
    # We use an asymmetric series instead: all gains are large,
    # all losses are tiny. Sortino should be much larger than Sharpe.
    r = _daily_series([0.02 if i % 2 == 0 else -0.001 for i in range(200)])
    assert sortino(r) > sharpe(r) * 2  # generous bound; the truth is ~5-10x


def test_max_drawdown_simple():
    # Series goes 1.0 -> 1.10 -> 0.99 (10% gain, then ~10% loss).
    # Drawdown = 0.99/1.10 - 1 = -0.10
    daily_returns = [0.10, -0.10]
    r = _daily_series(daily_returns)
    expected = (1.10 * 0.90) / 1.10 - 1  # ≈ -0.10 (with one rounding error)
    assert max_drawdown(r) == pytest.approx(expected, abs=1e-9)


def test_max_drawdown_never_positive():
    # Even a monotone-rising series must produce dd <= 0 (no drawdown).
    rng = np.random.default_rng(1)
    r = _daily_series((np.abs(rng.normal(0, 0.01, 200)) + 0.001).tolist())
    assert max_drawdown(r) <= 0.0


def test_hit_rate_counts_strictly_positive_only():
    # 3 wins, 2 losses, 1 zero → 3/6 = 0.5 hit rate.
    r = _daily_series([0.01, -0.01, 0.02, -0.02, 0.0, 0.005])
    assert hit_rate(r) == pytest.approx(0.5)


def test_turnover_buy_and_hold_is_minimal():
    # Always +1.0: one initial trade (delta = 1 on row 0), zero
    # changes thereafter. Mean |Δposition| = 1/N → annualised = 252/N.
    pos = _daily_series([1.0] * 252)
    assert turnover(pos) == pytest.approx(TRADING_DAYS_PER_YEAR / 252)


def test_turnover_flip_every_day():
    # Alternating +1/-1: |Δ| = 2 every day after the first.
    # First-day delta is forced to the initial position (1.0).
    # So mean |Δ| ≈ 2 * (N-1)/N + 1*1/N → ~2.
    pos = _daily_series([1.0 if i % 2 == 0 else -1.0 for i in range(252)])
    assert turnover(pos) == pytest.approx(2 * TRADING_DAYS_PER_YEAR, rel=0.01)


def test_newey_west_tstat_zero_mean_series():
    # Mean ≈ 0 → t-stat near 0.
    rng = np.random.default_rng(2)
    r = _daily_series(rng.normal(0, 0.01, 1000).tolist())
    assert abs(newey_west_tstat(r)) < 3.0  # ~95% of random draws satisfy this


def test_newey_west_tstat_clearly_positive_mean():
    # Constant + tiny noise → very strong evidence mean > 0.
    rng = np.random.default_rng(3)
    r = _daily_series((0.001 + rng.normal(0, 0.0001, 1000)).tolist())
    assert newey_west_tstat(r) > 10.0  # massive t


def test_scorecard_dict_has_stable_keys():
    r = _daily_series([0.001] * 100)
    pos = _daily_series([1.0] * 100)
    row = scorecard(r, pos, label="test")
    expected_keys = {
        "label", "n_obs", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "hit_rate", "turnover", "nw_tstat",
    }
    assert expected_keys.issubset(row.keys())
