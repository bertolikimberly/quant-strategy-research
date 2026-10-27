"""Unit tests for src.strategies.cointegration_pairs.

We test the building blocks (OU half-life estimator, state machine,
block-bootstrap indexer, Engle-Granger residual extraction) against
synthetic data with known properties. The full Strategy class is
NOT exercised end-to-end here because it pulls real prices via
yfinance; that path is covered by the bake-off smoke run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.cointegration_pairs import (
    _block_bootstrap_indices,
    _engle_granger,
    _ou_half_life,
    _pair_state_series,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2018-01-01", periods=n, freq="B")


# --------------------------------------------------------------------------
# OU half-life
# --------------------------------------------------------------------------


def test_ou_half_life_recovers_known_kappa():
    """Simulate an OU process with kappa=0.05/day and check that
    the estimator recovers a half-life of ln(2)/0.05 ≈ 13.86 days
    within ~20% tolerance on a 3000-sample path."""
    rng = np.random.default_rng(0)
    n = 3000
    kappa = 0.05
    sigma = 0.01
    x = np.zeros(n)
    for t in range(1, n):
        # Discrete OU update with mu=0, dt=1
        x[t] = x[t - 1] - kappa * x[t - 1] + sigma * rng.normal()
    spread = pd.Series(x, index=_idx(n))
    hl = _ou_half_life(spread)
    expected = np.log(2) / kappa  # ≈ 13.86
    # Allow ±20% — half-life estimators have meaningful sampling noise.
    assert hl == pytest.approx(expected, rel=0.20)


def test_ou_half_life_returns_inf_for_random_walk():
    """A pure random walk has no mean reversion (kappa=0)."""
    rng = np.random.default_rng(1)
    rw = np.cumsum(rng.normal(0, 0.01, 2000))
    spread = pd.Series(rw, index=_idx(2000))
    hl = _ou_half_life(spread)
    # A random walk's regression coefficient is near zero → lambda
    # near zero → half-life enormous. We accept either a very large
    # finite half-life or infinity (both signal "fails the filter").
    assert hl > 100  # any reasonable filter would reject this


def test_ou_half_life_returns_inf_for_short_series():
    """The function should not crash on too-short input — it should
    return infinity (i.e. "rejectable")."""
    short = pd.Series([1.0, 2.0, 3.0], index=_idx(3))
    assert _ou_half_life(short) == float("inf")


# --------------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------------


def test_state_machine_basic_entries_and_exits():
    """Walk a hand-crafted z-score path through the state machine and
    check the transitions hit at exactly the right rows."""
    # Build a path that exercises every transition:
    #   start flat, drop below -2 → long-spread (+1)
    #   recover to 0 → exit to flat
    #   spike above +2 → short-spread (-1)
    #   crash to -3.5 → stop out to flat (we're short and z plunged)
    z_path = [0.0, -1.0, -2.5, -1.0, 0.5, 1.5, 2.5, 1.0, -3.5, 0.0]
    z = pd.Series(z_path, index=_idx(len(z_path)))
    state = _pair_state_series(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    # Expected, row by row:
    #  i=0 (z=0):     flat, no entry → 0
    #  i=1 (z=-1):    flat, still no entry → 0
    #  i=2 (z=-2.5):  flat, z < -entry → enter long (+1)
    #  i=3 (z=-1):    +1, z < exit_z=0, not hit stop yet → stay +1
    #  i=4 (z=0.5):   +1, z >= exit_z → exit → 0
    #  i=5 (z=1.5):   flat, |z| < entry → 0
    #  i=6 (z=2.5):   flat, z > entry → enter short (-1)
    #  i=7 (z=1.0):   -1, z > -exit_z=0 ... wait, the rule is "exit when z <= -exit_z"
    #                 With exit_z=0: exit short when z <= 0. z=1.0 > 0 → stay -1.
    #  i=8 (z=-3.5):  -1, z <= -exit_z=0 → exit → 0
    #                 (note: also passes through z<-stop=-3, but exit fires first)
    #  i=9 (z=0):     flat → 0
    expected = [0, 0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert state.tolist() == [float(x) for x in expected]


def test_state_machine_stops_out_when_long_spread_blows_through():
    # Start by entering long-spread; then z plunges below -stop.
    z_path = [-2.5, -2.8, -3.5, -2.0, 0.0]
    z = pd.Series(z_path, index=_idx(5))
    state = _pair_state_series(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    # i=0: enter long (+1)
    # i=1: still long, |z| < stop → +1
    # i=2: z < -stop → stop out → 0
    # i=3: flat, |z| < entry → 0
    # i=4: flat → 0
    assert state.tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]


def test_state_machine_handles_nan_by_going_flat():
    z = pd.Series([1.0, np.nan, -2.5, np.nan, -1.0], index=_idx(5))
    state = _pair_state_series(z, entry_z=2.0, exit_z=0.0, stop_z=3.0)
    # Any NaN forces flat (per the docstring contract).
    # i=0 (z=1):    flat → 0
    # i=1 (NaN):    flat → 0
    # i=2 (z=-2.5): flat → enter long → +1
    # i=3 (NaN):    flat (state reset) → 0
    # i=4 (z=-1):   flat, |z|<entry → 0
    assert state.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# Block bootstrap indices
# --------------------------------------------------------------------------


def test_block_bootstrap_indices_length_and_range():
    rng = np.random.default_rng(7)
    idx = _block_bootstrap_indices(n=100, block_size=10, rng=rng)
    assert len(idx) == 100
    # Every index must be a valid offset into the original series.
    assert idx.min() >= 0 and idx.max() < 100


def test_block_bootstrap_preserves_local_structure():
    """Within each block, indices should be consecutive (mod n).

    A simple way to verify: many adjacent index-differences should be
    exactly +1 (the in-block step), with occasional jumps at block
    boundaries. We assert at least 80% of step-differences are +1
    (mod n) — far above what random sampling without blocks would give."""
    rng = np.random.default_rng(11)
    n = 500
    block_size = 25
    idx = _block_bootstrap_indices(n=n, block_size=block_size, rng=rng)
    diffs = np.diff(idx) % n
    frac_step_plus_one = (diffs == 1).mean()
    # Random uniform indices would give 1/n ≈ 0.002 frac. Block
    # bootstrap with block_size=25 gives at least (24/25) = 0.96
    # in-block, minus a small boundary effect → expect >0.9.
    assert frac_step_plus_one > 0.85


# --------------------------------------------------------------------------
# Engle-Granger residuals
# --------------------------------------------------------------------------


def test_engle_granger_recovers_known_beta_on_cointegrated_series():
    """Build two log-price series with a known cointegrating
    relationship and confirm the regression recovers the right beta
    and produces a low p-value."""
    rng = np.random.default_rng(3)
    n = 1500
    # Common stochastic trend
    trend = np.cumsum(rng.normal(0, 0.01, n))
    # Series B is the trend with noise; series A is 1 + 0.8*B + stationary noise.
    log_b = trend + rng.normal(0, 0.005, n)
    # Stationary noise (small-AR(1))
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.5 * noise[t - 1] + 0.005 * rng.normal()
    log_a = 1.0 + 0.8 * log_b + noise

    s_a = pd.Series(log_a, index=_idx(n))
    s_b = pd.Series(log_b, index=_idx(n))
    beta, pvalue, residuals = _engle_granger(s_a, s_b)

    # Beta should be close to 0.8 (the construction coefficient).
    assert beta == pytest.approx(0.8, abs=0.05)
    # Cointegration test should fire a strong rejection.
    assert pvalue < 0.05
    # Residuals come back as a Series indexed like log_a.
    assert isinstance(residuals, pd.Series)
    assert residuals.index.equals(s_a.index)


def test_engle_granger_pvalue_large_for_independent_walks():
    """Two independent random walks are NOT cointegrated → high p."""
    rng = np.random.default_rng(13)
    n = 1500
    a = np.cumsum(rng.normal(0, 0.01, n))
    b = np.cumsum(rng.normal(0, 0.01, n))
    s_a = pd.Series(a, index=_idx(n))
    s_b = pd.Series(b, index=_idx(n))
    _, pvalue, _ = _engle_granger(s_a, s_b)
    # With independent walks, the EG p-value should NOT be small.
    # We allow up to 0.5 (a single draw can land anywhere under
    # the null) but reject anything that looks like significance.
    assert pvalue > 0.10
