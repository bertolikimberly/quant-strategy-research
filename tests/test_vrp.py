"""Unit tests for src.strategies.vrp_term_structure.

We exercise the signal helper (_vrp_signal) on synthetic VIX/VXV
series with hand-computable expected outputs. The full Strategy
class is NOT exercised end-to-end here — that goes through yfinance,
which we cover via the bake-off smoke run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.vrp_term_structure import _vrp_signal


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="B")


# --------------------------------------------------------------------------
# Regime classification
# --------------------------------------------------------------------------


def test_signal_long_in_steady_contango_no_smoothing():
    """Pure contango regime (VIX < VXV) with smoothing=1 should produce
    position = 1 every day from row 2 onward.

    Row 0: NaN (no T-1 data yet) → 0
    Row 1+: ratio[T-1] = 0.5 < 1.0 → position = 1
    """
    short = pd.Series([10.0] * 20, index=_idx(20))
    long_ = pd.Series([20.0] * 20, index=_idx(20))
    pos = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=1)
    # Row 0 has no T-1 ratio available → flat by NaN-comparison default.
    assert pos.iloc[0] == 0.0
    # Rows 1..19 all see ratio[T-1]=0.5 → long.
    assert (pos.iloc[1:] == 1.0).all()


def test_signal_flat_in_steady_backwardation_no_smoothing():
    """Pure backwardation (VIX > VXV) → position = 0 throughout."""
    short = pd.Series([30.0] * 20, index=_idx(20))
    long_ = pd.Series([20.0] * 20, index=_idx(20))
    pos = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=1)
    assert (pos == 0.0).all()


def test_signal_warmup_is_flat_with_smoothing():
    """With signal_smoothing_days=5, the first 5 rows should be flat
    (smoothed rolling mean is NaN; (NaN < threshold) → False → 0)."""
    short = pd.Series([10.0] * 20, index=_idx(20))
    long_ = pd.Series([20.0] * 20, index=_idx(20))
    pos = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=5)
    # Smoothed is NaN for rows 0..3; valid from row 4.
    # Then .shift(1) means positions are NaN-comparison (→0) until
    # row 5. Rows 5..19 should be 1.
    assert (pos.iloc[:5] == 0.0).all()
    assert (pos.iloc[5:] == 1.0).all()


# --------------------------------------------------------------------------
# Lookahead protection
# --------------------------------------------------------------------------


def test_signal_uses_only_data_through_T_minus_1():
    """The position at row T must not depend on row-T data.

    Construction: a steady contango regime (R=0.5) that abruptly flips
    to backwardation (R=1.5) on day k. With smoothing=1, the
    transition in the position series should appear at row k+1, NOT
    at row k.

    Row k:   position[k] uses ratio[k-1] = 0.5 → still long.
    Row k+1: position[k+1] uses ratio[k] = 1.5 → flat.
    """
    n = 20
    k = 10
    short = pd.Series([10.0] * n, index=_idx(n))
    long_ = pd.Series([20.0] * n, index=_idx(n))
    # Flip the short_vol on day k so the ratio crosses 1.0 there.
    short.iloc[k] = 30.0  # now ratio[k] = 30/20 = 1.5
    # Keep short=30 for rows k+1 onward as well so backwardation persists.
    short.iloc[k:] = 30.0

    pos = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=1)
    # Day k: still long (used ratio[k-1] = 0.5)
    assert pos.iloc[k] == 1.0
    # Day k+1: flips to flat (used ratio[k] = 1.5)
    assert pos.iloc[k + 1] == 0.0


def test_signal_uses_only_past_data_with_smoothing():
    """Same as above but with smoothing=3. The smoothed ratio at T
    averages rows [T-2..T]; .shift(1) means the position at T sees
    the average over [T-3..T-1]. So a single-day spike on day k
    affects positions starting at day k+1 (when the spike first
    enters the window) and is fully through by day k+3 (last day
    the spike is in the lookback window before the next shift)."""
    n = 20
    k = 10
    short = pd.Series([10.0] * n, index=_idx(n))
    long_ = pd.Series([20.0] * n, index=_idx(n))
    short.iloc[k:] = 30.0  # persistent regime flip

    pos = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=3)
    # Day k: position uses smoothed[k-1] = mean(ratio[k-3..k-1])
    # All three ratios are 0.5 → smoothed = 0.5 < 1.0 → position = 1.
    assert pos.iloc[k] == 1.0
    # Day k+1: position uses smoothed[k] = mean(ratio[k-2..k])
    # ratios are (0.5, 0.5, 1.5), mean = 0.833 < 1.0 → position = 1.
    assert pos.iloc[k + 1] == 1.0
    # Day k+2: smoothed[k+1] = mean(ratio[k-1..k+1]) = (0.5, 1.5, 1.5),
    # mean = 1.167 > 1.0 → position = 0. Transition.
    assert pos.iloc[k + 2] == 0.0
    # Day k+3 onward: fully flipped (smoothed = 1.5 > 1.0)
    assert pos.iloc[k + 3] == 0.0


# --------------------------------------------------------------------------
# Threshold parameter
# --------------------------------------------------------------------------


def test_threshold_above_one_makes_more_long_days():
    """A threshold > 1.0 lets weakly backwardated days still count as
    long. Same data, higher threshold ⇒ at least as many long days."""
    short = pd.Series(np.linspace(10, 25, 30), index=_idx(30))
    long_ = pd.Series([20.0] * 30, index=_idx(30))
    pos_strict = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=1)
    pos_loose = _vrp_signal(short, long_, threshold=1.1, signal_smoothing_days=1)
    assert pos_loose.sum() >= pos_strict.sum()


def test_threshold_below_one_makes_more_flat_days():
    """Threshold < 1.0 demands deeper contango → at least as many flat days."""
    short = pd.Series(np.linspace(10, 25, 30), index=_idx(30))
    long_ = pd.Series([20.0] * 30, index=_idx(30))
    pos_strict = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=1)
    pos_strict_minus = _vrp_signal(short, long_, threshold=0.9, signal_smoothing_days=1)
    # Strict-minus rejects some borderline days, so it has <= long days.
    assert pos_strict_minus.sum() <= pos_strict.sum()


# --------------------------------------------------------------------------
# Output shape / dtypes
# --------------------------------------------------------------------------


def test_signal_output_is_float_series_aligned_to_inputs():
    short = pd.Series([10.0, 11.0, 12.0], index=_idx(3))
    long_ = pd.Series([20.0, 21.0, 22.0], index=_idx(3))
    pos = _vrp_signal(short, long_, threshold=1.0, signal_smoothing_days=1)
    assert isinstance(pos, pd.Series)
    assert pos.dtype == float
    assert pos.index.equals(short.index)
