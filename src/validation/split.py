"""Temporal in-sample / out-of-sample split.

The bake-off splits each strategy's full history into two contiguous
slices — the EARLIER chunk for in-sample work (parameter selection,
sanity-checking the implementation) and the LATER chunk for
out-of-sample confirmation. The OOS scorecard is the honest one;
the IS scorecard exists mainly so we can spot strategies whose
performance collapses on the OOS slice (i.e. strategies that were
silently overfit).

WHY A TEMPORAL SPLIT, NOT RANDOM K-FOLD
---------------------------------------
Financial returns are NOT exchangeable: there's a clear arrow of
time (regimes, autocorrelation, structural breaks). Shuffling days
into random train/test folds leaks information across the
boundary — the test set's volatility cluster is statistically
related to the training set's, and any cross-validation Sharpe is
optimistically biased. A single forward chronological split is the
defensible thing to do for a course-project sample size; serious
researchers use walk-forward cross-validation, which is the same
idea repeated with rolling windows.

WHY 70/30
---------
A 70/30 split is a defensible default: enough data on the IS side
to estimate any reasonable strategy parameters, enough on the OOS
side to get a meaningful out-of-sample Sharpe (~2-3 years on a
10-year window). The split ratio is exposed as an argument; the
bake-off itself uses 0.7 for ALL strategies so the comparison is
fair.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """A pair of slices: (in_sample, out_of_sample), preserving order.

    Both are pandas objects (Series or DataFrames depending on what
    was split) sharing the original index. The two are guaranteed
    non-overlapping and chronologically ordered: every index in
    `in_sample` is strictly earlier than every index in `out_of_sample`.
    """

    in_sample: pd.Series
    out_of_sample: pd.Series


def train_test_split(
    series: pd.Series,
    train_ratio: float = 0.7,
) -> Split:
    """Cut a time-indexed Series into the first `train_ratio` rows and the rest.

    Args:
        series: any time-indexed Series (typically the backtest's
            net_returns, but could be a positions or signal series).
        train_ratio: fraction in [0.1, 0.9] going to the in-sample
            slice. We clamp away from 0/1 because either extreme
            makes the other slice useless.

    Returns:
        Split namedtuple. If `series` has N rows, the in-sample slice
        is the first floor(N * train_ratio) rows and the OOS slice is
        the remaining rows.

    Edge cases:
        - Empty input: returns Split with two empty Series.
        - All-equal timestamps (shouldn't happen with daily data, but
          we don't try to be clever about it): the cut is by row
          count, not by date, so duplicate dates are split arbitrarily.
    """
    if not 0.1 <= train_ratio <= 0.9:
        raise ValueError(
            f"train_ratio={train_ratio} is outside [0.1, 0.9]; either "
            f"the IS or OOS slice would be too small to evaluate."
        )

    if series.empty:
        empty = series.iloc[0:0]  # preserve dtype + index type
        return Split(in_sample=empty, out_of_sample=empty)

    n = len(series)
    cut = int(n * train_ratio)
    # iloc with integer endpoints is half-open on the right, so
    # series.iloc[:cut] is rows 0..cut-1 and series.iloc[cut:] is
    # rows cut..n-1, which is exactly the partition we want.
    return Split(
        in_sample=series.iloc[:cut],
        out_of_sample=series.iloc[cut:],
    )
