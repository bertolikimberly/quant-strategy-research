"""Moving-average crossover — a textbook trend-following strategy.

This is the classic "go long when the short MA is above the long MA,
go flat otherwise" rule. Optionally short when the short MA is below
the long MA (a two-sided version). It is included here for ONE reason:
to exercise the framework end-to-end with a strategy that actually
trades. It is NOT meant to be the recommended candidate — MA
crossovers on index ETFs have been heavily arbitraged, and their
honest out-of-sample Sharpe on SPY-style data is typically near zero
once realistic costs are charged. Seeing that result come out of the
bake-off correctly is itself a good sanity check on the framework.

DESIGN NOTES
------------
- Lookback windows are constructor arguments so the same class can
  instantiate multiple crossover variants (e.g. 20/100, 50/200) and
  they appear as distinct rows in the scorecard.
- The 'allow_short' flag picks between long-only (position in {0, 1})
  and long/short (position in {-1, 1}). Both are common in practice.
- The signal uses pandas' `.rolling()` with the default closed='right'
  setting, which is end-of-bar — i.e. the MA at row T includes the
  close of T. That's CAUSAL (no future data) and matches the engine's
  lookahead convention.
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import get_prices
from src.strategies.base import Strategy


class MovingAverageCrossover(Strategy):
    """Long when short MA > long MA; optionally short when reversed.

    Args:
        ticker:      underlying symbol (default: 'SPY').
        fast_window: lookback for the short MA. 50 days is the most
                     common choice; 20 is faster (more turnover);
                     100 is slower (more conservative).
        slow_window: lookback for the long MA. 200 days is the
                     "golden cross" textbook value.
        allow_short: if True, go -1 when fast < slow. If False, stay
                     flat (position = 0) in that regime.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        fast_window: int = 50,
        slow_window: int = 200,
        allow_short: bool = False,
    ) -> None:
        if fast_window >= slow_window:
            # Otherwise the "fast" MA isn't faster, and the rule is
            # nonsense. Failing loudly here beats silently producing
            # a strategy that always trades in the wrong direction.
            raise ValueError(
                f"fast_window ({fast_window}) must be strictly less "
                f"than slow_window ({slow_window})."
            )

        self.ticker = ticker
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.allow_short = allow_short

        # Encode every parameter that affects results in the name, so
        # two MA-crossover variants don't collide in the scorecard.
        side = "ls" if allow_short else "lo"  # long/short vs long-only
        self.name = (
            f"ma_xover_{ticker}_{fast_window}_{slow_window}_{side}"
        )

    def prepare_data(self, start: str, end: str) -> pd.DataFrame:
        # We need extra history before `start` so the slow MA is
        # warm by the time the scorecard period begins. Pull
        # `slow_window` trading days of extra prefix, which on a
        # business-day calendar is roughly slow_window * 7/5 calendar
        # days. We add 1.5x just to be safe (holidays, weekends).
        prefix_days = int(self.slow_window * 1.5) + 7
        warm_start = (
            pd.Timestamp(start) - pd.Timedelta(days=prefix_days)
        ).strftime("%Y-%m-%d")
        data = get_prices(self.ticker, warm_start, end)
        # Compute the two moving averages once here so they're cached
        # on the frame; generate_positions just reads them.
        data = data.copy()
        data["ma_fast"] = data["close"].rolling(self.fast_window).mean()
        data["ma_slow"] = data["close"].rolling(self.slow_window).mean()
        # Trim back to the requested window. The MAs computed above
        # are still valid (they were computed with the full prefix).
        return data.loc[start:end]

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        # Boolean "fast above slow" series. Where either MA is NaN
        # (first few rows after a trim), the comparison is False —
        # we treat that as "flat" rather than guessing.
        bullish = data["ma_fast"] > data["ma_slow"]

        if self.allow_short:
            # +1 when bullish, -1 when bearish. Note: where bullish is
            # NaN-derived False, we get -1, which would mean shorting
            # during warm-up if any NaNs remained. The .loc[start:end]
            # in prepare_data ALMOST removes that — but the first row
            # of the trimmed frame might still have NaN MAs if start
            # lands on a trading day before slow_window prior days
            # exist. Be defensive: set to 0 where either MA is NaN.
            position = pd.Series(0.0, index=data.index, name="position")
            valid = data["ma_fast"].notna() & data["ma_slow"].notna()
            position[valid & bullish] = 1.0
            position[valid & ~bullish] = -1.0
            return position
        else:
            position = pd.Series(0.0, index=data.index, name="position")
            valid = data["ma_fast"].notna() & data["ma_slow"].notna()
            position[valid & bullish] = 1.0
            # else stay flat (already 0)
            return position
