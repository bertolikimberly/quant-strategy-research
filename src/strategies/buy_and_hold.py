"""Buy-and-hold benchmark.

Every active-strategy bake-off needs a passive benchmark, because the
question we ultimately care about is "is this strategy worth running
INSTEAD of just owning the asset?" — not "is this strategy's Sharpe
greater than zero?" The honest comparison is against a 100%-long
position in the same underlying, charged with realistic costs (one
initial trade, then financing at 0 because we hold it outright).

This strategy holds +1.0 from day one and never trades again. Its
backtest under the engine therefore pays:
  - ONE round-trip's worth of half-spread + commission + slippage
    on day 1 (the engine's first-row-is-a-trade convention does this
    automatically),
  - zero financing cost (|position| <= 1 and not short),
  - the underlying's daily total return for every other day.

So its annualised return is approximately the underlying's annualised
total return minus a one-off ~3.5 bp cost, amortised over the period.
On a 10-year backtest that's effectively zero.
"""

from __future__ import annotations

import pandas as pd

from src.data.loader import get_prices
from src.strategies.base import Strategy


class BuyAndHold(Strategy):
    """Always-long-the-underlying benchmark.

    The hyperparameter is just which ticker we're benchmarking
    against. Every active strategy in the bake-off should be
    compared to a buy-and-hold of the SAME underlying — otherwise
    we'd be conflating "stock-picking skill" with "asset choice".
    """

    def __init__(self, ticker: str = "SPY") -> None:
        self.ticker = ticker
        # Name encodes the parameter so multiple buy-and-hold rows
        # in the scorecard (e.g. SPY vs QQQ) don't collide.
        self.name = f"buy_and_hold_{ticker}"

    def prepare_data(self, start: str, end: str) -> pd.DataFrame:
        # Single ticker, plain price pull. The loader handles the
        # parquet cache so re-running is fast.
        return get_prices(self.ticker, start, end)

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        # Constant +1.0 across the entire window. We name the series
        # for clearer error messages downstream if something blows up
        # (pandas prints the Series name in many assertion messages).
        return pd.Series(1.0, index=data.index, name="position")
