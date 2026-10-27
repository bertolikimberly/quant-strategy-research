"""The Strategy interface.

WHY THIS FILE IS THE CORE OF THE FRAMEWORK
------------------------------------------
The bake-off is only "fair" if every candidate strategy is poked and
prodded in exactly the same way: same data range, same cost model, same
metrics, same validation. The way we enforce that in code is by funnelling
every strategy through one tight interface — the abstract base class
defined below.

The interface intentionally splits a strategy into TWO pure functions:

    1. prepare_data(start, end) -> DataFrame
         Returns the OHLCV-like frame the strategy needs. This is where
         a strategy declares its data dependencies (a single ticker?
         a basket? auxiliary features like VIX or rates?). The frame's
         index is dates; the 'close' column is required because the
         backtest engine uses it to compute the underlying's daily
         return. Anything else the strategy needs (e.g. moving averages,
         RSI, fundamental ratios) can live in additional columns.

    2. generate_positions(data) -> Series
         Given the prepared data, return a Series indexed identically
         to `data` whose value at row T is the TARGET POSITION at
         close of T. Convention:
             +1.0  = fully long the underlying (100% of capital)
              0.0  = flat
             -1.0  = fully short
         Fractional values (e.g. 0.5) are allowed and represent
         partial sizing. Values beyond [-1, 1] represent leverage and
         will incur financing costs in the engine.

Why this split? Because the cost / lookahead / financing accounting is
identical across strategies and belongs in ONE place — the engine. If
each strategy computed its own return series, every strategy would
have its own (subtly different, and inevitably buggy) cost model, and
the bake-off would be measuring "who has the most generous cost
assumptions" instead of "who has the best signal".

LOOKAHEAD CONTRACT
------------------
`generate_positions` MUST be causal: the value of the output Series at
date T is allowed to use ONLY data available on or before close of T.
The engine then lags positions by one day before applying them to next-
day returns (a position decided at close of T earns the return from
close T to close T+1). Concretely:

    daily_return[T+1] = position[T] * asset_return[T+1] - costs

If a strategy needs "tomorrow's open" or "this bar's high after the
close" it must be reformulated; otherwise the backtest is fiction.

We can't fully enforce causality from inside the framework (a strategy
could, in principle, use `data['close'].shift(-1)` in its signal logic
and produce great-looking nonsense). Code review is the only real
defence. A useful sanity check: any strategy that looks too good on
in-sample data and falls apart in the permutation test almost always
has lookahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Abstract base class for every candidate strategy in the bake-off.

    Concrete subclasses must:

        * set a class-level `name` (short identifier used in the
          scorecard table and filenames),
        * implement `prepare_data(start, end)`,
        * implement `generate_positions(data)`.

    Subclasses MAY override `__init__` to accept tunable hyperparameters
    (lookbacks, thresholds, ticker symbols, etc.). The validation module
    treats those hyperparameters as fixed — the OOS split tests whether
    the chosen parameters generalise; it does not itself search over
    parameter grids. If you want to tune, do that against the in-sample
    portion of the data, then re-instantiate with the chosen parameters
    and run the full bake-off.
    """

    # Short identifier (e.g. "buy_and_hold", "ma_crossover_50_200").
    # Used as a row label in the scorecard CSV and as a filename stem.
    # Subclasses MUST override this; the default empty string would
    # collide in the results table if two strategies forgot to set it.
    name: str = ""

    @abstractmethod
    def prepare_data(self, start: str, end: str) -> pd.DataFrame:
        """Load and prepare all data this strategy needs.

        Args:
            start: ISO date string (YYYY-MM-DD), inclusive.
            end:   ISO date string (YYYY-MM-DD), inclusive.

        Returns:
            DataFrame indexed by `pd.DatetimeIndex` (business days).
            MUST contain at least a column named 'close' which is the
            (split/dividend-adjusted) closing price of the strategy's
            primary underlying. The engine uses 'close' to compute the
            asset's daily return; any other columns are for the
            strategy's own use (indicators, etc.) and are ignored by
            the engine.

        Notes:
            Implementations should be idempotent and cache-friendly:
            calling twice with the same arguments should return the
            same frame. The provided `src.data.loader.get_prices`
            handles caching for the common case of yfinance-sourced
            equity data.
        """

    @abstractmethod
    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        """Translate the prepared data into a target-position series.

        Args:
            data: the DataFrame returned by `prepare_data`.

        Returns:
            Series with the SAME index as `data`. The value at row T is
            the target position to hold at the close of T, in units of
            "fraction of capital deployed":
                +1.0 = fully long
                 0.0 = flat
                -1.0 = fully short
            Values outside [-1, 1] are interpreted as leverage and
            will incur financing costs.

        Causality requirement:
            Element T may only depend on `data` rows with index <= T.
            The backtest engine will lag this series by one period
            before computing PnL, so a position "decided at T" earns
            the return from T to T+1.
        """
