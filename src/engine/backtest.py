"""The backtest engine.

Takes a `Strategy` object, calls its data-prep + position-generation
methods, and turns the result into a daily NET return series (after
costs and financing). This is the only place in the codebase where
PnL is computed; every strategy gets the exact same treatment, which
is the whole point of the framework.

THE PnL FORMULA, IN PROSE
-------------------------
For day T+1, three things happen:

  (a) MARK-TO-MARKET: the position we held at close of T earns the
      underlying's return from close T to close T+1.

         gross_return[T+1] = position[T] * asset_return[T+1]

      Note the time index: position[T] (decided at close of T) earns
      return[T+1] (close T -> close T+1). This shift is THE thing that
      keeps the backtest free of lookahead — Strategy.generate_positions
      returns its decisions unshifted; the engine does the shift, in
      one place, where it can be reviewed.

  (b) TRADING COST: if the position changed from T-1 to T, we pay the
      per-unit trade cost on the absolute change. Conceptually this
      cost is incurred AT close of T (when the trade happens), but
      since the strategy hasn't started earning the new position's
      return until T+1, charging it on T+1 vs T is purely cosmetic in
      this discrete-time model. We charge it on T+1 so the cost shows
      up in the same row as the gross return that justified the trade.

         turnover[T] = |position[T] - position[T-1]|
         trade_cost[T+1] = turnover[T] * cost_model.per_unit_trade_cost()

      EDGE CASE: on the very first day with a non-zero position,
      turnover[T] is the full |position[T]| (we're trading from zero).
      pandas .diff() returns NaN for the first row, which we fill
      with the absolute initial position so that establishing the
      book is correctly costed.

  (c) FINANCING: any leverage (|position| > 1) or any short
      (position < 0) costs the daily financing rate on the leveraged
      / borrowed notional. We compute this as

         leveraged_notional[T] = max(0, |position[T]| - 1)
                                  + max(0, -position[T])

      i.e. (a) the part of the absolute position above 1x, plus
      (b) the absolute short component. For an unlevered long this
      is zero; for a 50/50 long-short market-neutral book (position
      = +1 long minus +1 short, net 0), leveraged_notional = 1 (just
      the short borrow). That matches how a real prime broker would
      bill: gross exposure pays funding, net exposure does not.

Final daily net return:

    net_return[T+1] = gross_return[T+1] - trade_cost[T+1] - financing[T+1]

ASSUMPTIONS WORTH REPEATING
---------------------------
- Returns are simple (arithmetic), not log. The compounding error
  is below 1bp/day for any reasonable equity strategy.
- Capital is rebalanced to the target position at every close — i.e.
  the position is expressed as a fraction of CURRENT equity, not
  initial equity. This matches how most fund-of-fund-style strategies
  are run; if you wanted "fixed-fractional" sizing on initial capital,
  you'd need a different accounting model.
- The cost is taken out of the strategy return, NOT the position.
  So a strategy holding +1 in SPY that pays 1bp in costs earns
  (SPY_return - 1bp) on day T+1, not (0.9999 * SPY_return). The
  difference between these two formulations is second-order and
  irrelevant at realistic cost levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.costs import CostModel, DEFAULT_COST_MODEL
from src.strategies.base import Strategy


@dataclass
class BacktestResult:
    """Everything the engine produces for one strategy run.

    `net_returns` is the headline output — every downstream metric is
    computed from it. The other fields are stored so the validation
    module and the user-facing reports can inspect *why* a strategy
    made or lost money without re-running the backtest.
    """

    net_returns: pd.Series       # daily strategy return after all costs
    gross_returns: pd.Series     # before any costs (PnL of the raw signal)
    positions: pd.Series         # the actually-applied (i.e. lagged) positions
    raw_positions: pd.Series     # the strategy's unshifted positions
    asset_returns: pd.Series     # the underlying's daily return
    turnover: pd.Series          # |Δposition| per day
    trade_costs: pd.Series       # cost series subtracted on day-of-trade
    financing_costs: pd.Series   # financing series subtracted daily

    def equity_curve(self, initial: float = 1.0) -> pd.Series:
        """Compound the net returns into a wealth path.

        Args:
            initial: starting equity. 1.0 makes the y-axis
                read as "growth multiple"; any positive value works.

        Why simple compounding instead of `(1+r).cumprod()`-only:
            We prepend a 1.0 (or `initial`) so the curve starts at
            the chosen initial value on day zero, not on the first
            return day. This makes plots more intuitive — the curve
            actually starts AT 1.0.
        """
        if self.net_returns.empty:
            return pd.Series(dtype=float)
        wealth = (1.0 + self.net_returns.fillna(0.0)).cumprod() * initial
        return wealth


def run_backtest(
    strategy: Strategy,
    start: str,
    end: str,
    cost_model: CostModel = DEFAULT_COST_MODEL,
) -> BacktestResult:
    """Execute one strategy through the engine.

    Args:
        strategy:   any concrete Strategy subclass instance.
        start, end: ISO date strings (YYYY-MM-DD), inclusive on both ends.
                    These bound BOTH the data load and the resulting
                    return series. A strategy that needs more history
                    for warm-up (e.g. a 200-day MA) should pull a wider
                    window inside prepare_data and trim back to
                    `data.loc[start:end]` itself if it needs to —
                    otherwise the first ~200 days will be NaN positions
                    (legal; the engine treats NaN as zero).
        cost_model: cost assumptions. Pass DEFAULT_COST_MODEL unless
                    you are explicitly running a sensitivity test.

    Returns:
        BacktestResult with the full breakdown.
    """
    # --- 1. Gather inputs from the strategy ------------------------------
    # We require the strategy to do its own data loading so it can pull
    # whatever auxiliary columns it needs. The engine only cares that
    # 'close' is present.
    data = strategy.prepare_data(start, end)
    if "close" not in data.columns:
        raise ValueError(
            f"Strategy {strategy.name!r} returned a data frame without "
            f"a 'close' column. The engine needs 'close' to compute the "
            f"underlying's return."
        )

    raw_positions = strategy.generate_positions(data)
    if not raw_positions.index.equals(data.index):
        raise ValueError(
            f"Strategy {strategy.name!r}: generate_positions returned a "
            f"Series whose index does not match the data index. The "
            f"contract requires aligned indices; otherwise the lookahead "
            f"protection (a one-day .shift) would silently misalign."
        )

    # --- 2. Compute the underlying's daily return ------------------------
    # .pct_change() gives NaN at row 0 — fill with 0 so the cumulative
    # return calc downstream doesn't propagate the NaN. The first day's
    # return is just "we started here, no PnL yet".
    asset_returns = data["close"].pct_change().fillna(0.0)

    # --- 3. Lag positions by one bar to prevent lookahead ----------------
    # Position decided at close T must earn return from close T to close
    # T+1. Implementing this as .shift(1) is the single most important
    # line in the engine. If you ever refactor it out, FIRST add a unit
    # test that breaks: a strategy that perfectly knows tomorrow's return
    # should give an impossible Sharpe; with the shift in place, the
    # same strategy gets a near-zero Sharpe.
    positions = raw_positions.shift(1).fillna(0.0)

    # --- 4. Gross return (no costs) --------------------------------------
    gross_returns = positions * asset_returns

    # --- 5. Turnover and trade costs -------------------------------------
    # .diff() gives the change from row T-1 to row T. On the very first
    # day, .diff() is NaN. We treat the initial position as a trade FROM
    # zero, so fill that NaN with the absolute starting position.
    delta = raw_positions.diff()
    delta.iloc[0] = raw_positions.iloc[0]  # honest first-day trade
    turnover = delta.abs().fillna(0.0)

    # Trade cost is incurred on the day the trade happens (close of T)
    # but, per the docstring discussion, we book it on T+1 so it shows
    # up against the gross return that the trade was meant to capture.
    # The .shift(1) here mirrors the position shift; .fillna(0.0) takes
    # care of the first row (no PnL yet, no cost yet).
    trade_costs = (turnover * cost_model.per_unit_trade_cost()).shift(1).fillna(0.0)

    # --- 6. Financing -----------------------------------------------------
    # The "leveraged notional" definition is in the module docstring of
    # backtest.py. Implementation note: using .clip(lower=0) is the
    # vectorised way to write max(0, x); it's faster and clearer than
    # an .apply() with a lambda.
    abs_pos = positions.abs()
    leveraged_portion = (abs_pos - 1.0).clip(lower=0.0)  # 0 if |pos| <= 1
    short_portion = (-positions).clip(lower=0.0)         # |pos| if pos < 0
    leveraged_notional = leveraged_portion + short_portion

    financing_costs = leveraged_notional * cost_model.daily_financing_rate()

    # --- 7. Net return ----------------------------------------------------
    net_returns = gross_returns - trade_costs - financing_costs

    # Sanity: every series must share the same index. If something
    # diverged we'd rather raise here than produce subtly misaligned
    # downstream metrics.
    for name, s in [
        ("gross_returns", gross_returns),
        ("trade_costs", trade_costs),
        ("financing_costs", financing_costs),
        ("net_returns", net_returns),
    ]:
        assert s.index.equals(data.index), (
            f"engine bug: {name} index drifted from data index"
        )

    return BacktestResult(
        net_returns=net_returns,
        gross_returns=gross_returns,
        positions=positions,
        raw_positions=raw_positions,
        asset_returns=asset_returns,
        turnover=turnover,
        trade_costs=trade_costs,
        financing_costs=financing_costs,
    )
