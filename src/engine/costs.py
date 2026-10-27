"""Transaction-cost, slippage, and financing model.

DESIGN CHOICE: a single dataclass `CostModel` carries every dollar that
leaks out of a strategy as it trades. We pass ONE instance through the
backtest engine, so changing assumptions for every strategy
simultaneously is a one-line edit (it has to be — if costs differed
between strategies the bake-off would be meaningless).

WHAT WE MODEL
-------------
1. Half-spread + commission (entry/exit):
       cost per unit traded = (half_spread_bps + commission_bps) / 10000
   This is applied to TURNOVER (the absolute change in target position
   between consecutive days), in units of fraction-of-capital. Putting
   on a +1.0 position from flat costs the same as flipping from -0.5
   to +0.5: both move one unit of notional through the market.

   We deliberately model only HALF the spread per trade, because cross-
   ing the spread is a one-sided cost: you pay half on the way in and
   half on the way out (round-trip is the full spread). This is the
   convention most academic backtests use.

2. Per-trade slippage (impact-of-our-own-order proxy):
       extra cost per unit traded = slippage_bps / 10000
   This is a placeholder for the gap between the mid-price the engine
   uses and the price you'd actually fill at. For SPY-sized notional
   on a course project it's basically zero; we keep the parameter so
   we can stress-test sensitivity to it.

3. Overnight financing (carry on leverage and shorts):
       daily financing rate = financing_rate_annual / 252
       financing cost on day T = |position[T]| * 1_{leverage or short}
                                 * daily financing rate
   This represents the cost a broker charges to fund a leveraged or
   short position overnight. For an unlevered long-only book it is
   zero (you own the asset outright). For a short position, you pay
   the borrow rate. For a 2x-long position, you pay margin interest
   on the 1x of borrowed capital.

   Simplifying assumption we make explicit: we charge the same rate
   for short borrow and for long margin. In reality, hard-to-borrow
   short rates can be wildly higher. For broad ETFs (SPY, QQQ), the
   error is small; for individual small-cap shorts, this model would
   understate cost meaningfully.

WHAT WE DO NOT MODEL
--------------------
- Market impact (size-dependent slippage): orders move the book; we
  treat fills as price-takers. Defensible for course-project notional
  in a liquid ETF; not defensible at fund-scale.
- Bid/ask asymmetry conditional on direction (offer-side liquidity is
  often thinner than bid-side, especially intraday around news).
- Borrow availability: we assume any short is fillable.
- Stamp duties / SEC fees / FINRA TAF: tiny but real on US equities;
  bundled into commission_bps as a rough approximation.

DEFAULTS
--------
The defaults below correspond roughly to "a retail brokerage trading
liquid US ETFs in 2025": ~1 bp commission (most brokers are now zero,
but exchange/regulatory fees still apply), ~2 bp half-spread (SPY is
typically a one-cent wide market, ~0.2 bp; a basket-weighted average
across less liquid names lands around 2 bp), 0.5 bp slippage, and a
5% annual financing rate (close to the Fed Funds rate as of 2025).

Course-project guidance: leave the defaults alone for the official
bake-off. If you want to show robustness, RE-run with 2x these costs
in a separate report — strategies whose ranking flips are fragile.
"""

from __future__ import annotations

from dataclasses import dataclass


# 252 US equity trading days per year — the standard annualisation
# constant used throughout this codebase. Hardcoded here instead of
# imported from metrics/ to keep the cost module free of cross-package
# dependencies (engine.costs is a leaf module).
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    """All assumptions about money leaking out of a strategy.

    Frozen so it can't be mutated mid-backtest. If you want to change
    a number, construct a new CostModel — that way any change is
    explicit in the call site, not buried in a state mutation.

    Units note: every "bps" field is in basis points (1 bp = 0.0001).
    `financing_rate_annual` is in raw decimal (0.05 = 5%/yr).
    """

    half_spread_bps: float = 2.0       # see module docstring
    commission_bps: float = 1.0        # see module docstring
    slippage_bps: float = 0.5          # impact placeholder
    financing_rate_annual: float = 0.05

    def per_unit_trade_cost(self) -> float:
        """Fractional cost charged on each unit of |Δposition| traded.

        A "unit" here is "fraction of total capital deployed", so a
        move from position=0 to position=+1 has |Δposition|=1 and
        incurs this many basis points worth of cost.

        We sum the three trade-related components because they all
        scale linearly with traded notional. The financing component
        is separate because it accrues per day held, not per trade.
        """
        return (
            self.half_spread_bps
            + self.commission_bps
            + self.slippage_bps
        ) / 10_000.0

    def daily_financing_rate(self) -> float:
        """Per-day financing rate, computed from the annual rate.

        We use simple (non-compound) division by 252. For a 5% annual
        rate over a one-day horizon, the difference between simple and
        continuous compounding is on the order of 1e-7 — well below
        any other source of error in the model.
        """
        return self.financing_rate_annual / TRADING_DAYS_PER_YEAR


# A convenience instance with the defaults documented above. Use
# `DEFAULT_COST_MODEL` for "the bake-off's official cost assumption".
# When testing sensitivity, construct a new CostModel locally.
DEFAULT_COST_MODEL = CostModel()
