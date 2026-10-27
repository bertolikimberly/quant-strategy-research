"""Performance metrics ("scorecard") for a daily return series.

Every metric here takes a `pd.Series` of NET daily returns (the
`net_returns` produced by the backtest engine) and returns a single
float — except `scorecard()` at the bottom, which bundles them all
into a dict for tabular reporting.

A NOTE ON ANNUALISATION
-----------------------
We annualise by treating each calendar year as 252 trading days. This
is the US-equity convention; for FX or crypto, 365 is closer.
Annualisation of vol uses the sqrt-of-time rule, which is correct
under the (false but useful) assumption that returns are i.i.d.

WHAT EACH METRIC TELLS YOU — AND WHAT IT DOESN'T
------------------------------------------------
- annualized_return: average daily return scaled by 252. Easy to
  interpret. Does NOT account for compounding within a year; for
  small daily returns the difference is small.
- annualized_volatility: stdev of daily returns * sqrt(252).
  Standard. Assumes constant variance; if your strategy has
  regime-dependent vol (most do), this is a blunt average.
- sharpe: ann_return / ann_vol, assuming risk-free rate ~ 0.
  We do not subtract a risk-free rate here because (a) the
  difference is tiny on daily-rebalanced strategies and (b) the
  bake-off compares strategies against each other, not against
  cash. ANY claim of a Sharpe number should disclose the rf
  assumption. We disclose: rf = 0.
  Caveat: Sharpe punishes upside volatility just as much as
  downside, which is why we also report Sortino.
- sortino: same as Sharpe but using downside deviation
  (semi-deviation below 0). Better matches an investor's pain
  function. Less standard, so always report both.
- max_drawdown: largest peak-to-trough fall in the equity curve,
  as a negative fraction. -0.30 means the strategy at some point
  was 30% below its prior high. The single most important "can
  you stomach this strategy" number.
- hit_rate: fraction of days with strictly positive net return.
  A surprisingly weak signal of skill — many great trend
  strategies have hit rates below 50% but big winners.
- turnover: average daily |Δposition|. Annualised by * 252 so it
  reads as "how many times per year do we trade through our
  capital base". High turnover doesn't mean bad; it just means
  the strategy is more cost-sensitive.
- newey_west_tstat: t-statistic on mean daily return using a
  HAC (heteroskedasticity- and autocorrelation-consistent)
  standard error. The right t-stat to report for a strategy
  return series because daily returns are typically auto-
  correlated. See the function docstring for the lag choice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


# Annualisation constant. Matches engine.costs.TRADING_DAYS_PER_YEAR;
# we redeclare here instead of importing to keep the metrics module
# free of cross-package dependencies (it should be readable as a
# standalone statistics library).
TRADING_DAYS_PER_YEAR = 252


def _clean(returns: pd.Series) -> pd.Series:
    """Drop NaNs, coerce to float64.

    Most metrics blow up on NaNs (.mean() silently skips them, but
    .std() with ddof=1 doesn't know how many obs you really had).
    We standardise on "drop NaNs at the top of each metric so the
    sample size is unambiguous".
    """
    return returns.dropna().astype(float)


def annualized_return(returns: pd.Series) -> float:
    """Arithmetic mean of daily returns * 252.

    We use the arithmetic (not geometric) mean because (a) it's the
    correct estimator of the EXPECTED daily return, and (b) it
    matches what's used inside Sharpe — keeping them consistent
    means ann_return / ann_vol equals the Sharpe number reported.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float(r.mean() * TRADING_DAYS_PER_YEAR)


def annualized_volatility(returns: pd.Series) -> float:
    """Sample stdev of daily returns * sqrt(252).

    ddof=1 is the unbiased estimator; the difference vs ddof=0 is
    irrelevant beyond ~50 daily obs, but using ddof=1 is the
    statistical convention.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio assuming a (small) constant risk-free rate.

    Args:
        returns: daily net returns (as fractions, not %).
        risk_free_annual: annual risk-free rate; default 0. If you
            change this you must also disclose it in the report.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess = r - daily_rf
    sd = excess.std(ddof=1)
    if sd == 0:
        # All-constant return series. Either a bug, or a strategy
        # that never traded. Returning NaN rather than +/-inf
        # signals "undefined" downstream.
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    """Sortino ratio: like Sharpe but the denominator only counts downside.

    Downside deviation is the RMS of negative excess returns (treating
    positive ones as 0). We use the standard convention of taking the
    mean of squared *negative* excess (not squared deviations from the
    mean), which is the asymmetric loss function Sortino originally
    proposed.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    daily_rf = risk_free_annual / TRADING_DAYS_PER_YEAR
    excess = r - daily_rf
    downside = excess.clip(upper=0.0)
    # mean of squared losses (zeros contribute nothing). sqrt gives
    # the downside deviation in daily-return units.
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the equity curve.

    Returns a NEGATIVE number (or zero). -0.20 means the strategy was
    20% below its prior peak at the worst point.

    Why this is a critical metric: investors quit during drawdowns.
    A strategy with a 1.5 Sharpe and a 50% max drawdown is much
    harder to actually run than a 0.8 Sharpe with a 10% drawdown.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    # running max gives the "peak so far"; the ratio is the fractional
    # equity remaining at the worst point relative to that peak.
    running_max = wealth.cummax()
    drawdown_series = wealth / running_max - 1.0
    return float(drawdown_series.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with strictly positive net return.

    Zero-return days (flat strategy days) are NOT counted as hits.
    Common alternative: "wins / (wins + losses)" excluding zero
    days. We use the simpler definition because for any active
    strategy zero-return days are rare and the distinction doesn't
    matter; for a benchmark like buy-and-hold there are no zero
    days at all.
    """
    r = _clean(returns)
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def turnover(positions: pd.Series) -> float:
    """Average annualised turnover, i.e. (mean |Δposition| per day) * 252.

    Reads as "how many round-trips through capital per year". A
    buy-and-hold strategy has turnover near 0 (single initial trade).
    A daily-rebalanced strategy that flips sign every day has
    turnover ~ 2 * 252 = 504.

    Note: we accept the LAGGED position series (the one actually
    applied to PnL), not the strategy's raw output. Using raw
    positions would slightly overstate turnover for the first row
    where the engine doesn't actually trade yet.
    """
    if positions.empty:
        return float("nan")
    # First-day full position is also a trade; account for it the
    # same way the engine does (see engine/backtest.py).
    delta = positions.diff()
    delta.iloc[0] = positions.iloc[0]
    return float(delta.abs().mean() * TRADING_DAYS_PER_YEAR)


def newey_west_tstat(
    returns: pd.Series,
    lags: int | None = None,
) -> float:
    """t-statistic on the mean of `returns` with Newey-West (HAC) SE.

    Why HAC instead of a plain t-test:
        Daily strategy returns are usually autocorrelated (a trend
        strategy holds for many days; a mean-reversion strategy
        oscillates). The plain-OLS standard error of the mean
        assumes independence and will report a t-statistic that's
        too big (in absolute value) when there's positive
        autocorrelation. Newey-West corrects for that by widening
        the SE.

    Lag choice:
        We default to the Newey-West (1994) rule of thumb:
            L = floor(4 * (T/100)^(2/9))
        for T = number of observations. Caller can override if they
        have a reason (e.g. a strategy with known weekly seasonality
        might want L >= 5).

    Returns:
        The two-sided t-statistic. |t| > ~2 corresponds to p < 0.05
        (asymptotically), but multiple-testing corrections apply
        when comparing many strategies — see CLAUDE.md.
    """
    r = _clean(returns)
    if len(r) < 5:
        return float("nan")

    if lags is None:
        # Newey-West (1994) automatic lag. floor() because lags must
        # be integer; max(1, ...) so we always include at least one
        # lag (the formula yields 0 for T < ~25).
        lags = max(1, int(np.floor(4 * (len(r) / 100.0) ** (2.0 / 9.0))))

    # Regress returns on a constant only — the OLS coefficient is
    # the mean, and statsmodels gives us the HAC SE of that mean
    # when we ask for cov_type='HAC'.
    X = sm.add_constant(np.ones(len(r)))  # design matrix is just a column of 1s
    # statsmodels complains if you have a duplicate constant column,
    # so build X explicitly as a single column of 1s instead.
    X = np.ones((len(r), 1))
    model = sm.OLS(r.values, X)
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(result.tvalues[0])


def scorecard(
    returns: pd.Series,
    positions: pd.Series | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Compute every metric and return as a dict (one scorecard row).

    Args:
        returns: NET daily return series from the backtest engine.
        positions: lagged position series. Required to compute
            turnover; if None, turnover is reported as NaN.
        label: free-text identifier (typically the strategy name +
            the split it represents, e.g. "ma_crossover [OOS]"). Goes
            into the returned dict under the 'label' key so a list
            of these can be concatenated into a comparison table.

    Returns:
        Dict with all the metric names as keys. Stable key order so
        the eventual CSV columns are predictable.
    """
    return {
        "label": label,
        "n_obs": int(_clean(returns).shape[0]),
        "ann_return": annualized_return(returns),
        "ann_vol": annualized_volatility(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "hit_rate": hit_rate(returns),
        "turnover": (
            turnover(positions) if positions is not None else float("nan")
        ),
        "nw_tstat": newey_west_tstat(returns),
    }
