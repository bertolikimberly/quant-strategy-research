"""Volatility Risk Premium via VIX Term Structure.

THE HYPOTHESIS (well-documented in academic + practitioner literature)
---------------------------------------------------------------------
S&P 500 options systematically price in more volatility than is later
realised — the "volatility risk premium" (VRP). This premium is paid
by buyers of equity vol (e.g. portfolio insurers) and earned by net
sellers of equity vol (typically the market-maker community plus
short-vol funds). One way to read the premium without writing options
is the SHAPE of the VIX futures term structure:

  CONTANGO (longer-dated VIX > front-month):
      Normal regime. Forward implied vol is higher than spot implied
      vol — buyers of forward vol are paying up. Historically a
      "risk-on, harvest the premium" environment for being long
      equity beta.

  BACKWARDATION (front-month VIX > longer-dated):
      Stress regime. The market is paying up for IMMEDIATE
      protection — typically only happens during dislocations
      (March 2020, Oct 2008). A historically poor time to be
      long equity beta.

Strategy: be long SPY in contango, flat in backwardation. The idea
isn't to harvest the premium directly (we'd need to short vol for
that); it's that the term-structure REGIME is a cheap, lagged-but-
informative signal for whether equity beta is currently a paid risk
or an unpaid one.

DATA NOTE — WE USE A PROXY FOR THE TERM STRUCTURE
-------------------------------------------------
Individual VIX futures (VX1, VX2, ...) are not on yfinance. The
classic free-data proxy is:
    front-month implied vol = ^VIX  (1-month constant-maturity)
    longer-dated implied vol = ^VXV (3-month constant-maturity)
    ratio R = VIX / VXV
    R < 1 → contango  → risk-on (long SPY)
    R > 1 → backwardation → risk-off (flat)

UNFORTUNATELY ^VXV is no longer pullable through yfinance as of
the time this code was written; the symbol is recognised but
returns zero rows. We therefore fall back automatically to the
short-end proxy:
    short_dated = ^VIX9D  (9-day implied vol)
    long_dated  = ^VIX    (1-month implied vol)
    ratio R = VIX9D / VIX
The contango/backwardation interpretation is qualitatively the
same (R < 1 means the very front is sloping UP, i.e. contango),
but the signal now reflects only the 9-day-to-1-month section
of the curve rather than the 1m-to-3m section. This is a real
substitution and may meaningfully affect results — a backtest
done with proper VXV data would not match this one exactly.
The substitution is logged loudly at runtime so it can't be
forgotten when reading results.

(yfinance also doesn't have ^VIX9D before 2011-01-03, so the
strategy's effective backtest start date is ~one year later than
the rest of the bake-off. This shows up as a smaller n_obs in
the scorecard; comparisons are still informative because we use
the same OOS split RATIO on each strategy's own window.)

LOOKAHEAD CONTRACT (PER THE BRIEF)
----------------------------------
The brief asks for the position used on date T+1's return to be the
signal computed from data through T-1 — i.e. an extra-conservative
two-day lag between data observation and PnL application. We
implement that explicitly:

    position[T] = (smoothed_ratio.shift(1)[T] < threshold)
                = (smoothed_ratio[T-1]      < threshold)
                = decision using data through T-1.

Then the framework engine's standard `.shift(1)` on positions gives:

    position USED on day T = position[T-1]
                           = decision from data through T-2.

So the position applied to T+1's return is the one decided at T
using data through T-1 — exactly what the brief specifies.

Caveat we accept: with a 5-day smoothing window we further dilute
the signal's recency. The default smoothing is here because the
ratio whipsaws around 1.0 on quiet days and produces a lot of
in-and-out trades that bleed P&L to costs without real signal value.
A user wanting a purer raw-signal test can pass signal_smoothing_days=1.
"""

from __future__ import annotations

import warnings

import pandas as pd

from src.data.loader import get_prices
from src.strategies.base import Strategy


# Preferred (long-dated) proxy and its fallback. Module-level so the
# fallback rule is visible and unit tests can reference the same
# constants the strategy uses.
_PRIMARY_SHORT_VOL = "^VIX"
_PRIMARY_LONG_VOL = "^VXV"
_FALLBACK_SHORT_VOL = "^VIX9D"
_FALLBACK_LONG_VOL = "^VIX"


def _vrp_signal(
    short_vol: pd.Series,
    long_vol: pd.Series,
    threshold: float = 1.0,
    signal_smoothing_days: int = 5,
) -> pd.Series:
    """Compute the regime-position series from a pair of vol indices.

    Args:
        short_vol: shorter-dated implied vol (e.g. VIX or VIX9D),
            indexed by date. Must align with long_vol's index.
        long_vol:  longer-dated implied vol (e.g. VXV or VIX),
            indexed by date.
        threshold: ratio cutoff. R < threshold → long; R >= threshold
            → flat. 1.0 is the natural break between contango and
            backwardation; values >1 make the rule LESS likely to
            sit flat (more risk-on), <1 more conservative.
        signal_smoothing_days: rolling-mean window applied to the
            ratio before thresholding. >=2 smooths whipsaws around
            the boundary; 1 disables smoothing entirely.

    Returns:
        Series of {0.0, 1.0} positions, indexed like the inputs.
        The value at row T is the position the strategy intends to
        hold at close of T, computed using R values through T-1.

    Causality discussion:
        We compute the smoothed ratio THEN shift it forward by one
        day. Smoothing first and shifting second gives the same
        result as shifting first and smoothing second (both yield
        a window over [T-window..T-1]); the chosen order makes the
        intent — "use yesterday's smoothed signal" — visible in
        a single .shift(1) call.

    NaN handling:
        On warm-up days (first `signal_smoothing_days` rows), the
        rolling mean is NaN. (NaN < threshold) evaluates to False in
        pandas, so the position correctly defaults to 0.0 (flat)
        during warm-up. We also stay flat if R is NaN later in the
        series (gap in either input), again because (NaN < t) → False.
    """
    ratio = short_vol / long_vol
    if signal_smoothing_days > 1:
        # min_periods=signal_smoothing_days means we WAIT for a full
        # window before producing any value. The first 4 rows stay
        # NaN; positions default to 0 there.
        smoothed = ratio.rolling(
            window=signal_smoothing_days,
            min_periods=signal_smoothing_days,
        ).mean()
    else:
        # No smoothing requested — pass the raw ratio through.
        smoothed = ratio

    # .shift(1): position at T uses smoothed value from T-1. Then
    # the framework engine's own lag (positions.shift(1) inside
    # run_backtest) supplies the second day of lag.
    positions = (smoothed.shift(1) < threshold).astype(float)
    positions.name = "position"
    return positions


class VRPTermStructure(Strategy):
    """Long SPY in VIX-curve contango, flat in backwardation.

    Constructor parameters mirror the brief:
        long_ticker:        the traded equity index ETF (default 'SPY').
        short_vol_ticker:   front-end implied-vol proxy (default '^VIX').
        long_vol_ticker:    longer-dated implied-vol proxy
                            (default '^VXV', falls back to '^VIX9D'
                            with a swap of short→'^VIX9D' and
                            long→'^VIX' if VXV is unavailable).
        threshold:          ratio cutoff for contango (default 1.0).
        signal_smoothing_days: rolling SMA on the ratio (default 5).
    """

    def __init__(
        self,
        *,
        long_ticker: str = "SPY",
        short_vol_ticker: str = _PRIMARY_SHORT_VOL,
        long_vol_ticker: str = _PRIMARY_LONG_VOL,
        threshold: float = 1.0,
        signal_smoothing_days: int = 5,
    ) -> None:
        self.long_ticker = long_ticker
        self.short_vol_ticker = short_vol_ticker
        self.long_vol_ticker = long_vol_ticker
        self.threshold = threshold
        self.signal_smoothing_days = signal_smoothing_days

        smoothing_tag = (
            f"sm{signal_smoothing_days}"
            if signal_smoothing_days > 1 else "raw"
        )
        # Encode every consequential parameter in the name so multiple
        # parameter variants don't collide in the scorecard.
        self.name = (
            f"vrp_term_{long_ticker}_"
            f"thr{int(round(threshold * 100))}_"
            f"{smoothing_tag}"
        )

        # Will hold the actual pair of vol tickers used after the
        # fallback decision in prepare_data. Useful for reporting and
        # for the diagnostics CSV.
        self._used_short_vol: str | None = None
        self._used_long_vol: str | None = None

    def prepare_data(self, start: str, end: str) -> pd.DataFrame:
        # 1. Equity leg.
        spy = get_prices(self.long_ticker, start, end)["close"]

        # 2. Vol legs. Try the configured pair; if the long-dated
        # leg comes back empty (yfinance's current behaviour for
        # ^VXV) fall back to the (^VIX9D, ^VIX) pair and warn.
        short_vol, long_vol, used_pair = self._load_vol_pair(start, end)
        self._used_short_vol, self._used_long_vol = used_pair

        # 3. Inner-join all three series on common trading days.
        # If the vol legs start later than SPY (e.g. ^VIX9D starts
        # 2011-01-03 while SPY goes back to 1993), the dropna()
        # here trims the strategy's window to the common range.
        # We don't pad with zeros — staying flat on days where the
        # signal can't be computed is the correct default, but only
        # within the strategy's own backtest window.
        df = pd.concat(
            {"close": spy, "short_vol": short_vol, "long_vol": long_vol},
            axis=1,
        ).dropna()

        if df.empty:
            raise ValueError(
                f"After aligning {self.long_ticker}, {used_pair[0]}, "
                f"{used_pair[1]} on common dates, the DataFrame is empty. "
                f"Check that all three symbols return data on the "
                f"requested window."
            )

        return df

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        # All causality / smoothing logic is in the standalone helper
        # so the unit tests can exercise it without going through the
        # data loader.
        return _vrp_signal(
            short_vol=data["short_vol"],
            long_vol=data["long_vol"],
            threshold=self.threshold,
            signal_smoothing_days=self.signal_smoothing_days,
        )

    # ---- Internal --------------------------------------------------------

    def _load_vol_pair(
        self,
        start: str,
        end: str,
    ) -> tuple[pd.Series, pd.Series, tuple[str, str]]:
        """Pull the configured (short, long) vol pair, falling back if needed.

        Returns (short_vol_series, long_vol_series, used_tickers_tuple).

        Fallback rule: if the configured long_vol_ticker is the default
        '^VXV' and either leg fails to load, swap to (^VIX9D, ^VIX).
        For any non-default configuration, we re-raise the original
        loader error — the caller asked for something specific, so
        silently substituting would hide a real problem.
        """
        is_default_pair = (
            self.short_vol_ticker == _PRIMARY_SHORT_VOL
            and self.long_vol_ticker == _PRIMARY_LONG_VOL
        )

        def _try(ticker: str) -> pd.Series:
            # `get_prices` raises ValueError on empty data. We catch
            # only that specific failure — network errors should still
            # propagate so the user sees them.
            return get_prices(ticker, start, end)["close"]

        try:
            short_vol = _try(self.short_vol_ticker)
            long_vol = _try(self.long_vol_ticker)
            return short_vol, long_vol, (
                self.short_vol_ticker, self.long_vol_ticker
            )
        except ValueError as exc:
            if not is_default_pair:
                raise
            warnings.warn(
                f"[vrp_term_structure] Could not load the default vol "
                f"pair ({_PRIMARY_SHORT_VOL}, {_PRIMARY_LONG_VOL}) from "
                f"yfinance ({exc}). Falling back to "
                f"({_FALLBACK_SHORT_VOL}, {_FALLBACK_LONG_VOL}). This "
                f"measures the 9D-vs-1M section of the curve instead of "
                f"the 1M-vs-3M section the strategy is designed for; "
                f"contango/backwardation semantics are similar but the "
                f"backtest is NOT a like-for-like replacement.",
                stacklevel=2,
            )
            short_vol = _try(_FALLBACK_SHORT_VOL)
            long_vol = _try(_FALLBACK_LONG_VOL)
            return short_vol, long_vol, (
                _FALLBACK_SHORT_VOL, _FALLBACK_LONG_VOL,
            )
