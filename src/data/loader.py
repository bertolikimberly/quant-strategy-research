"""Price-data loader with an on-disk parquet cache.

The bake-off needs a SINGLE source of truth for prices, so that two
strategies on the same underlying see literally the same numbers. This
module is that source of truth: every strategy should pull its data
through `get_prices()` rather than calling yfinance directly.

Why caching matters
-------------------
yfinance hits Yahoo's web endpoints on every call. Two consequences:

  1. Hammering the endpoint is rude and rate-limited.
  2. The same call on different days can yield slightly different
     adjusted prices (Yahoo re-applies split/dividend adjustments
     retroactively when corporate actions are processed). If we want
     reproducible backtests, we have to PIN the data the first time
     we see it. That's what the parquet cache does.

If you need to force a fresh pull (e.g. after a real corporate-action
correction), delete the relevant file under `src/data/cache/`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yfinance as yf


# Cache location. Lives inside the package directory and is gitignored
# (see .gitignore). Using a path relative to this file means the cache
# works regardless of the CWD from which the program is invoked.
_CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _cache_path(ticker: str, start: str, end: str) -> Path:
    """Build the parquet path for a (ticker, range) tuple.

    The filename encodes the full request so two different date ranges
    cache independently — important because yfinance returns slightly
    different frames depending on the request window (e.g. timezone-
    inferred start/end alignment).
    """
    # Replace characters that aren't safe in filenames. Tickers like
    # 'BRK-B' contain hyphens (fine) but exchange-prefixed symbols
    # like '^GSPC' have characters that can confuse some filesystems.
    safe_ticker = ticker.replace("^", "_idx_").replace("/", "_")
    return _CACHE_DIR / f"{safe_ticker}__{start}__{end}.parquet"


def get_prices(
    ticker: str,
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch daily OHLCV for `ticker` between `start` and `end`, inclusive.

    Args:
        ticker: yfinance-compatible symbol, e.g. 'SPY', 'AAPL', '^GSPC'.
        start:  ISO date string (YYYY-MM-DD). Treated as inclusive.
        end:    ISO date string (YYYY-MM-DD). Treated as inclusive — we
                add one day before passing to yfinance because yfinance
                treats `end` as exclusive, which is a frequent source
                of off-by-one bugs in vendor wrappers.
        force_refresh: if True, ignore the cache and re-download.

    Returns:
        DataFrame indexed by date with at least the columns:
            ['open', 'high', 'low', 'close', 'volume']
        all lowercase. yfinance returns 'Open'/'Close' (title case) and
        an 'Adj Close' column; we normalise here so downstream code
        doesn't have to know about that.

        IMPORTANT: the 'close' column we return is the ADJUSTED close
        (split- and dividend-adjusted). For a backtest of a long/short
        equity strategy this is the right convention — total return is
        what the strategy earns. If you ever need the raw unadjusted
        close (e.g. for option strike comparison), pull it explicitly
        from yfinance directly; do not extend this loader by adding a
        flag, because that flag will eventually get misused and pollute
        a backtest.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(ticker, start, end)

    if cache_file.exists() and not force_refresh:
        # Read straight from disk. parquet preserves dtypes (including
        # the DatetimeIndex), so this round-trips cleanly.
        return pd.read_parquet(cache_file)

    # yfinance treats `end` as exclusive; we want inclusive semantics
    # at our API boundary because that's what every other date-range
    # tool in pandas/numpy uses. Bump by one day to compensate.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        ticker,
        start=start,
        end=end_exclusive,
        # auto_adjust=True replaces the 'Close' column with the adjusted
        # close (so 'Adj Close' is gone). This is what we want — every
        # downstream consumer uses 'close' as the total-return price.
        auto_adjust=True,
        # progress=False keeps logs clean when called inside a loop
        # over many strategies.
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker} {start}..{end}. "
            f"Check the ticker spelling and that the date range "
            f"intersects the security's trading history."
        )

    # yfinance may return a MultiIndex on columns when multiple tickers
    # are requested. We always request one ticker here, but be defensive
    # in case yfinance changes its mind in a future version.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Lower-case the column names so callers don't have to care about
    # yfinance's TitleCase convention.
    raw.columns = [str(c).lower() for c in raw.columns]

    # Some yfinance versions name the adjusted close 'close' (when
    # auto_adjust=True) and some still surface 'adj close'. Reconcile.
    if "close" not in raw.columns and "adj close" in raw.columns:
        raw = raw.rename(columns={"adj close": "close"})

    # Drop rows where 'close' is NaN — happens at the very start/end of
    # the requested window when the exchange was closed. Strategies
    # cannot do anything sensible with a NaN price.
    raw = raw.dropna(subset=["close"])

    # Sort just in case yfinance ever returns out-of-order rows. Cheap
    # insurance; .sort_index() is O(n log n) on already-sorted input.
    raw = raw.sort_index()

    # Persist to the cache so subsequent runs are deterministic.
    raw.to_parquet(cache_file)

    return raw


def clear_cache(ticker: str | None = None) -> int:
    """Delete cached price files.

    Args:
        ticker: if given, only delete cache files for this ticker.
                If None, wipe the whole cache.

    Returns:
        Number of files deleted (mostly useful for tests).
    """
    if not _CACHE_DIR.exists():
        return 0
    deleted = 0
    for path in _CACHE_DIR.glob("*.parquet"):
        if ticker is None or path.name.startswith(
            ticker.replace("^", "_idx_").replace("/", "_") + "__"
        ):
            os.remove(path)
            deleted += 1
    return deleted
