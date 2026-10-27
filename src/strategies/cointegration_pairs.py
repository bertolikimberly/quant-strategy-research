"""Cointegration pairs trading strategy.

WHAT THIS STRATEGY DOES, IN PROSE
---------------------------------
1. Define a multi-asset universe spanning equity sectors, oil-related
   names, gold-related names, and rate-sensitive names. The point of
   the spread is that genuine cointegration is rare; a strategy that
   "discovers" pairs only from one sector is at the highest risk of
   simply rediscovering sector beta.
2. On the IN-SAMPLE portion of the data only, run an Engle-Granger
   cointegration test on every candidate pair (N choose 2).
3. Apply a Benjamini-Hochberg FDR correction across all those tests
   to control the expected proportion of false discoveries. This is
   the core defence against data-snooping: with 136 raw tests at
   alpha=0.05 we'd expect ~7 false positives even if NOTHING is truly
   cointegrated. BH-FDR at q=0.10 caps the expected fraction of false
   positives among the rejected hypotheses.
4. For each pair that survives FDR, fit a one-step Ornstein-Uhlenbeck
   process to the spread residual and compute the mean-reversion
   half-life. Reject pairs with half-life outside [1, 30] days: too
   short usually means microstructure noise dominating; too long
   means the pair can't realistically be traded inside a 25-day fund
   horizon.
5. For each surviving pair, the trading rule is a textbook z-score
   reversion strategy on the spread:
      enter long-cheap / short-rich when |z| > 2,
      exit when z crosses 0 (thesis realized) or |z| > 3 (stopped out).
   Sizing is equal-dollar-neutral per pair, and capital is allocated
   equally across all surviving pairs (each gets 1/N of capital).
6. Pair-leg trade costs and short-leg borrow costs are charged INSIDE
   this module (the framework's engine sees only the strategy NAV, so
   it can't see the leg-level activity). We use the SAME cost
   parameters as the framework default, so the comparison stays fair.

HOW THIS FITS THE FRAMEWORK'S Strategy INTERFACE
------------------------------------------------
The Strategy ABC expects a single-asset 'close' column. We adapt by:
  - precomputing the strategy's daily NET returns inside prepare_data,
  - building a synthetic 'close' = cumprod(1 + net_returns),
  - returning that as the 'close' column.
Then generate_positions returns 1.0 always — "stay invested in the
strategy". The engine reconstructs the strategy return as
pct_change(close), which equals our pre-computed net_return up to a
one-day shift. That one-day shift means the engine ignores day 0's
return (we have none anyway — we haven't traded yet) and charges one
tiny initial "entry" cost (~3.5 bp on a 1.0 unit position). Both of
these are second-order distortions over a multi-year backtest.

The strategy-specific null (block bootstrap of the spread series) is
implemented as the `bootstrap_null` method below. run_bakeoff prefers
that method over the framework's default permutation_test for any
strategy that exposes it.

HONEST LIMITATIONS WORTH STATING
--------------------------------
- We fit beta ONCE on the IS data and use it for the entire history.
  A real production system would re-fit beta walk-forward. The fixed-
  beta simplification probably HURTS the strategy in regime changes;
  it does NOT inflate results.
- Survivor-bias-free: we use a fixed universe defined ahead of time
  (no "pick today's top performers"), so survivorship bias at the
  TICKER level is minimal. There is still selection bias at the PAIR
  level (we only trade pairs that look cointegrated in-sample), which
  is exactly why the OOS scorecard is the honest one.
- The block bootstrap approximates pair P&L as state * d(spread).
  This is correct when beta = 1; for beta far from 1 it diverges
  from the equal-dollar-neutral P&L we actually trade. For typical
  betas in [0.5, 2] the approximation is fine for testing the SHAPE
  of the null distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

from src.data.loader import get_prices
from src.engine.costs import CostModel, DEFAULT_COST_MODEL, TRADING_DAYS_PER_YEAR
from src.metrics.scorecard import sharpe
from src.strategies.base import Strategy


# --------------------------------------------------------------------------
# Pre-specified, theory-driven preset: "economic substitute" pairs.
# Used as a SECOND test on a different, smaller hypothesis space when the
# broad-screen universe yields zero survivors. Pre-specifying which pairs
# are theoretically motivated BEFORE looking at the data is what makes
# this a single additional hypothesis test (with N≈9 hypotheses) rather
# than another data-mining sweep.
# --------------------------------------------------------------------------

SUBSTITUTE_PAIRS: list[tuple[str, str]] = [
    ("KO",   "PEP"),   # carbonated-beverages duopoly
    ("V",    "MA"),    # global payment-network duopoly
    ("GS",   "MS"),    # bulge-bracket investment banks
    ("JPM",  "BAC"),   # US universal banks
    ("HD",   "LOW"),   # home-improvement retail duopoly
    ("XOM",  "CVX"),   # US integrated oil majors
    ("GM",   "F"),     # US auto-OEM pair
    ("UPS",  "FDX"),   # US parcel-delivery duopoly
    ("COST", "WMT"),   # mass-retail substitutes (warehouse + general)
]

SUBSTITUTE_UNIVERSE_TAGS: dict[str, str] = {
    "KO":   "Beverages",
    "PEP":  "Beverages",
    "V":    "Payment networks",
    "MA":   "Payment networks",
    "GS":   "Investment banks",
    "MS":   "Investment banks",
    "JPM":  "Universal banks",
    "BAC":  "Universal banks",
    "HD":   "Home improvement retail",
    "LOW":  "Home improvement retail",
    "XOM":  "Oil major (integrated)",
    "CVX":  "Oil major (integrated)",
    "GM":   "US automaker",
    "F":    "US automaker",
    "UPS":  "Parcel delivery",
    "FDX":  "Parcel delivery",
    "COST": "Mass retail (warehouse)",
    "WMT":  "Mass retail (general)",
}


# Default candidate universe. Mapping is ticker -> short sector tag.
# The sector tag is used purely for annotation in the screen CSV so the
# user can see WHICH cross-sector pairs survived. The cointegration
# test does not use the tag — it's transparent metadata.
#
# Universe composition (per the brief):
#   - 8 sectorally-diverse US large-caps
#   - 3 oil-related (2 majors + a direct-oil ETF)
#   - 3 gold-related (the metal ETF, the miners ETF, one individual miner)
#   - 3 rate-sensitive (one defensive, one cyclical, one pure-rates)
# Total: 17 tickers → C(17,2) = 136 candidate pairs.
DEFAULT_UNIVERSE: dict[str, str] = {
    # US large-cap equities, sectorally spread
    "AAPL": "Tech",
    "MSFT": "Tech",
    "JPM":  "Financials",
    "JNJ":  "Healthcare",
    "PG":   "Consumer Staples",
    "KO":   "Consumer Staples",
    "CAT":  "Industrials",
    "VZ":   "Telecom",
    # Oil-related
    "XOM":  "Oil major",
    "CVX":  "Oil major",
    "USO":  "Oil ETF",
    # Gold / precious metals
    "GLD":  "Gold ETF",
    "GDX":  "Gold miners ETF",
    "NEM":  "Gold miner (single name)",
    # Rate-sensitive
    "XLU":  "Utilities (rate-defensive)",
    "XLF":  "Financials sector ETF (rate-sensitive)",
    "TLT":  "Long Treasury ETF (pure-rate)",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _engle_granger(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float, pd.Series]:
    """Engle-Granger two-step on log prices.

    Returns (beta, pvalue, residuals) where:
      beta:     OLS slope of log_a on log_b (the hedge ratio).
      pvalue:   p-value from statsmodels.tsa.stattools.coint, which
                uses MacKinnon critical values appropriate for testing
                cointegration (NOT a plain ADF on the residuals — that
                would understate p-values because residuals are
                estimated, not observed).
      residuals: the OLS residuals, indexed like log_a. Used downstream
                 for the OU half-life fit and for trading.

    Why log prices: cointegration in log space corresponds to a
    long-run multiplicative relationship between the two price series,
    which is what economic intuition suggests for equity pairs (e.g.
    "Coke trades at a roughly constant ratio to Pepsi over the long
    run" — that's a relation in log space, not in level space).
    """
    # OLS regression: log_a = alpha + beta * log_b + epsilon
    X = sm.add_constant(log_b.values)
    reg = sm.OLS(log_a.values, X).fit()
    beta = float(reg.params[1])
    residuals = pd.Series(reg.resid, index=log_a.index)

    # statsmodels' coint() function does the EG test with the right
    # critical values. It re-runs the regression internally, so we
    # do a small amount of redundant work — fine for ~140 pairs.
    # Returns (test_stat, pvalue, critical_values_dict).
    _, pvalue, _ = coint(log_a.values, log_b.values)
    return beta, float(pvalue), residuals


def _ou_half_life(residuals: pd.Series) -> float:
    """Estimate the OU mean-reversion half-life from residuals (in days).

    Continuous-time OU: dX = -kappa * (X - mu) * dt + sigma * dW
    Discrete one-day estimator: regress ΔX on X_{t-1}:
        ΔX_t = -lambda * X_{t-1} + noise
    Then half-life = ln(2) / lambda.

    Returns float('inf') if the regression suggests no mean reversion
    (lambda <= 0) or if there isn't enough data. Returning a large
    finite number would still satisfy "< 30" filters with the wrong
    semantics; +inf is the unambiguous "fail this filter" sentinel.
    """
    s = residuals.dropna()
    if len(s) < 30:
        return float("inf")
    # Center the residuals — OU is a process around zero. Centering is
    # cosmetic here because OLS without an intercept on a zero-mean
    # series gives the same lambda as OLS with intercept on the raw
    # series, but it makes the algebra match the textbook formula.
    s_centered = s - s.mean()
    delta = s_centered.diff().dropna()
    lagged = s_centered.shift(1).dropna().reindex(delta.index)
    if len(delta) < 30:
        return float("inf")
    # OLS without constant: delta = -lambda * lagged + noise.
    # Use np.linalg.lstsq for speed (no statsmodels overhead needed).
    X = lagged.values.reshape(-1, 1)
    y = delta.values
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    lam = -float(coef[0])
    if lam <= 0:
        # No mean reversion in this sample (or actually divergent).
        return float("inf")
    return float(np.log(2) / lam)


def _pair_state_series(
    z: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float,
) -> pd.Series:
    """Run the entry / exit / stop state machine on a pair's z-score.

    State values:
       0 = flat
      +1 = long the spread  (long A, short B; the spread is "cheap" below mean)
      -1 = short the spread (short A, long B; the spread is "rich" above mean)

    Transitions (decisions are made at close of T; framework's engine
    lags positions by one day before applying them to T+1 returns,
    so this loop is causal):
        flat → +1     if z < -entry_z
        flat → -1     if z > +entry_z
        +1   → 0      if z >= exit_z (mean reversion realized)
        +1   → 0      if z < -stop_z (blew through, thesis broken)
        -1   → 0      if z <= -exit_z
        -1   → 0      if z > +stop_z

    We use a Python loop rather than vectorise because the state has
    persistence (today's position depends on yesterday's), which is
    exactly the case where a loop is clearer than vectorised pandas.
    For ~3700 daily obs per pair × ~10 pairs the loop runtime is
    a few tens of milliseconds — not a bottleneck.
    """
    out = np.zeros(len(z), dtype=float)
    state = 0
    z_values = z.values
    for i in range(len(z_values)):
        zi = z_values[i]
        if np.isnan(zi):
            # No signal — must stay flat. Importantly, we DON'T carry
            # forward a previous state across NaN: NaN means we can't
            # evaluate the rule, so the safe default is "flat".
            state = 0
            out[i] = 0.0
            continue
        if state == 0:
            if zi < -entry_z:
                state = +1
            elif zi > entry_z:
                state = -1
        elif state == +1:
            if zi >= exit_z or zi < -stop_z:
                state = 0
        elif state == -1:
            if zi <= -exit_z or zi > stop_z:
                state = 0
        out[i] = float(state)
    return pd.Series(out, index=z.index, name="pair_state")


def _block_bootstrap_indices(
    n: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate index array of length `n` via non-overlapping circular block bootstrap.

    Pick floor(n / block_size) + 1 starting points uniformly from
    [0, n), then concatenate the block of `block_size` consecutive
    indices (wrapping around the end) from each. Truncate to length n.

    "Circular" means we wrap at the end so all blocks have equal
    probability of being sampled — without wrapping, the last few
    indices would be under-sampled.
    """
    n_blocks = (n // block_size) + 1
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([
        (np.arange(block_size) + s) % n for s in starts
    ])
    return idx[:n]


# ---------------------------------------------------------------------------
# Per-pair record + main Strategy class
# ---------------------------------------------------------------------------


@dataclass
class _PairInfo:
    """Everything we need to know about ONE surviving pair to trade it.

    Stored as instance state on the strategy so prepare_data and
    bootstrap_null can both access it without recomputing.
    """
    ticker_a: str
    ticker_b: str
    beta: float
    eg_pvalue: float
    fdr_adjusted_pvalue: float
    half_life: float
    spread: pd.Series          # log_a - beta * log_b, full-history
    state: pd.Series           # ±1/0 position series, full-history
    pair_gross_return: pd.Series   # daily return per unit of pair capital


class CointegrationPairs(Strategy):
    """Multi-pair, equal-dollar-neutral cointegration strategy.

    Constructor parameters all carry the bake-off's defaults; override
    them at the call site if you want to run sensitivity studies, but
    the bake-off run uses the defaults so every candidate strategy
    sees the same hyperparameters.

    Args:
        universe: dict mapping ticker -> sector tag. Defaults to
            DEFAULT_UNIVERSE defined at the top of this module.
        train_ratio: fraction of the data window used for pair
            selection. Matches the framework's IS/OOS split so that
            "in-sample" really is "what the strategy saw at fit time".
        fdr_q: BH-FDR q-level. 0.10 means "control the expected
            proportion of false discoveries among rejections to 10%".
        zscore_window: rolling window for the spread z-score in days.
        entry_z, exit_z, stop_z: trading rule thresholds.
        min_half_life, max_half_life: OU half-life filter, in days.
        cost_model: cost assumptions; defaults to DEFAULT_COST_MODEL so
            the cointegration strategy is charged the same way as the
            other candidates.
        screen_csv_path: where to write the pair-screen output. None
            means don't write.
    """

    def __init__(
        self,
        universe: dict[str, str] | None = None,
        *,
        candidate_pairs: list[tuple[str, str]] | None = None,
        universe_label: str = "broad",
        train_ratio: float = 0.70,
        fdr_q: float = 0.10,
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        stop_z: float = 3.0,
        min_half_life: float = 1.0,
        max_half_life: float = 30.0,
        cost_model: CostModel = DEFAULT_COST_MODEL,
        screen_csv_path: str | Path | None = None,
    ) -> None:
        # `candidate_pairs` lets the caller pre-specify exactly which
        # pairs to test, instead of letting the strategy enumerate
        # C(N, 2) from the universe. This is essential for HONEST
        # theory-driven tests on small universes: with 9 hypotheses,
        # BH-FDR at q=0.10 needs the smallest raw p <= 0.011 to start
        # rejecting; with 136 hypotheses, the smallest raw p must be
        # <= 0.00074. Pre-specifying which 9 pairs you cared about
        # BEFORE seeing the data is exactly the assumption that
        # converts the 136-test universe into the 9-test universe.
        #
        # The first element of each tuple becomes the dependent var
        # in the Engle-Granger regression (beta is from log(a) on
        # log(b)). Order is preserved as the caller supplies it.
        # Duplicate pairs (regardless of order) are deduped here so
        # the FDR denominator equals the real test count.
        if candidate_pairs is not None:
            seen: set[frozenset[str]] = set()
            deduped: list[tuple[str, str]] = []
            for a, b in candidate_pairs:
                key = frozenset((a, b))
                if a == b or key in seen:
                    continue
                seen.add(key)
                deduped.append((a, b))
            self.candidate_pairs: list[tuple[str, str]] | None = deduped
            # Universe of tickers needed = the union of the pair list.
            # If the caller passed an explicit `universe` dict (with
            # sector tags), keep only the tickers we actually need; if
            # they didn't, build a tag-less dict from the pairs.
            needed = {t for pair in deduped for t in pair}
            if universe is None:
                self.universe = {t: "" for t in sorted(needed)}
            else:
                self.universe = {
                    k: v for k, v in universe.items() if k in needed
                }
                missing = needed - set(self.universe)
                if missing:
                    # Caller passed a universe dict but it doesn't cover
                    # every ticker in candidate_pairs. Add empty-tag
                    # entries so prepare_data can still pull the prices.
                    for t in sorted(missing):
                        self.universe[t] = ""
        else:
            self.candidate_pairs = None
            self.universe = dict(universe or DEFAULT_UNIVERSE)

        self.universe_label = universe_label
        self.train_ratio = train_ratio
        self.fdr_q = fdr_q
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.cost_model = cost_model
        self.screen_csv_path = (
            Path(screen_csv_path) if screen_csv_path is not None else None
        )

        # Bake every consequential hyperparameter (including the
        # universe label) into the name so two variants on different
        # universes don't collide in the scorecard / diagnostics CSV.
        self.name = (
            f"coint_pairs_{universe_label}"
            f"_q{int(fdr_q*100):02d}"
            f"_e{int(entry_z*10)}_s{int(stop_z*10)}"
            f"_hl{int(min_half_life)}-{int(max_half_life)}"
        )

        # Lazy / cached state filled in by prepare_data:
        self._cached_data: pd.DataFrame | None = None
        self._screen_df: pd.DataFrame | None = None
        self._pairs: list[_PairInfo] = []
        self._n_pairs: int = 0

    # ---- Strategy interface methods ---------------------------------------

    def prepare_data(self, start: str, end: str) -> pd.DataFrame:
        # Use a fast path on repeat calls: prepare_data is deterministic
        # given (start, end), and the cointegration screen is the
        # expensive step. Cache the result.
        if self._cached_data is not None:
            return self._cached_data

        # 1. Load all tickers and align to common trading days.
        # Inner-join on dates: only days where EVERY universe member
        # has a price (otherwise cointegration tests would be biased
        # by missing data).
        per_ticker = {}
        for ticker in self.universe:
            df = get_prices(ticker, start, end)
            per_ticker[ticker] = df["close"]
        prices = pd.concat(per_ticker, axis=1).dropna()
        # `prices.columns` is now a list of tickers; rows are trading days.

        if len(prices) < 252:
            raise ValueError(
                f"After aligning the universe on common dates, only "
                f"{len(prices)} rows remain — not enough for a "
                f"meaningful cointegration test."
            )

        # 2. Split into IS / OOS for pair selection. We do the cointegration
        # screen on the IS slice ONLY — this is the whole reason the
        # OOS scorecard is honest later.
        cut = int(len(prices) * self.train_ratio)
        is_prices = prices.iloc[:cut]

        # 3. Run Engle-Granger on every pair.
        log_is = np.log(is_prices)
        log_full = np.log(prices)  # used for trading after screen
        screen_rows = self._screen_pairs(log_is, log_full)
        screen_df = pd.DataFrame(screen_rows)

        # 4. BH-FDR across all raw p-values. The number of tests is
        # len(screen_df) — every candidate pair contributes one test.
        if not screen_df.empty:
            reject, p_adj, _, _ = multipletests(
                screen_df["eg_pvalue"].values,
                alpha=self.fdr_q,
                method="fdr_bh",
            )
            screen_df["fdr_adjusted_pvalue"] = p_adj
            screen_df["survived_fdr"] = reject
        else:
            screen_df["fdr_adjusted_pvalue"] = []
            screen_df["survived_fdr"] = []

        # 5. Half-life filter.
        screen_df["survived_half_life"] = (
            (screen_df["half_life"] >= self.min_half_life)
            & (screen_df["half_life"] <= self.max_half_life)
        )
        screen_df["survived_all"] = (
            screen_df["survived_fdr"] & screen_df["survived_half_life"]
        )
        # Sort for human readability of the CSV: surviving pairs first
        # (by p-value), then non-survivors by raw p-value.
        screen_df = screen_df.sort_values(
            by=["survived_all", "eg_pvalue"],
            ascending=[False, True],
        ).reset_index(drop=True)

        # Persist the screen output if requested.
        if self.screen_csv_path is not None:
            self.screen_csv_path.parent.mkdir(parents=True, exist_ok=True)
            screen_df.to_csv(self.screen_csv_path, index=False)
        self._screen_df = screen_df

        survivors = screen_df[screen_df["survived_all"]].copy()
        print(
            f"  [coint_pairs] tested {len(screen_df)} pairs; "
            f"{int(screen_df['survived_fdr'].sum())} survived FDR (q={self.fdr_q}); "
            f"{len(survivors)} also passed half-life filter "
            f"[{self.min_half_life}, {self.max_half_life}] days.",
            flush=True,
        )

        # 6. If nothing survived, return a flat synthetic asset so the
        # framework can still produce a (zero-everywhere) scorecard.
        # This is an honest "no signal" result — distinguishable from a
        # losing strategy by Sharpe = NaN and zero turnover.
        if survivors.empty:
            flat = pd.DataFrame(
                {"close": np.ones(len(prices))},
                index=prices.index,
            )
            self._cached_data = flat
            return flat

        # 7. For each surviving pair, build the trading state and
        # compute the GROSS daily pair return over the full history.
        self._pairs = []
        for _, row in survivors.iterrows():
            a, b = row["ticker_a"], row["ticker_b"]
            beta = float(row["beta"])
            # Spread over FULL history (using IS-fitted beta — no walk-
            # forward refit; honest limitation noted in module docstring).
            spread = log_full[a] - beta * log_full[b]
            mu = spread.rolling(self.zscore_window).mean()
            sd = spread.rolling(self.zscore_window).std(ddof=1)
            z = (spread - mu) / sd

            state = _pair_state_series(
                z, self.entry_z, self.exit_z, self.stop_z
            )

            # Gross pair return (per unit of pair capital, before costs).
            # Equal-dollar-neutral sizing means each leg is 1.0 of pair
            # capital:
            #   state=+1: long A (+1*r_A), short B (-1*r_B)
            #     → return = r_A - r_B
            #   state=-1: short A (-1*r_A), long B (+1*r_B)
            #     → return = r_B - r_A = -(r_A - r_B)
            # Per-pair return is therefore state * (r_A - r_B).
            # We .shift(1) to apply yesterday's decision to today's
            # return — matches the engine's lookahead protection.
            r_a = prices[a].pct_change().fillna(0.0)
            r_b = prices[b].pct_change().fillna(0.0)
            pair_gross = state.shift(1).fillna(0.0) * (r_a - r_b)

            self._pairs.append(_PairInfo(
                ticker_a=a, ticker_b=b, beta=beta,
                eg_pvalue=float(row["eg_pvalue"]),
                fdr_adjusted_pvalue=float(row["fdr_adjusted_pvalue"]),
                half_life=float(row["half_life"]),
                spread=spread,
                state=state,
                pair_gross_return=pair_gross,
            ))
        self._n_pairs = len(self._pairs)

        # 8. Aggregate to portfolio level. Equal-weight across pairs:
        # each pair gets 1/N of capital. Daily portfolio gross return
        # is the equally-weighted average of pair returns.
        portfolio_gross = pd.Series(0.0, index=prices.index)
        portfolio_trade_cost = pd.Series(0.0, index=prices.index)
        portfolio_borrow_cost = pd.Series(0.0, index=prices.index)
        per_unit_cost = self.cost_model.per_unit_trade_cost()
        daily_borrow = self.cost_model.daily_financing_rate()

        for p in self._pairs:
            weight = 1.0 / self._n_pairs
            portfolio_gross = portfolio_gross + weight * p.pair_gross_return

            # Trade cost: when state changes from a to b, each leg's
            # notional changes. The TOTAL leg turnover per pair is
            # 2 * |Δstate| * (pair capital), because both legs move
            # by the same magnitude (equal-dollar sizing). E.g., a
            # flat→+1 entry trades both legs from 0 → ±1 unit = 2
            # units of leg turnover per pair capital unit. A +1→-1
            # flip trades both legs by 2 units = 4 units total.
            delta_state = p.state.diff().abs().fillna(p.state.abs())
            leg_turnover_per_pair_capital = 2.0 * delta_state
            # Trade cost is per-unit on the LEG turnover, scaled by
            # this pair's portfolio weight.
            portfolio_trade_cost = (
                portfolio_trade_cost
                + weight * leg_turnover_per_pair_capital * per_unit_cost
            )

            # Borrow cost: every day a pair is in ±1 state, the SHORT
            # leg pays the daily borrow rate on 1 unit of pair capital
            # (the short notional). We bill it on day T because that's
            # when the position is actually held overnight.
            short_notional = (p.state != 0).astype(float)
            portfolio_borrow_cost = (
                portfolio_borrow_cost
                + weight * short_notional * daily_borrow
            )

        portfolio_net = portfolio_gross - portfolio_trade_cost - portfolio_borrow_cost

        # 9. Synthetic close = NAV path. Engine will pct_change this
        # back into per-day returns, which equals portfolio_net (up to
        # the engine's one-day lag for position application).
        synthetic_close = (1.0 + portfolio_net.fillna(0.0)).cumprod()

        out = pd.DataFrame(
            {"close": synthetic_close},
            index=prices.index,
        )
        self._cached_data = out
        return out

    def generate_positions(self, data: pd.DataFrame) -> pd.Series:
        # Position is constant 1.0 — "always invested in the pairs
        # portfolio". The actual time-varying allocation across
        # individual pairs lives inside the synthetic close.
        # Note: if no pairs survived, the synthetic close is constant
        # 1.0 → asset_return ≡ 0, so positions of 1.0 still produce a
        # zero return series (the honest "no signal" outcome).
        return pd.Series(1.0, index=data.index, name="position")

    # ---- Helpers used internally ------------------------------------------

    def _screen_pairs(
        self,
        log_is: pd.DataFrame,
        log_full: pd.DataFrame,
    ) -> list[dict]:
        """Run Engle-Granger + OU half-life on every candidate pair.

        Returns a list of dicts (one per pair), unsorted. The caller
        adds BH-FDR adjustments and writes the CSV.

        Pair source:
          - If `candidate_pairs` was passed to the constructor, we test
            EXACTLY those pairs (in the caller's order). This is the
            honest path for theory-driven tests on small universes.
          - Otherwise we enumerate every C(N, 2) pair from the
            universe — the standard "screen everything" path.

        We use IS data for the cointegration test AND for the OU
        half-life (the half-life is a property we'd estimate at fit
        time, before trading). The trading itself runs over full
        history using the IS-estimated beta.
        """
        if self.candidate_pairs is not None:
            pair_iter: list[tuple[str, str]] | object = self.candidate_pairs
        else:
            pair_iter = list(combinations(log_is.columns, 2))

        rows = []
        for a, b in pair_iter:
            beta, pvalue, residuals = _engle_granger(log_is[a], log_is[b])
            half_life = _ou_half_life(residuals)
            rows.append({
                "ticker_a": a,
                "sector_a": self.universe.get(a, ""),
                "ticker_b": b,
                "sector_b": self.universe.get(b, ""),
                "beta": beta,
                "eg_pvalue": pvalue,
                "half_life": half_life,
            })
        return rows

    # ---- Strategy-specific null -------------------------------------------

    def bootstrap_null(
        self,
        n_bootstraps: int = 500,
        block_size: int = 20,
        seed: int = 42,
    ) -> "_BootstrapResult":
        """Block-bootstrap null distribution of portfolio Sharpe.

        Why a custom null instead of the framework's permutation_test:
            The framework's default test shuffles the strategy's
            position series. For this strategy, the positions exposed
            to the engine are constant 1.0 — shuffling them is a
            no-op and the test returns the degenerate "everything
            equals observed" result. The TIMING question for a pairs
            strategy lives at the SPREAD level, not the meta-position
            level. So we resample the spreads.

        Procedure:
            For each bootstrap iteration:
              1. For each surviving pair:
                 - Block-bootstrap the spread series (length-preserving,
                   circular blocks of `block_size` days). This
                   preserves the short-range autocorrelation in the
                   spread — essential, because mean reversion IS
                   autocorrelation.
                 - Recompute the rolling z-score and the trading state.
                 - Approximate the pair return as state * d(spread).
                   This is exact when the pair's beta = 1 and a close
                   approximation otherwise; see module docstring.
              2. Equally weight across pairs (same 1/N as live).
              3. Compute Sharpe of the resulting bootstrap return series.
            p-value = fraction of bootstrap Sharpes >= observed Sharpe.

        Caveats:
            - We compute a GROSS-of-cost Sharpe in both the observed
              and the null. Costs depend on turnover, which depends on
              timing, so including them would conflate "edge in
              timing" with "cost sensitivity to timing". The null is
              cleaner gross-of-cost.
            - "Observed" here is the strategy's actual gross Sharpe
              over the same full window, NOT the engine's net Sharpe.
              The p-value answers: "is the gross-of-cost return higher
              than what mean-reverting noise alone would generate?"
              A small p-value is a necessary but not sufficient
              condition for the net strategy to be profitable.

        Returns:
            _BootstrapResult with the observed (gross) Sharpe, the
            full null distribution, and the one-sided p-value.
        """
        if self._cached_data is None:
            raise RuntimeError(
                "bootstrap_null requires prepare_data to have been "
                "called first (so the surviving pairs are known)."
            )
        if not self._pairs:
            # No pairs survived; bootstrap is undefined.
            return _BootstrapResult(
                observed_gross_sharpe=float("nan"),
                null_sharpes=np.array([]),
                p_value=float("nan"),
                n_bootstraps=0,
                block_size=block_size,
                n_surviving_pairs=0,
            )

        rng = np.random.default_rng(seed)
        # Observed GROSS Sharpe: equal-weighted pair gross returns.
        gross_portfolio = sum(
            (1.0 / self._n_pairs) * p.pair_gross_return for p in self._pairs
        )
        observed = sharpe(gross_portfolio)

        null = np.empty(n_bootstraps, dtype=float)
        n = len(gross_portfolio)

        for i in range(n_bootstraps):
            boot_returns = pd.Series(0.0, index=gross_portfolio.index)
            for p in self._pairs:
                idx = _block_bootstrap_indices(n, block_size, rng)
                # Bootstrap the spread itself. We keep the *time index*
                # the original spread had so the rolling z-score window
                # is comparable to live; only the VALUES are resampled.
                boot_spread = pd.Series(
                    p.spread.values[idx],
                    index=p.spread.index,
                )
                mu = boot_spread.rolling(self.zscore_window).mean()
                sd = boot_spread.rolling(self.zscore_window).std(ddof=1)
                z = (boot_spread - mu) / sd
                state = _pair_state_series(
                    z, self.entry_z, self.exit_z, self.stop_z
                )
                # Pair return approximation: state * d(spread).
                # This is exact if beta = 1 (then spread = log_a -
                # log_b and d(spread) = r_a - r_b for small returns);
                # for beta far from 1, it diverges by a factor that
                # doesn't depend on TIMING, so the null distribution
                # SHAPE is still informative.
                d_spread = boot_spread.diff().fillna(0.0)
                pair_ret = state.shift(1).fillna(0.0) * d_spread
                boot_returns = boot_returns + (1.0 / self._n_pairs) * pair_ret
            null[i] = sharpe(boot_returns)

        if np.isnan(observed):
            p_value = float("nan")
        else:
            p_value = float((null >= observed).mean())

        return _BootstrapResult(
            observed_gross_sharpe=observed,
            null_sharpes=null,
            p_value=p_value,
            n_bootstraps=n_bootstraps,
            block_size=block_size,
            n_surviving_pairs=self._n_pairs,
        )


@dataclass
class _BootstrapResult:
    """Output of the block-bootstrap null. Public surface for printing.

    Lives next to the strategy class because it's specific to this
    strategy's null. If we ever generalise the bootstrap into the
    framework's validation module, promote this into validation/.
    """
    observed_gross_sharpe: float
    null_sharpes: np.ndarray
    p_value: float
    n_bootstraps: int
    block_size: int
    n_surviving_pairs: int

    def __str__(self) -> str:
        if self.n_bootstraps == 0:
            return (
                f"_BootstrapResult(n_surviving_pairs=0; "
                f"no test possible)"
            )
        return (
            f"_BootstrapResult("
            f"obs_gross_sharpe={self.observed_gross_sharpe:.3f}, "
            f"null_mean={self.null_sharpes.mean():.3f}, "
            f"null_p95={np.percentile(self.null_sharpes, 95):.3f}, "
            f"p={self.p_value:.4f}, "
            f"n_boot={self.n_bootstraps}, "
            f"block={self.block_size}, "
            f"pairs={self.n_surviving_pairs})"
        )
