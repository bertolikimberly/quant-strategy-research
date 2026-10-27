"""Permutation test on signal/event timing.

THE QUESTION THE TEST ANSWERS
-----------------------------
"Could the strategy's observed Sharpe have plausibly arisen just from
random timing of trades on this asset, given the strategy's
distribution of positions (long/short/flat)?"

The null hypothesis is "the timing of this strategy's positions is
uninformative about future returns". Under that null, if we keep the
SAME set of positions (same number of long days, short days, flat
days, same magnitudes) but RESHUFFLE the order they're applied to the
underlying's returns, the observed Sharpe should be statistically
indistinguishable from the resulting null distribution.

The procedure:

    1. Run the strategy through the engine once to get the observed
       Sharpe S_obs and the position series.
    2. For each of N permutations:
         a. Randomly shuffle the strategy's RAW position series in
            time. (We shuffle the raw positions, not the lagged ones,
            so the engine still applies the lookahead-protection
            .shift(1) in the rebuild step. We do NOT shuffle the
            already-lagged series — it doesn't matter mathematically
            because shuffling commutes with the shift after warm-up,
            but it's cleaner to keep the engine in charge of all PnL
            accounting.)
         b. Recompute the net return series with the shuffled positions
            (against the SAME asset returns, the SAME cost model).
         c. Compute the permutation's Sharpe.
    3. p-value = fraction of permutation Sharpes >= S_obs.
       (One-sided: we only care if the strategy is better than chance.
       For a strategy that's WORSE than chance — negative observed
       Sharpe — the same procedure still works; the p-value will be
       close to 1.)

WHY THIS, AND NOT A BOOTSTRAP / MONTE-CARLO ALTERNATIVE
-------------------------------------------------------
Many alternatives exist:
  - Bootstrap the returns themselves. Tests "is the mean return
    different from zero", but does NOT test signal informativeness
    specifically (a buy-and-hold of a rising asset has a great
    bootstrap p-value but no skill).
  - Simulate random strategies (random uniform positions). Tests
    against a generic random benchmark but doesn't account for the
    strategy's own position distribution. A strategy that's mostly
    flat will be unfairly compared to one that's mostly fully long.
  - Block bootstrap on returns. Preserves return autocorrelation but
    still doesn't isolate the timing question.

The shuffle-positions test isolates THE thing we want to know: does
the strategy's timing have edge, conditional on the marginal mix of
positions it happens to take? This is the right null for a
discretionary-style strategy bake-off.

CAVEATS WE OWE THE READER
-------------------------
1. Shuffling positions destroys any autocorrelation in the position
   series (a trend strategy's permuted "trend" no longer trends).
   This is intentional — the test asks about edge in TIMING, and
   timing IS what creates autocorrelation. But it means the null
   distribution can have systematically different turnover from the
   observed strategy. We chose this trade-off because preserving
   autocorrelation (e.g. via block shuffle) requires choosing a
   block length, and the block length itself becomes a fudge factor.
2. Multiple testing: running this test on K strategies and reporting
   the minimum p-value WITHOUT a Bonferroni or Benjamini-Hochberg
   correction is intellectually dishonest. The bake-off runner
   surfaces individual p-values; the user must apply the correction
   themselves and disclose it.
3. The asset return series is held fixed across permutations. So
   strategies on different assets are NOT directly comparable on
   p-value (a strategy on a low-vol asset has a different null
   distribution width). Within one bake-off, all strategies should
   trade comparable underliers — or the user should disclose the
   apples-to-oranges issue.
4. DEGENERATE CASE — constant-position strategies:
   If the strategy holds the SAME position every day (e.g. a buy-and-
   hold), then permuting its position series is a no-op: every
   shuffle is identical to the original. The null distribution
   collapses to a single point, observed Sharpe equals it exactly,
   and the test reports p = 1.0. This is mathematically correct —
   a strategy with no timing decisions has no timing edge to test —
   but it's NOT a "failed" test result, it's the test telling you
   the strategy is outside its question's scope. For benchmarks
   like buy-and-hold, treat the p-value column as N/A and judge
   the strategy purely on its Sharpe/Sortino vs the active
   candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.backtest import BacktestResult, run_backtest
from src.engine.costs import CostModel, DEFAULT_COST_MODEL
from src.metrics.scorecard import sharpe
from src.strategies.base import Strategy


@dataclass
class PermutationResult:
    """The output of one permutation test.

    `null_sharpes` is the full N-element distribution so the caller
    can plot a histogram or compute alternative statistics from it
    (e.g. a two-sided p-value, or the 99th percentile as a
    "skill threshold").
    """

    observed_sharpe: float
    null_sharpes: np.ndarray
    p_value: float
    n_permutations: int

    def __str__(self) -> str:
        return (
            f"PermutationResult(observed_sharpe={self.observed_sharpe:.3f}, "
            f"null_mean={self.null_sharpes.mean():.3f}, "
            f"null_p95={np.percentile(self.null_sharpes, 95):.3f}, "
            f"p={self.p_value:.4f}, n={self.n_permutations})"
        )


def _shuffle_series(s: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Return a Series with the same index but values permuted in place.

    We use a fresh permutation each call (no cached state). The
    rng is passed in so the caller controls reproducibility.
    """
    perm = rng.permutation(len(s))
    return pd.Series(s.values[perm], index=s.index, name=s.name)


def permutation_test(
    strategy: Strategy,
    start: str,
    end: str,
    n_permutations: int = 1000,
    cost_model: CostModel = DEFAULT_COST_MODEL,
    seed: int = 42,
    baseline_result: BacktestResult | None = None,
) -> PermutationResult:
    """Run a positions-shuffle permutation test for one strategy.

    Args:
        strategy: the Strategy to test.
        start, end: same as run_backtest — full window for the test.
        n_permutations: number of shuffles. 1000 is a defensible
            default; the minimum achievable p-value with N
            permutations is 1/N, so for "p < 0.01"-style claims
            you want N >= 1000. For the bake-off's quick-look
            report we use 1000; for a final report on the winning
            strategy, bump to 5000-10000.
        cost_model: same cost model used in the headline backtest.
        seed: RNG seed for reproducibility. Each run with the same
            seed gives the exact same null distribution.
        baseline_result: optional, the already-computed BacktestResult
            for the unpermuted strategy. Pass it if you already ran
            the engine once (e.g. inside run_bakeoff) — it saves one
            data-load + position-generation pass. If None, we run
            the engine once internally.

    Returns:
        PermutationResult with the observed Sharpe, the full null
        distribution, and the one-sided p-value.

    Implementation notes:
        - We re-derive the net returns INSIDE the permutation loop
          rather than just shuffling the gross_returns directly. Why?
          Because trade costs and financing costs depend on the
          position series and turn out differently for each shuffle.
          If we shuffled net returns we'd be implicitly assuming
          costs are independent of position pattern, which they
          aren't (a strategy permuted to flip every day would pay
          enormous turnover cost). Doing it the slow-but-honest way
          keeps the null distribution comparable to the observed
          strategy under the SAME cost model.
        - We share the asset_returns Series across permutations
          (it's the same underlying) — no need to re-load it.
    """
    # --- 1. Establish the observed baseline ------------------------------
    if baseline_result is None:
        baseline_result = run_backtest(strategy, start, end, cost_model)
    observed_sharpe = sharpe(baseline_result.net_returns)

    # Pull out the raw (unshifted) positions and the asset returns. We
    # will permute the positions and re-run the post-strategy half of
    # the engine pipeline manually. This is a duplication of logic with
    # engine/backtest.py; we accept it because looping run_backtest()
    # N=1000 times would re-do data prep + signal generation on every
    # iteration (slow, and wasteful since they're deterministic).
    raw_positions = baseline_result.raw_positions
    asset_returns = baseline_result.asset_returns

    rng = np.random.default_rng(seed)
    null_sharpes = np.empty(n_permutations, dtype=float)

    per_unit_cost = cost_model.per_unit_trade_cost()
    daily_financing = cost_model.daily_financing_rate()

    for i in range(n_permutations):
        # Shuffle the raw position series — preserves the multiset of
        # positions, destroys their original timing.
        shuffled = _shuffle_series(raw_positions, rng)

        # Re-apply the SAME lookahead shift the engine uses.
        applied = shuffled.shift(1).fillna(0.0)

        # Gross return under permuted timing.
        gross = applied * asset_returns

        # Turnover and trade cost — first-day full position is a trade.
        delta = shuffled.diff()
        delta.iloc[0] = shuffled.iloc[0]
        trade_cost = (delta.abs() * per_unit_cost).shift(1).fillna(0.0)

        # Financing (gross-of-net leverage and short borrow).
        abs_pos = applied.abs()
        lev = (abs_pos - 1.0).clip(lower=0.0) + (-applied).clip(lower=0.0)
        fin = lev * daily_financing

        net = gross - trade_cost - fin
        null_sharpes[i] = sharpe(net)

    # One-sided p-value: how often did the null beat or tie the
    # observed Sharpe? Tie counted as "as extreme as", which is
    # the standard conservative convention. NaN observed (e.g. an
    # all-zero strategy) collapses to p = 1.
    if np.isnan(observed_sharpe):
        p_value = float("nan")
    else:
        # `>=` is conservative (counts ties as null wins). Add-one
        # smoothing (1 + count) / (1 + n) would prevent p=0 exactly,
        # but we prefer to surface the raw value and let the reader
        # interpret p == 0 as "p < 1/N".
        p_value = float((null_sharpes >= observed_sharpe).mean())

    return PermutationResult(
        observed_sharpe=observed_sharpe,
        null_sharpes=null_sharpes,
        p_value=p_value,
        n_permutations=n_permutations,
    )
