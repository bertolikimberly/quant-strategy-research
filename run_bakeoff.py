"""Top-level bake-off orchestrator.

This script is the one place that knows which strategies are competing.
Add a strategy to the `CANDIDATES` list at the bottom and it picks up
the full evaluation pipeline automatically: backtest, IS/OOS scorecard,
permutation test. Outputs land in `results/`.

EVALUATION PIPELINE (identical for every candidate)
---------------------------------------------------
For each candidate strategy:
    1. Run the engine on the FULL window [START, END] → net_returns.
    2. Split net_returns 70/30 in time → IS slice + OOS slice.
    3. Compute the full scorecard on the FULL series, the IS slice,
       and the OOS slice. Store all three rows.
    4. Run the permutation test on the FULL window (1000 shuffles
       by default).
    5. Save the equity curve to CSV and (if matplotlib is available)
       a PNG.

After all strategies are processed:
    * Concatenate every scorecard row into one comparison CSV.
    * Print the table to stdout sorted by OOS Sharpe (the metric we
      most trust).

WHY ONE WINDOW FOR EVERYBODY
----------------------------
The bake-off is comparison-only, so every strategy MUST see the same
[START, END] window. If a strategy can't operate on that window (e.g.
an ETF that didn't exist before 2010), that's a fact about the
strategy and should disqualify or restrict it. Letting strategies
self-select their own windows would be a stealth form of data dredging.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

from src.engine.backtest import run_backtest
from src.engine.costs import DEFAULT_COST_MODEL
from src.metrics.scorecard import scorecard
from src.strategies.base import Strategy
from src.strategies.buy_and_hold import BuyAndHold
from src.strategies.cointegration_pairs import (
    CointegrationPairs,
    SUBSTITUTE_PAIRS,
    SUBSTITUTE_UNIVERSE_TAGS,
)
from src.strategies.ma_crossover import MovingAverageCrossover
from src.strategies.vrp_term_structure import VRPTermStructure
from src.validation.permutation import permutation_test
from src.validation.split import train_test_split


# --- Bake-off configuration -------------------------------------------------
# These are constants on purpose: changing them changes the comparison
# and should be a deliberate, reviewable code edit, not a CLI flag.

START = "2010-01-01"            # earliest date the bake-off considers
END = "2024-12-31"              # latest date the bake-off considers
TRAIN_RATIO = 0.70              # IS / OOS split (see validation/split.py)
N_PERMUTATIONS = 1000           # permutation test count (1000 → min p = 0.001)
PERMUTATION_SEED = 42           # for reproducibility

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate(strategy: Strategy) -> list[dict]:
    """Run a single strategy through the full pipeline.

    Returns a list of three scorecard dicts (full / IS / OOS), each
    already tagged with the strategy name and split label so they can
    be concatenated directly into the final comparison table.

    Side effect: writes one CSV (equity curve + diagnostics) per
    strategy into results/.
    """
    print(f"\n=== Evaluating: {strategy.name} ===", flush=True)

    # 1. Backtest on the full window.
    result = run_backtest(strategy, START, END, DEFAULT_COST_MODEL)
    print(
        f"  backtest produced {len(result.net_returns)} daily obs "
        f"(first: {result.net_returns.index[0].date()}, "
        f"last: {result.net_returns.index[-1].date()})",
        flush=True,
    )

    # 2. Split for IS/OOS scoring. We split the NET RETURNS series,
    # not the raw position series — what matters for the scorecard
    # comparison is the realised PnL in each window.
    returns_split = train_test_split(result.net_returns, TRAIN_RATIO)
    positions_split = train_test_split(result.positions, TRAIN_RATIO)

    # 3. Scorecards.
    rows = [
        scorecard(result.net_returns, result.positions,
                  label=f"{strategy.name} [FULL]"),
        scorecard(returns_split.in_sample, positions_split.in_sample,
                  label=f"{strategy.name} [IS]"),
        scorecard(returns_split.out_of_sample, positions_split.out_of_sample,
                  label=f"{strategy.name} [OOS]"),
    ]

    # 4. Null-distribution test on the FULL window.
    # If a strategy exposes its own `bootstrap_null` method we prefer
    # it — the default permutation test is wrong for strategies whose
    # engine-level positions are constant (like the cointegration
    # pairs strategy: it does its real signal work inside the
    # synthetic NAV, so shuffling its constant-1 positions is a
    # degenerate no-op). Duck-typed: any strategy can opt into a
    # custom null by implementing `bootstrap_null` with the same
    # `p_value` / `n_*` surface.
    if hasattr(strategy, "bootstrap_null"):
        null = strategy.bootstrap_null(seed=PERMUTATION_SEED)
        print(f"  bootstrap (strategy-specific) null: {null}", flush=True)
        rows[0]["perm_pvalue"] = null.p_value
        rows[0]["perm_n"] = null.n_bootstraps
        rows[0]["null_type"] = "block_bootstrap"
    else:
        perm = permutation_test(
            strategy,
            START,
            END,
            n_permutations=N_PERMUTATIONS,
            cost_model=DEFAULT_COST_MODEL,
            seed=PERMUTATION_SEED,
            baseline_result=result,
        )
        print(f"  permutation test: {perm}", flush=True)
        rows[0]["perm_pvalue"] = perm.p_value
        rows[0]["perm_n"] = perm.n_permutations
        rows[0]["null_type"] = "position_shuffle"

    # IS/OOS rows: we deliberately do NOT stamp a null p-value on
    # them. Their slices are shorter and would need their own null
    # distributions to get a matching reference; mixing a full-window
    # p-value into a split-window row would be misleading.
    for row_idx in (1, 2):
        rows[row_idx]["perm_pvalue"] = float("nan")
        rows[row_idx]["perm_n"] = 0
        rows[row_idx]["null_type"] = ""

    # 5. Save per-strategy artifacts. CSV is always written; PNG is
    # best-effort (matplotlib may not be installed in a minimal env).
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eq = result.equity_curve()
    diag = pd.DataFrame({
        "net_return": result.net_returns,
        "gross_return": result.gross_returns,
        "position": result.positions,
        "raw_position": result.raw_positions,
        "asset_return": result.asset_returns,
        "trade_cost": result.trade_costs,
        "financing_cost": result.financing_costs,
        "equity": eq,
    })
    diag.to_csv(RESULTS_DIR / f"{strategy.name}_diagnostics.csv")

    # PNG of the equity curve, with a buy-and-hold reference line for
    # quick visual sanity. We skip silently if matplotlib is missing.
    try:
        import matplotlib

        # Use a non-interactive backend so this runs in headless envs
        # (CI, ssh sessions) without trying to open a window.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        eq.plot(ax=ax, label=strategy.name)
        ax.set_title(f"Equity curve — {strategy.name}")
        ax.set_ylabel("Growth multiple (start = 1.0)")
        ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{strategy.name}_equity.png", dpi=120)
        plt.close(fig)
    except ImportError:
        warnings.warn(
            "matplotlib not installed; skipping PNG output. Install it "
            "with 'pip install matplotlib' if you want equity-curve plots."
        )

    return rows


def main() -> int:
    """Run the bake-off and write the comparison CSV. Returns exit code."""
    # The candidate set. Order is preserved in the output table.
    # To add a strategy, write a new module in src/strategies/ and add
    # an instance here.
    candidates: list[Strategy] = [
        BuyAndHold(ticker="SPY"),
        MovingAverageCrossover(
            ticker="SPY", fast_window=50, slow_window=200, allow_short=False
        ),
        MovingAverageCrossover(
            ticker="SPY", fast_window=20, slow_window=100, allow_short=False
        ),
        MovingAverageCrossover(
            ticker="SPY", fast_window=50, slow_window=200, allow_short=True
        ),
        # Cointegration pairs — BROAD universe (17 tickers, C(17,2)=136
        # candidate pairs). Tests whether ANY pair in a diversified
        # multi-asset universe survives FDR + half-life screening.
        CointegrationPairs(
            universe_label="broad",
            train_ratio=TRAIN_RATIO,
            screen_csv_path=RESULTS_DIR / "cointegration_pair_screen.csv",
        ),
        # VRP via VIX term structure. Long SPY in contango, flat in
        # backwardation. Uses the framework's DEFAULT permutation test
        # (positions actually vary, so the standard null is the right
        # one — no `bootstrap_null` defined). yfinance no longer
        # serves ^VXV, so this strategy will automatically fall back
        # to (^VIX9D, ^VIX); see the warning + docstring in the
        # strategy module.
        VRPTermStructure(),
        # Cointegration pairs — pre-specified ECONOMIC-SUBSTITUTE pairs
        # (9 pairs). This is a separate, theory-driven hypothesis test
        # added after the broad screen returned zero survivors. Same
        # thresholds, much smaller hypothesis space → BH-FDR is far
        # more powerful here. THIS IS A SINGLE ADDITIONAL TEST: if it
        # also returns zero survivors, pairs trading is dropped from
        # the fund. No third universe.
        CointegrationPairs(
            universe=SUBSTITUTE_UNIVERSE_TAGS,
            candidate_pairs=SUBSTITUTE_PAIRS,
            universe_label="substitutes",
            train_ratio=TRAIN_RATIO,
            screen_csv_path=(
                RESULTS_DIR / "cointegration_pair_screen_substitutes.csv"
            ),
        ),
    ]

    all_rows: list[dict] = []
    for strat in candidates:
        try:
            all_rows.extend(evaluate(strat))
        except Exception as exc:
            # One failing strategy shouldn't block the rest of the
            # bake-off. Log loudly and continue.
            print(
                f"!! strategy {strat.name} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue

    if not all_rows:
        print("No strategies completed successfully.", file=sys.stderr)
        return 1

    # Build comparison table.
    table = pd.DataFrame(all_rows)

    # Save the comparison CSV.
    out_csv = RESULTS_DIR / "scorecard_comparison.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nWrote comparison table → {out_csv}", flush=True)

    # Print the OOS rows sorted by Sharpe — that's the "what wins" view.
    oos_rows = table[table["label"].str.contains(r"\[OOS\]", regex=True)]
    oos_sorted = oos_rows.sort_values("sharpe", ascending=False)

    # Pretty-print the headline columns. Wider columns would still
    # render fine but truncating keeps the terminal output readable.
    cols = [
        "label", "n_obs", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "hit_rate", "turnover", "nw_tstat",
    ]
    print("\n=== OOS leaderboard (sorted by Sharpe) ===")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(oos_sorted[cols].to_string(index=False))

    # Show the full table with permutation p-values for the FULL rows.
    print("\n=== Full scorecard (every row, every split) ===")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        print(table.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
