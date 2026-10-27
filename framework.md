# quant-fund-sim — Strategy Evaluation Framework

## Purpose

This repository is the evaluation harness for a 5-week hedge-fund-simulation
course project. Several candidate trading strategies are run through an
**identical** backtest + validation pipeline so their performance statistics
can be compared on equal footing. The winner of that bake-off becomes the
single strategy the simulated fund will trade.

The framework itself is what's being graded for rigor: a strategy that "wins"
because it was given more data, looser cost assumptions, or a peek at the
future is not a real winner. Every design decision in this codebase exists
to prevent that.

## Repository layout

```
quant-fund-sim/
├── framework.md                 # this file
├── README.md                 # short user-facing overview
├── requirements.txt          # pinned dependency floor
├── .gitignore                # excludes caches, results, venvs
├── run_bakeoff.py            # top-level orchestration script
├── src/
│   ├── strategies/           # one file per candidate strategy + the ABC
│   │   ├── base.py           # Strategy abstract base class (the interface)
│   │   ├── buy_and_hold.py   # benchmark
│   │   └── ma_crossover.py   # example trend strategy
│   ├── engine/
│   │   ├── backtest.py       # turns Strategy -> daily net return series
│   │   └── costs.py          # transaction-cost / slippage / financing model
│   ├── metrics/
│   │   └── scorecard.py      # Sharpe, Sortino, MDD, hit rate, NW t-stat, ...
│   ├── validation/
│   │   ├── split.py          # in-sample / out-of-sample temporal split
│   │   └── permutation.py    # signal-shuffle null distribution + p-value
│   └── data/
│       └── loader.py         # yfinance fetch with on-disk parquet cache
├── results/                  # gitignored — backtest outputs land here
└── tests/                    # pytest unit tests for engine / metrics / validation
```

## How to run

```bash
# 1. Create and activate a virtual environment (one-time).
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies.
pip install -r requirements.txt

# 3. Run the bake-off across all registered strategies.
python run_bakeoff.py

# 4. Run the unit tests.
pytest tests/
```

`run_bakeoff.py` writes a CSV scorecard and an equity-curve plot per
strategy into `results/`.

## Conventions

- **Python style**: PEP 8, 4-space indent, ~88-char line limit. We don't
  enforce with a linter in CI (this is a course project), but new code
  should match the surrounding style.
- **Type hints**: required on every public function signature. They double
  as documentation and catch a real class of bugs early.
- **Docstrings**: every module, class, and non-trivial function gets one.
  Format is informal — describe *what it returns*, *what it assumes*, and
  *what it does not handle*.
- **Imports**: standard library, then third-party, then local — separated
  by a blank line.
- **No magic numbers**: cost assumptions, lookback windows, etc., live in
  named constants or function arguments so they show up in the scorecard.

## What this framework deliberately does NOT model

Honesty matters more than optimism in a backtest report. The following
are **known unmodelled effects** — strategy results should be read with
them in mind:

- **Market impact**: we assume our orders are price-takers. A real fund
  trading any meaningful size moves the book; for the small notional in a
  course project this is defensible, but it would not be in production.
- **Liquidity constraints**: we do not check ADV or bid/ask depth before
  executing. Cheap, liquid ETFs (SPY, QQQ) are fine; thinly traded
  small-caps would not be.
- **Survivorship bias**: if a strategy picks names from a current index
  list, dead companies are silently absent. Use total-market ETFs or
  hand-curated lists with delisted names to avoid this.
- **Look-ahead bias in fundamentals**: we do not currently use fundamental
  data (earnings, etc.); if added, point-in-time data is required because
  reported numbers get revised.
- **Tax**: ignored. Realistic for an institutional fund; not for a
  personal account.
- **Borrow costs for shorts**: we apply a generic financing rate. Hard-to-
  borrow names cost much more and the rate is volatile; the model is a
  rough approximation.

## What the validation actually proves (and doesn't)

- **OOS split**: confirms the strategy was not literally overfit to one
  period. It does NOT prove the edge will persist — markets regime-shift,
  and 30% of a 10-year window is still only ~3 years of test data.
- **Permutation p-value**: tests the null "the timing of this strategy's
  signals is uninformative about returns". A small p-value means it's
  unlikely the observed Sharpe arose from random signal timing on this
  asset. It does NOT correct for multiple testing across strategies — if
  we run 20 random strategies, one will hit p < 0.05 by chance. The
  bake-off is comparison-only; we should not interpret any individual
  p-value as a discovery without a multiple-testing correction.
  DEGENERATE CASE: a constant-position strategy (e.g. buy-and-hold)
  has p = 1.0 by construction, because shuffling identical positions
  is a no-op. That's a feature of the test, not a defect; treat the
  p-value column as N/A for such benchmarks and judge them on
  Sharpe/Sortino vs the active candidates.
- **Newey-West t-stat**: corrects the standard error of the mean return
  for serial correlation. A |t| > 2 is the usual rough cutoff for "the
  mean isn't zero", but it shares all the multiple-testing caveats above.

When reporting results, restate these caveats. A clean backtest is a
*necessary* but nowhere near *sufficient* condition for live deployment.
