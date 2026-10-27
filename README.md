# Quant Strategy Research | Hedge Fund Simulation

A Python research framework for comparing systematic trading strategies under consistent backtesting and validation assumptions. Developed alongside a five-week **simulated** macro-fund mandate; this repository focuses on the **quantitative strategy research**, not the discretionary trading ledger.

## What it does

- Compares a buy-and-hold benchmark with moving-average crossover, cointegration pairs, and volatility-risk-premium (VRP) strategies.
- Models trading costs and portfolio returns in a common backtesting engine.
- Applies chronological in-sample/out-of-sample splits, performance scorecards, and permutation / bootstrap-based checks.
- Generates comparable metrics and equity-curve visualizations.
- Includes automated tests for the engine, metrics, statistical validation, and strategies.

**Research takeaway:** a promising backtest alone is not sufficient evidence to trade a strategy. Results need to be evaluated for sensitivity, statistical significance, and out-of-sample robustness. This is a research/simulation project, not a live trading system.

## Stack

Python, pandas, NumPy, SciPy, statsmodels, yfinance, matplotlib, pytest, and pyarrow.

## Run it

Requires Python 3 and internet access to download historical market data on the first run.

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_bakeoff.py
pytest tests/
```

Outputs, including scorecards and plots, are generated under `results/`. Historical-price caches are deliberately not committed.

## Repository map

```text
run_bakeoff.py       # runs comparable strategy evaluations
src/data/            # historical-data loading and caching
src/engine/          # backtest and transaction-cost assumptions
src/strategies/      # candidate trading strategies
src/metrics/         # returns and risk scorecards
src/validation/      # data splits and statistical checks
tests/               # automated tests
```

## Methodology and limitations

- The research window configured in `run_bakeoff.py` is **2010-01-01 to 2024-12-31**, with a 70/30 chronological split and 1,000 permutations by default.
- Historical data are retrieved through yfinance; revised prices, missing data, slippage assumptions, and ticker availability can affect results.
- Backtested outcomes are hypothetical. They are **not** verified live returns and do not guarantee future performance.
- See [`framework.md`](framework.md) for the original detailed technical design notes and explicit model assumptions.

**Author:** Kimberly Bertoli. Educational quantitative-finance research. Not investment advice.
