"""Engine subpackage — converts a Strategy into a net daily return series.

Keep this package thin: `costs.py` for the cost model, `backtest.py` for
the PnL accounting. Nothing strategy-specific belongs here; that's the
whole point of the abstraction.
"""
