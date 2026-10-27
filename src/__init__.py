"""quant-fund-sim package root.

This file exists to mark `src/` as a Python package. We do not re-export
anything from here on purpose — callers should import from the explicit
subpackages (`src.engine.backtest`, `src.metrics.scorecard`, etc.) so it
is always obvious where a symbol lives.
"""
