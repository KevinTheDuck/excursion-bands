"""
Benchmark strategies.
"""

import pandas as pd

from excursion_bands.backtesting.engine import run_backtest
from excursion_bands.backtesting.models import BacktestResult, Position
from excursion_bands.backtesting.specification import BacktestConfig


def _buy_and_hold_signal(
    index: int, _bar: pd.Series, _bars: pd.DataFrame, position: Position | None
) -> str | None:
    if index == 0 and position is None:
        return "long"
    return None


def run_buy_and_hold(bars: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    return run_backtest("buy_and_hold", bars, config, _buy_and_hold_signal)
