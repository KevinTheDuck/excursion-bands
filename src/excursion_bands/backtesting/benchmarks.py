"""
Benchmark strategies.
"""

import pandas as pd
from dataclasses import replace

from excursion_bands.backtesting.engine import run_backtest
from excursion_bands.backtesting.models import BacktestResult, Position
from excursion_bands.backtesting.specification import BacktestConfig, SizingConfig


def _buy_and_hold_signal(
    index: int, _bar: pd.Series, _bars: pd.DataFrame, position: Position | None
) -> str | None:
    if index == 0 and position is None:
        return "long"
    return None


def run_buy_and_hold(bars: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    benchmark_config = config
    if config.sizing.mode != "fixed_units":
        benchmark_config = replace(
            config,
            sizing=SizingConfig(
                mode="fixed_units",
                fixed_units=config.sizing.fixed_units,
                risk_pct=config.sizing.risk_pct,
                max_units=config.sizing.max_units,
            ),
        )
    return run_backtest("buy_and_hold", bars, benchmark_config, _buy_and_hold_signal)
