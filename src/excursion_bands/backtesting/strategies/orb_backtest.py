
"""
[backtesting/strategies/orb_backtest.py]
Backtester for ORB strategy
"""

from excursion_bands.backtesting import load_backtest_config, load_core_data


def start_backtest(config_path: str) -> None:
    config = load_backtest_config(config_path)
    intraday, bands = load_core_data(config.core_data)

    print(intraday.tail(3))
    print(bands.tail(3))
