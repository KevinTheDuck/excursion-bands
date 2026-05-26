"""
Backtest entrypoint used by executable research scripts.
"""

from excursion_bands.backtesting.benchmarks import run_buy_and_hold
from excursion_bands.backtesting.loader import load_backtest_config, load_core_data
from excursion_bands.backtesting.reports import print_receipt, save_report


def start_backtest(config_path: str) -> None:
    config = load_backtest_config(config_path)
    intraday, _bands = load_core_data(config.core_data)

    if config.benchmark.enabled and config.benchmark.strategy == "buy_and_hold":
        result = run_buy_and_hold(intraday.to_pandas(), config)
        output_dir = save_report(result, config)
        print_receipt(result, output_dir)
        return

    raise ValueError(f"Unsupported benchmark strategy: {config.benchmark.strategy}")
