import polars as pl

from typing import Any
from excursion_bands.backtesting import BacktestConfig, CoreDataConfig
from excursion_bands.data import load_yaml
from excursion_bands.paths import resolve_path
from excursion_bands.pipeline import (
    load_raw_data, load_processed_data,
    process_raw_data, load_aggregated_data,
    load_excursion_bands_data
)
from excursion_bands.utils import logger


"""
[backtesting/loader.py]
Loader modules for backtesting pipeline
"""

def load_config_from_yaml(path: str) -> dict[str, Any]:
    _tag_str = "[backtesting/loader/load_config_from_yaml]"
    print(logger(_tag_str, f"Loading config data from {path}"))

    resolved_path, exists = resolve_path(path)
    if not exists:
        raise FileNotFoundError(
            logger(_tag_str, f"File doesn't exists! {path}")
        )
    return load_yaml(resolved_path)


def load_core_data(config: CoreDataConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    _tag_str = "[backtesting/loader/load_core_data]"
    print(logger(_tag_str, f"Loading core data"))

    data_cfg = load_config_from_yaml(config.data_config)
    sessions_cfg = load_config_from_yaml(config.sessions_config)
    volatility_cfg = load_config_from_yaml(config.volatility_config)
    bands_cfg = load_config_from_yaml(config.bands_config)

    raw_1m = load_raw_data(data_cfg)
    processed = load_processed_data(data_cfg, raw_1m)
    intraday = process_raw_data(data_cfg, sessions_cfg, processed)
    aggregated = load_aggregated_data(data_cfg, intraday)
    bands = load_excursion_bands_data(
        data_cfg, volatility_cfg, bands_cfg, aggregated
    )

    return intraday, bands

def load_backtest_config(config_path: str) -> BacktestConfig:
    _tag_str = "[backtesting/loader/load_backtest_config]"
    logger(_tag_str, f"Loading backtest config from {config_path}")

    resolved_path, exists = resolve_path(config_path)

    if not exists:
        raise FileNotFoundError(
            logger(_tag_str, f"File doesn't exists! {config_path}")
        )

    config = load_yaml(resolved_path)
    return BacktestConfig(
        core_data=CoreDataConfig(**config["data_path"])
    )
