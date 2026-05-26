"""
Loader modules for the backtesting pipeline.
"""

from datetime import date
from typing import Any

import polars as pl

from excursion_bands.backtesting.specification import (
    BacktestConfig,
    BacktestSettingConfig,
    BenchmarkConfig,
    CoreDataConfig,
    ExecutionConfig,
    InstrumentConfig,
    MonteCarloConfig,
    ReportConfig,
    RiskConfig,
    SizingConfig,
    VariantConfig,
    WfoConfig,
)
from excursion_bands.data import load_yaml
from excursion_bands.paths import resolve_path
from excursion_bands.pipeline import (
    load_aggregated_data,
    load_excursion_bands_data,
    load_processed_data,
    load_raw_data,
    process_raw_data,
)
from excursion_bands.utils import logger


def load_config_from_yaml(path: str) -> dict[str, Any]:
    _tag_str = "[backtesting/loader/load_config_from_yaml]"
    print(logger(_tag_str, f"Loading config data from {path}"))

    resolved_path, exists = resolve_path(path)
    if not exists:
        raise FileNotFoundError(logger(_tag_str, f"File doesn't exists! {path}"))
    return load_yaml(resolved_path)


def load_core_data(config: CoreDataConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    _tag_str = "[backtesting/loader/load_core_data]"
    print(logger(_tag_str, "Loading core data"))

    data_cfg = load_config_from_yaml(config.data_config)
    sessions_cfg = load_config_from_yaml(config.sessions_config)
    volatility_cfg = load_config_from_yaml(config.volatility_config)
    bands_cfg = load_config_from_yaml(config.bands_config)

    raw_1m = load_raw_data(data_cfg)
    processed = load_processed_data(data_cfg, raw_1m)
    intraday = process_raw_data(data_cfg, sessions_cfg, processed)
    aggregated = load_aggregated_data(data_cfg, intraday)
    bands = load_excursion_bands_data(data_cfg, volatility_cfg, bands_cfg, aggregated)

    return intraday, bands


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_backtest_setting_config(config: dict[str, Any]) -> BacktestSettingConfig:
    _tag_str = "[backtesting/loader/load_backtest_setting_config]"
    print(logger(_tag_str, "Loading backtest setting"))

    return BacktestSettingConfig(
        start_date=_parse_date(config["start_date"]),
        end_date=_parse_date(config.get("end_date")),
        initial_cash=float(config["initial_cash"]),
    )


def load_variants(path: str | None) -> tuple[VariantConfig, ...]:
    if path is None:
        return ()

    _tag_str = "[backtesting/loader/load_variants]"
    print(logger(_tag_str, f"Loading variants from {path}"))

    resolved_path, exists = resolve_path(path)
    variants = []

    if not exists:
        return ()

    for file in resolved_path.iterdir():
        if file.is_file() and file.suffix in {".yaml", ".yml"}:
            variant = load_yaml(file)

            if variant["active"] is False:
                continue

            variant_label = variant["metadata"]["label"]
            variant_configuration = variant["configuration"]

            print(logger(_tag_str, f"Loading {variant_label}"))
            variants.append(
                VariantConfig(
                    label=variant_label,
                    use_band_filter=bool(variant_configuration["use_band_filter"]),
                    side_mode=str(variant_configuration["side_mode"]),
                    description=variant["metadata"].get("description"),
                )
            )

    return tuple(variants)


def load_backtest_config(config_path: str) -> BacktestConfig:
    _tag_str = "[backtesting/loader/load_backtest_config]"
    print(logger(_tag_str, f"Loading backtest config from {config_path}"))

    resolved_path, exists = resolve_path(config_path)

    if not exists:
        raise FileNotFoundError(logger(_tag_str, f"File doesn't exists! {config_path}"))

    config = load_yaml(resolved_path)
    data_config = config.get("data", config.get("data_path"))
    if data_config is None:
        raise ValueError("Backtest config requires a data section")

    reports = config["reports"]
    monte_carlo = reports.get("monte_carlo", {})
    wfo = config.get("wfo", {})

    core_data = CoreDataConfig(**data_config)

    return BacktestConfig(
        core_data=core_data,
        backtest=load_backtest_setting_config(config["backtest"]),
        instrument=InstrumentConfig(**config["instrument"]),
        execution=ExecutionConfig(**config["execution"]),
        sizing=SizingConfig(**config["sizing"]),
        risk=RiskConfig(**config["risk"]),
        benchmark=BenchmarkConfig(**config["benchmark"]),
        reports=ReportConfig(
            output_dir=reports["output_dir"],
            save_charts=bool(reports["save_charts"]),
            monte_carlo=MonteCarloConfig(
                enabled=bool(monte_carlo.get("enabled", False)),
                simulations=int(monte_carlo.get("simulations", 0)),
                sample_trades=int(monte_carlo.get("sample_trades", 100)),
                max_paths_plotted=int(monte_carlo.get("max_paths_plotted", 100)),
                seed=int(monte_carlo.get("seed", 42)),
            ),
        ),
        wfo=WfoConfig(
            enabled=bool(wfo.get("enabled", False)),
            max_parameter_combinations=int(wfo.get("max_parameter_combinations", 100)),
            max_workers=int(wfo.get("max_workers", 1)),
        ),
        variants=load_variants(core_data.variants),
    )
