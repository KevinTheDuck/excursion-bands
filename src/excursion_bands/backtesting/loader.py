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
    BreakEvenConfig,
    CoreDataConfig,
    ExecutionConfig,
    InstrumentConfig,
    MLConfig,
    MonteCarloConfig,
    ReportConfig,
    RiskConfig,
    SizingConfig,
    OpeningRangeConfig,
    OrbAtrConfig,
    OrbAtrStopConfig,
    OrbStopConfig,
    OrbTakeProfitConfig,
    StrategyConfig,
    VariantConfig,
    WfoConfig,
    XGBoostConfig,
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

            if not _parse_bool(variant.get("active", True)):
                continue

            variant_label = variant["metadata"]["label"]
            variant_configuration = variant["configuration"]
            side_mode = _normalize_side_mode(variant_configuration.get("side_mode", "both"))

            print(logger(_tag_str, f"Loading {variant_label}"))
            variants.append(
                VariantConfig(
                    label=variant_label,
                    use_band_filter=_parse_bool(variant_configuration.get("use_band_filter", False)),
                    side_mode=side_mode,
                    use_atr_buffer=_parse_bool(variant_configuration.get("use_atr_buffer", False)),
                    use_vwap_filter=_parse_bool(variant_configuration.get("use_vwap_filter", False)),
                    use_ml_filter=_parse_bool(variant_configuration.get("use_ml_filter", False)),
                    ml_execution_mode=str(variant_configuration.get("ml_execution_mode", "filter")),
                    ml_candidate_scope=str(variant_configuration.get("ml_candidate_scope", "variant")),
                    description=variant["metadata"].get("description"),
                )
            )

    return tuple(variants)


def _normalize_side_mode(value: object) -> str:
    side_mode = str(value).lower().strip()
    aliases = {
        "both": "both",
        "all": "both",
        "long": "long_only",
        "long_only": "long_only",
        "long-only": "long_only",
        "short": "short_only",
        "short_only": "short_only",
        "short-only": "short_only",
    }
    if side_mode not in aliases:
        raise ValueError(
            f"Unsupported side_mode '{value}'. Use both, long, short, long_only, or short_only."
        )
    return aliases[side_mode]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        if normalized in {"true", "yes", "y", "1", "enabled", "enable", "on"}:
            return True
        if normalized in {"false", "no", "n", "0", "disabled", "disable", "off"}:
            return False
    raise ValueError(f"Expected boolean-like value, got {value!r}")


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
    ml = config.get("ml", {})
    strategy = config.get("strategy")

    core_data = CoreDataConfig(**data_config)

    return BacktestConfig(
        core_data=core_data,
        backtest=load_backtest_setting_config(config["backtest"]),
        instrument=InstrumentConfig(**config["instrument"]),
        execution=ExecutionConfig(**config["execution"]),
        sizing=SizingConfig(**config["sizing"]),
        risk=RiskConfig(**config["risk"]),
        benchmark=BenchmarkConfig(
            enabled=_parse_bool(config["benchmark"].get("enabled", False)),
            strategy=config["benchmark"]["strategy"],
        ),
        reports=ReportConfig(
            output_dir=reports["output_dir"],
            save_charts=_parse_bool(reports["save_charts"]),
            monte_carlo=MonteCarloConfig(
                enabled=_parse_bool(monte_carlo.get("enabled", False)),
                simulations=int(monte_carlo.get("simulations", 0)),
                sample_trades=int(monte_carlo.get("sample_trades", 100)),
                max_paths_plotted=int(monte_carlo.get("max_paths_plotted", 100)),
                seed=int(monte_carlo.get("seed", 42)),
            ),
        ),
        wfo=WfoConfig(
            enabled=_parse_bool(wfo.get("enabled", False)),
            max_parameter_combinations=int(wfo.get("max_parameter_combinations", 100)),
            max_workers=int(wfo.get("max_workers", 1)),
            mode=str(wfo.get("mode", "calendar")),
            train_months=int(wfo.get("train_months", 6)),
            test_months=int(wfo.get("test_months", 3)),
            step_months=int(wfo.get("step_months", wfo.get("test_months", 3))),
            train_sessions=int(wfo.get("train_sessions", 504)),
            test_sessions=int(wfo.get("test_sessions", 63)),
            step_sessions=int(wfo.get("step_sessions", wfo.get("test_sessions", 63))),
            objective=str(wfo.get("objective", "sharpe")),
            parameter_grid=wfo.get("parameter_grid", {}),
        ),
        ml=None if not ml else MLConfig(
            enabled=_parse_bool(ml.get("enabled", False)),
            probability_threshold=float(ml.get("probability_threshold", 0.55)),
            min_train_samples=int(ml.get("min_train_samples", 100)),
            warmup_start_date=_parse_date(ml.get("warmup_start_date")),
            train_lookback_months=None
            if ml.get("train_lookback_months") is None
            else int(ml.get("train_lookback_months")),
            refit_frequency_sessions=int(ml.get("refit_frequency_sessions", 20)),
            fallback=str(ml.get("fallback", "allow_all")),
            xgboost=XGBoostConfig(**ml.get("xgboost", {})),
        ),
        strategy=None if strategy is None else StrategyConfig(
            name=strategy["name"],
            opening_range=OpeningRangeConfig(**strategy["opening_range"]),
            entry_bars_after_or=int(strategy["entry_bars_after_or"]),
            force_exit_time=strategy["force_exit_time"],
            atr=OrbAtrConfig(**strategy["atr"]),
            stop=OrbStopConfig(**strategy["stop"]),
            atr_stop=OrbAtrStopConfig(
                enabled=_parse_bool(strategy["atr_stop"].get("enabled", False)),
                length=int(strategy["atr_stop"]["length"]),
                multiplier=float(strategy["atr_stop"]["multiplier"]),
            ),
            take_profit=OrbTakeProfitConfig(**strategy["take_profit"]),
            break_even=BreakEvenConfig(
                enabled=_parse_bool(strategy["break_even"].get("enabled", False)),
                trigger_rr=float(strategy["break_even"]["trigger_rr"]),
                offset_points=float(strategy["break_even"]["offset_points"]),
            ),
        ),
        variants=load_variants(core_data.variants),
    )
