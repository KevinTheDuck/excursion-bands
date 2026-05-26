"""
Backtest entrypoint used by executable research scripts.
"""

from dataclasses import replace

from excursion_bands.backtesting.benchmarks import run_buy_and_hold
from excursion_bands.backtesting.loader import load_backtest_config, load_core_data
from excursion_bands.backtesting.ml import expanding_ml_filter, write_expanding_ml_artifacts
from excursion_bands.backtesting.reports import print_receipt, save_report
from excursion_bands.backtesting.strategies.orb import (
    default_orb_variants,
    prepare_orb_bars,
    run_orb_variant,
)
from excursion_bands.backtesting.wfo import run_wfo


def _print_comparison(results: list[tuple[str, dict]]) -> None:
    if not results:
        return
    print("\nVariant Comparison")
    print("------------------")
    print("variant | return% | cagr% | max_dd% | sharpe | trades | win_rate% | pf")
    for label, metrics in results:
        print(
            " | ".join(
                [
                    label,
                    _fmt(metrics.get("total_return_pct")),
                    _fmt(metrics.get("cagr_pct")),
                    _fmt(metrics.get("max_drawdown_pct")),
                    _fmt(metrics.get("sharpe")),
                    str(metrics.get("trade_count", 0)),
                    _fmt(metrics.get("win_rate_pct")),
                    _fmt(metrics.get("profit_factor")),
                ]
            )
        )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def start_backtest(config_path: str) -> None:
    config = load_backtest_config(config_path)
    intraday, bands = load_core_data(config.core_data)
    intraday_pdf = intraday.to_pandas()
    comparison = []

    if config.wfo.enabled:
        run_wfo(intraday_pdf, bands.to_pandas(), config)
        return

    if config.strategy is not None and config.strategy.name == "orb":
        variants = config.variants or default_orb_variants()
        prepared = prepare_orb_bars(intraday_pdf, config, bands.to_pandas())
        for variant in variants:
            allowed_signal_times = None
            ml_result = None
            if variant.use_ml_filter:
                ml_variant = variant
                if variant.ml_candidate_scope == "raw":
                    ml_variant = replace(variant, use_band_filter=False, use_ml_filter=False)
                elif variant.ml_candidate_scope != "variant":
                    raise ValueError(
                        f"Unsupported ml_candidate_scope: {variant.ml_candidate_scope}"
                    )
                ml_result = expanding_ml_filter(prepared, config, ml_variant)
                allowed_signal_times = ml_result.allowed_signal_times
            result = run_orb_variant(prepared, config, variant, allowed_signal_times)
            output_dir = save_report(result, config)
            if ml_result is not None:
                write_expanding_ml_artifacts(output_dir, variant.label, ml_result)
            print_receipt(result, output_dir)
            comparison.append((variant.label, result.metrics))

    if config.benchmark.enabled:
        if config.benchmark.strategy != "buy_and_hold":
            raise ValueError(f"Unsupported benchmark strategy: {config.benchmark.strategy}")
        result = run_buy_and_hold(intraday_pdf, config)
        output_dir = save_report(result, config)
        print_receipt(result, output_dir)
        comparison.append((result.name, result.metrics))

    if comparison:
        _print_comparison(comparison)
        return

    raise ValueError(f"Unsupported benchmark strategy: {config.benchmark.strategy}")
