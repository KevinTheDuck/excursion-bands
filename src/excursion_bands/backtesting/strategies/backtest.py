"""
Backtest entrypoint used by executable research scripts.
"""

from excursion_bands.backtesting.benchmarks import run_buy_and_hold
from excursion_bands.backtesting.hmm import train_hmm_filter, write_hmm_artifacts
from excursion_bands.backtesting.loader import load_backtest_config, load_core_data
from excursion_bands.backtesting.reports import print_receipt, save_report
from excursion_bands.backtesting.strategies.donchian import (
    default_donchian_variants,
    prepare_donchian_bars,
    run_donchian_variant,
)
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

    if config.strategy is not None and config.strategy.name in {"orb", "donchian"}:
        if config.strategy.name == "orb":
            variants = config.variants or default_orb_variants()
            prepared = prepare_orb_bars(intraday_pdf, config, bands.to_pandas())
            runner = run_orb_variant
        else:
            variants = config.variants or default_donchian_variants()
            prepared = prepare_donchian_bars(intraday_pdf, config, bands.to_pandas())
            runner = run_donchian_variant

        for variant in variants:
            allowed_signal_times = None
            hmm_result = None
            prepared_variant = prepared.copy()
            if variant.use_hmm_filter:
                if config.hmm is None:
                    raise ValueError("HMM variant requires hmm config")
                start = config.backtest.start_date
                train_prepared = prepared_variant[
                    prepared_variant["DateTime"].dt.date < start
                ]
                test_prepared = prepared_variant[
                    prepared_variant["DateTime"].dt.date >= start
                ]
                hmm_result = train_hmm_filter(train_prepared, test_prepared, config, variant)
                allowed_signal_times = hmm_result.allowed_signal_times
            result = runner(prepared_variant, config, variant, allowed_signal_times)
            output_dir = save_report(result, config)
            if hmm_result is not None:
                write_hmm_artifacts(output_dir, variant.label, 0, hmm_result)
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
