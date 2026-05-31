"""
Walk-forward optimization for ORB variants.
"""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from excursion_bands.backtesting.models import BacktestResult
from excursion_bands.backtesting.metrics import calculate_metrics, calculate_yearly_metrics
from excursion_bands.backtesting.hmm import train_hmm_filter, write_hmm_artifacts
from excursion_bands.backtesting.specification import BacktestConfig, WfoConfig
from excursion_bands.backtesting.strategies.donchian import prepare_donchian_bars, run_donchian_variant
from excursion_bands.backtesting.strategies.orb import prepare_orb_bars, run_orb_variant
from excursion_bands.data import load_yaml
from excursion_bands.features.excursion_bands.excursion_bands import calculate_excursion_bands
from excursion_bands.paths import resolve_path

type Fold = dict[str, Any]


def validate_parameter_sweep(parameters: Sequence[object], config: WfoConfig) -> None:
    if len(parameters) > config.max_parameter_combinations:
        raise ValueError(
            f"Parameter sweep has {len(parameters)} combinations, exceeding "
            f"configured max_parameter_combinations={config.max_parameter_combinations}."
        )


def run_wfo(
    intraday: pd.DataFrame,
    bands: pd.DataFrame,
    config: BacktestConfig,
) -> Path:
    if config.strategy is None:
        raise ValueError("WFO requires a strategy config")

    parameter_sets = _build_parameter_sets(config.wfo)
    validate_parameter_sweep(parameter_sets, config.wfo)
    prepare_bars, run_variant = _strategy_functions(config)

    variants = config.variants
    if not variants:
        raise ValueError("WFO requires at least one active variant")
    intraday = intraday.sort_values("DateTime").reset_index(drop=True).copy()
    intraday["DateTime"] = pd.to_datetime(intraday["DateTime"])
    warmup_start = (
        config.hmm.warmup_start_date
        if config.hmm is not None and config.hmm.warmup_start_date is not None
        else config.backtest.start_date
    )
    intraday = intraday[intraday["DateTime"].dt.date >= warmup_start]
    if config.backtest.end_date is not None:
        intraday = intraday[intraday["DateTime"].dt.date <= config.backtest.end_date]
    bands = bands.copy()
    if "Session" not in intraday.columns:
        intraday["Session"] = pd.to_datetime(intraday["DateTime"]).dt.date
    trading_intraday = intraday[intraday["DateTime"].dt.date >= config.backtest.start_date]
    sessions = list(pd.Series(trading_intraday["Session"].dropna().unique()).sort_values())
    if config.wfo.mode == "calendar":
        folds = _build_calendar_folds(
            trading_intraday,
            pd.Timestamp(config.backtest.start_date),
            config.wfo.train_months,
            config.wfo.test_months,
            config.wfo.step_months,
        )
    elif config.wfo.mode == "sessions":
        folds = _build_session_folds(
            sessions,
            config.wfo.train_sessions,
            config.wfo.test_sessions,
            config.wfo.step_sessions,
        )
    else:
        raise ValueError("wfo.mode must be 'calendar' or 'sessions'")

    output_dir = _create_wfo_output_dir(config)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    hmm_rows: list[dict[str, Any]] = []
    stitched_equity: dict[str, list[pd.DataFrame]] = {variant.label: [] for variant in variants}
    stitched_trades: dict[str, list[pd.DataFrame]] = {variant.label: [] for variant in variants}
    variant_capital = {variant.label: float(config.backtest.initial_cash) for variant in variants}
    needs_optimization = {variant.label: True for variant in variants}
    current_params: dict[str, dict[str, Any]] = {}
    current_train_score: dict[str, float | None] = {}

    for variant in variants:
        print(f"\nWFO Variant: {variant.label}")
        variant_parameter_sets = _variant_parameter_sets(parameter_sets, variant)
        validate_parameter_sweep(variant_parameter_sets, config.wfo)
        band_cache = _build_band_cache(bands, variant_parameter_sets, config, variant)
        for fold in folds:
            fold_number = fold["fold"]
            train_sessions = fold["train_sessions"]
            test_sessions = fold["test_sessions"]
            train_data = intraday[intraday["Session"].isin(train_sessions)]
            best_result: BacktestResult | None = None
            best_params: dict[str, Any] | None = None
            best_score: float | None = None

            optimize_this_fold = _should_optimize(
                config, variant.label, needs_optimization, current_params
            )
            if optimize_this_fold:
                trial_results = _evaluate_parameter_sets(
                    variant_parameter_sets,
                    train_data,
                    band_cache,
                    train_sessions,
                    config,
                    variant,
                    prepare_bars,
                    run_variant,
                )
                for params, result, score in trial_results:
                    train_rows.append(
                        _row_from_result(
                            fold_number,
                            "train",
                            variant.label,
                            fold,
                            params,
                            result,
                            score,
                            True,
                        )
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_result = result
                        best_params = params

                if best_result is None or best_params is None:
                    continue
                current_params[variant.label] = best_params
                current_train_score[variant.label] = best_score
                needs_optimization[variant.label] = False
            else:
                best_params = current_params[variant.label]
                best_score = current_train_score.get(variant.label)
                train_rows.append(
                    _skipped_optimization_row(
                        fold_number,
                        variant.label,
                        fold,
                        best_params,
                        best_score,
                    )
                )

            if best_params is None:
                continue

            test_config = _patch_config(config, best_params)
            test_data = intraday[intraday["Session"].isin(test_sessions)]
            patched_bands = _select_bands(band_cache, best_params)
            test_bands = patched_bands[patched_bands["Session"].isin(test_sessions)]
            test_prepared = prepare_bars(test_data, test_config, test_bands)
            allowed_signal_times = None
            if variant.use_hmm_filter:
                hmm_train_data, hmm_train_bands = _hmm_training_window(
                    intraday, patched_bands, fold, config
                )
                hmm_train_prepared = prepare_bars(hmm_train_data, test_config, hmm_train_bands)
                hmm_result = train_hmm_filter(
                    hmm_train_prepared, test_prepared, test_config, variant
                )
                write_hmm_artifacts(output_dir, variant.label, fold_number, hmm_result)
                allowed_signal_times = hmm_result.allowed_signal_times
                hmm_rows.append(
                    {
                        "fold": fold_number,
                        "variant": variant.label,
                        "train_start": fold["train_start"],
                        "train_end": fold["train_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        **hmm_result.diagnostics,
                    }
                )
            test_result = run_variant(
                test_prepared, test_config, variant, allowed_signal_times
            )
            compounded_equity = _compound_equity_curve(
                test_result.equity_curve,
                config.backtest.initial_cash,
                variant_capital[variant.label],
                fold_number,
                variant.label,
            )
            if not compounded_equity.empty:
                variant_capital[variant.label] = float(compounded_equity["Equity"].iloc[-1])
                stitched_equity[variant.label].append(compounded_equity)
            if not test_result.trades.empty:
                trades = test_result.trades.copy()
                trades["Fold"] = fold_number
                trades["Variant"] = variant.label
                stitched_trades[variant.label].append(trades)
            test_score = _objective_score(test_result.metrics, config.wfo.objective)
            degradation_score = _objective_score(test_result.metrics, config.wfo.degradation_objective)
            degraded = _is_degraded(config, degradation_score)
            if degraded:
                needs_optimization[variant.label] = True
            test_rows.append(
                _row_from_result(
                    fold_number,
                    "test",
                    variant.label,
                    fold,
                    best_params,
                    test_result,
                    test_score,
                    optimize_this_fold,
                    degradation_score,
                    degraded,
                )
            )
            print(
                f"fold={fold_number} variant={variant.label} "
                f"optimized={optimize_this_fold} "
                f"train_{config.wfo.objective}={_format_optional_score(best_score)} "
                f"test_return={test_result.metrics.get('total_return_pct'):.2f}% "
                f"degraded={degraded} "
                f"params={best_params}"
            )

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    hmm_df = pd.DataFrame(hmm_rows)
    train_df.to_csv(output_dir / "wfo_train_trials.csv", index=False)
    test_df.to_csv(output_dir / "wfo_test_folds.csv", index=False)
    if not hmm_df.empty:
        hmm_df.to_csv(output_dir / "hmm_wfo_diagnostics.csv", index=False)
    stitched_metrics = _write_stitched_outputs(
        output_dir, config, stitched_equity, stitched_trades
    )
    research_summary = _write_wfo_research_outputs(
        output_dir, config, train_df, test_df, stitched_equity, stitched_trades
    )
    _write_wfo_report(output_dir, config, train_df, test_df, stitched_metrics, hmm_df)
    if research_summary:
        _append_wfo_research_report(output_dir, research_summary)
    _print_wfo_summary(test_df, output_dir)
    return output_dir


def _strategy_functions(config: BacktestConfig):
    if config.strategy is None:
        raise ValueError("Missing strategy config")
    if config.strategy.name == "orb":
        return prepare_orb_bars, run_orb_variant
    if config.strategy.name == "donchian":
        return prepare_donchian_bars, run_donchian_variant
    raise ValueError(f"Unsupported WFO strategy: {config.strategy.name}")


def _build_parameter_sets(wfo: WfoConfig) -> list[dict[str, Any]]:
    grid = wfo.parameter_grid or {}
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _variant_parameter_sets(parameter_sets: list[dict[str, Any]], variant) -> list[dict[str, Any]]:
    if variant.use_band_filter:
        return parameter_sets
    seen = set()
    filtered_sets = []
    for params in parameter_sets:
        filtered = {
            key: value
            for key, value in params.items()
            if not key.startswith("features.bands.")
        }
        signature = tuple(sorted(filtered.items()))
        if signature in seen:
            continue
        seen.add(signature)
        filtered_sets.append(filtered)
    return filtered_sets


def _evaluate_parameter_sets(
    parameter_sets: list[dict[str, Any]],
    train_data: pd.DataFrame,
    band_cache: dict[int | None, pd.DataFrame],
    train_sessions: Sequence[object],
    config: BacktestConfig,
    variant,
    prepare_bars,
    run_variant,
) -> list[tuple[dict[str, Any], BacktestResult, float]]:
    if config.wfo.max_workers <= 1 or len(parameter_sets) <= 1:
        return [
            _evaluate_parameter_set(params, train_data, band_cache, train_sessions, config, variant, prepare_bars, run_variant)
            for params in parameter_sets
        ]
    with ThreadPoolExecutor(max_workers=config.wfo.max_workers) as executor:
        futures = [
            executor.submit(
                _evaluate_parameter_set,
                params,
                train_data,
                band_cache,
                train_sessions,
                config,
                variant,
                prepare_bars,
                run_variant,
            )
            for params in parameter_sets
        ]
        return [future.result() for future in futures]


def _evaluate_parameter_set(
    params: dict[str, Any],
    train_data: pd.DataFrame,
    band_cache: dict[int | None, pd.DataFrame],
    train_sessions: Sequence[object],
    config: BacktestConfig,
    variant,
    prepare_bars,
    run_variant,
) -> tuple[dict[str, Any], BacktestResult, float]:
    trial_config = _patch_config(config, params)
    trial_bands = _select_bands(band_cache, params)
    trial_bands = trial_bands[trial_bands["Session"].isin(train_sessions)]
    trial_prepared = prepare_bars(train_data, trial_config, trial_bands)
    result = run_variant(trial_prepared, trial_config, variant)
    score = _objective_score(result.metrics, config.wfo.objective)
    return params, result, score


def _build_band_cache(
    bands: pd.DataFrame, parameter_sets: list[dict[str, Any]], config: BacktestConfig, variant
) -> dict[int | None, pd.DataFrame]:
    if not variant.use_band_filter:
        return {None: bands}

    lookbacks = {
        int(params["features.bands.lookback_window"])
        for params in parameter_sets
        if "features.bands.lookback_window" in params
    }
    if not lookbacks:
        return {None: bands}

    bands_config_path, exists = resolve_path(config.core_data.bands_config)
    if not exists:
        raise FileNotFoundError(f"Band config not found: {config.core_data.bands_config}")
    base_config = load_yaml(bands_config_path)

    cache: dict[int | None, pd.DataFrame] = {None: bands}
    for lookback in sorted(lookbacks):
        bands_config = dict(base_config)
        bands_config["lookback_window"] = lookback
        recalculated = calculate_excursion_bands(bands_config, pl.from_pandas(bands))
        cache[lookback] = recalculated.to_pandas()
    return cache


def _select_bands(band_cache: dict[int | None, pd.DataFrame], params: dict[str, Any]) -> pd.DataFrame:
    lookback = params.get("features.bands.lookback_window")
    if lookback is None:
        return band_cache[None]
    return band_cache[int(lookback)]


def _hmm_training_window(
    intraday: pd.DataFrame, bands: pd.DataFrame, fold: Fold, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.Timestamp(fold["test_start"]).date()
    warmup_start = (
        config.hmm.warmup_start_date
        if config.hmm is not None and config.hmm.warmup_start_date is not None
        else config.backtest.start_date
    )
    if config.hmm is not None and config.hmm.train_lookback_months is not None:
        train_start = (pd.Timestamp(test_start) - pd.DateOffset(months=config.hmm.train_lookback_months)).date()
        train_start = max(train_start, warmup_start)
    else:
        train_start = warmup_start
    session_dates = pd.to_datetime(intraday["Session"]).dt.date
    mask = (session_dates >= train_start) & (session_dates < test_start)
    hmm_train = intraday[mask]
    train_sessions = set(hmm_train["Session"].dropna().unique())
    hmm_bands = bands[bands["Session"].isin(train_sessions)]
    return hmm_train, hmm_bands


def _patch_config(config: BacktestConfig, params: dict[str, Any]) -> BacktestConfig:
    strategy = config.strategy
    if strategy is None:
        raise ValueError("Cannot patch missing strategy config")

    patched_strategy = strategy
    patched_sizing = config.sizing
    patched_execution = config.execution

    for key, value in params.items():
        if key == "strategy.entry_bars_after_or":
            patched_strategy = replace(patched_strategy, entry_bars_after_or=int(value))
        elif key == "strategy.atr.buffer_mult":
            patched_strategy = replace(
                patched_strategy, atr=replace(patched_strategy.atr, buffer_mult=float(value))
            )
        elif key == "strategy.atr.lookback_sessions":
            patched_strategy = replace(
                patched_strategy, atr=replace(patched_strategy.atr, lookback_sessions=int(value))
            )
        elif key == "strategy.donchian.lookback_bars":
            if patched_strategy.donchian is None:
                raise ValueError("Cannot patch strategy.donchian.lookback_bars without donchian config")
            patched_strategy = replace(
                patched_strategy,
                donchian=replace(patched_strategy.donchian, lookback_bars=int(value)),
            )
        elif key == "strategy.atr_stop.enabled":
            patched_strategy = replace(
                patched_strategy, atr_stop=replace(patched_strategy.atr_stop, enabled=bool(value))
            )
        elif key == "strategy.atr_stop.length":
            patched_strategy = replace(
                patched_strategy, atr_stop=replace(patched_strategy.atr_stop, length=int(value))
            )
        elif key == "strategy.atr_stop.multiplier":
            patched_strategy = replace(
                patched_strategy, atr_stop=replace(patched_strategy.atr_stop, multiplier=float(value))
            )
        elif key == "strategy.take_profit.rr":
            patched_strategy = replace(
                patched_strategy, take_profit=replace(patched_strategy.take_profit, rr=float(value))
            )
        elif key == "strategy.break_even.enabled":
            patched_strategy = replace(
                patched_strategy,
                break_even=replace(patched_strategy.break_even, enabled=bool(value)),
            )
        elif key == "strategy.break_even.trigger_rr":
            patched_strategy = replace(
                patched_strategy,
                break_even=replace(patched_strategy.break_even, trigger_rr=float(value)),
            )
        elif key == "strategy.break_even.offset_points":
            patched_strategy = replace(
                patched_strategy,
                break_even=replace(patched_strategy.break_even, offset_points=float(value)),
            )
        elif key == "sizing.risk_pct":
            patched_sizing = replace(patched_sizing, risk_pct=float(value))
        elif key == "sizing.fixed_units":
            patched_sizing = replace(patched_sizing, fixed_units=float(value))
        elif key == "execution.slippage_ticks_per_side":
            patched_execution = replace(patched_execution, slippage_ticks_per_side=float(value))
        elif key == "features.bands.lookback_window":
            continue
        else:
            raise ValueError(f"Unsupported WFO parameter: {key}")

    return replace(
        config,
        strategy=patched_strategy,
        sizing=patched_sizing,
        execution=patched_execution,
    )


def _build_session_folds(
    sessions: list[object], train_size: int, test_size: int, step_size: int
) -> list[Fold]:
    folds = []
    start = 0
    fold_number = 1
    while start + train_size + test_size <= len(sessions):
        train_sessions = sessions[start : start + train_size]
        test_sessions = sessions[start + train_size : start + train_size + test_size]
        folds.append(
            {
                "fold": fold_number,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
                "train_start": str(train_sessions[0]),
                "train_end": str(train_sessions[-1]),
                "test_start": str(test_sessions[0]),
                "test_end": str(test_sessions[-1]),
            }
        )
        start += step_size
        fold_number += 1
    if not folds:
        raise ValueError("Not enough sessions to build WFO folds")
    return folds


def _build_calendar_folds(
    intraday: pd.DataFrame,
    start_date: pd.Timestamp,
    train_months: int,
    test_months: int,
    step_months: int,
) -> list[Fold]:
    session_dates = (
        intraday[["Session"]]
        .drop_duplicates()
        .assign(SessionDate=lambda df: pd.to_datetime(df["Session"]))
        .sort_values("SessionDate")
    )
    max_date = session_dates["SessionDate"].max()
    folds = []
    fold_number = 1
    cursor = start_date.normalize()
    while True:
        train_start = cursor
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > max_date + pd.Timedelta(days=1):
            break
        train_mask = (session_dates["SessionDate"] >= train_start) & (
            session_dates["SessionDate"] < train_end
        )
        test_mask = (session_dates["SessionDate"] >= train_end) & (
            session_dates["SessionDate"] < test_end
        )
        train_sessions = session_dates.loc[train_mask, "Session"].to_list()
        test_sessions = session_dates.loc[test_mask, "Session"].to_list()
        if train_sessions and test_sessions:
            folds.append(
                {
                    "fold": fold_number,
                    "train_sessions": train_sessions,
                    "test_sessions": test_sessions,
                    "train_start": train_start.date().isoformat(),
                    "train_end": (train_end - pd.Timedelta(days=1)).date().isoformat(),
                    "test_start": train_end.date().isoformat(),
                    "test_end": (test_end - pd.Timedelta(days=1)).date().isoformat(),
                }
            )
            fold_number += 1
        cursor = cursor + pd.DateOffset(months=step_months)
    if not folds:
        raise ValueError("Not enough data to build calendar WFO folds")
    return folds


def _objective_score(metrics: dict[str, Any], objective: str) -> float:
    value = metrics.get(objective)
    if value is None:
        return float("-inf")
    value = float(value)
    if objective == "max_drawdown_pct":
        return -abs(value)
    return value


def _should_optimize(
    config: BacktestConfig,
    variant_label: str,
    needs_optimization: dict[str, bool],
    current_params: dict[str, dict[str, Any]],
) -> bool:
    mode = config.wfo.reoptimization_mode
    if mode == "always":
        return True
    if mode == "degradation":
        return needs_optimization.get(variant_label, True) or variant_label not in current_params
    raise ValueError("wfo.reoptimization_mode must be 'always' or 'degradation'")


def _is_degraded(config: BacktestConfig, degradation_score: float) -> bool:
    if config.wfo.reoptimization_mode != "degradation":
        return False
    return degradation_score < config.wfo.degradation_threshold


def _format_optional_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _row_from_result(
    fold: int,
    split: str,
    variant: str,
    fold_info: Fold,
    params: dict[str, Any],
    result: BacktestResult,
    score: float,
    optimized: bool,
    degradation_score: float | None = None,
    degraded: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "split": split,
        "variant": variant,
        "train_start": fold_info["train_start"],
        "train_end": fold_info["train_end"],
        "test_start": fold_info["test_start"],
        "test_end": fold_info["test_end"],
        "objective_score": score,
        "optimized": optimized,
        "degradation_score": degradation_score,
        "degraded": degraded,
    }
    row.update(result.metrics)
    row.update({f"param.{key}": value for key, value in params.items()})
    return row


def _skipped_optimization_row(
    fold: int,
    variant: str,
    fold_info: Fold,
    params: dict[str, Any],
    score: float | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold,
        "split": "train",
        "variant": variant,
        "train_start": fold_info["train_start"],
        "train_end": fold_info["train_end"],
        "test_start": fold_info["test_start"],
        "test_end": fold_info["test_end"],
        "objective_score": score,
        "optimized": False,
        "degradation_score": None,
        "degraded": False,
    }
    row.update({f"param.{key}": value for key, value in params.items()})
    return row


def _create_wfo_output_dir(config: BacktestConfig) -> Path:
    root, _ = resolve_path(config.reports.output_dir)
    output_dir = root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_wfo"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_wfo_report(
    output_dir: Path,
    config: BacktestConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stitched_metrics: pd.DataFrame,
    hmm_df: pd.DataFrame,
) -> None:
    lines = [
        "# Walk-Forward Optimization Report",
        "",
        f"- objective: {config.wfo.objective}",
        f"- reoptimization_mode: {config.wfo.reoptimization_mode}",
        f"- degradation_objective: {config.wfo.degradation_objective}",
        f"- degradation_threshold: {config.wfo.degradation_threshold}",
        f"- mode: {config.wfo.mode}",
        f"- train_months: {config.wfo.train_months}",
        f"- test_months: {config.wfo.test_months}",
        f"- step_months: {config.wfo.step_months}",
        f"- train_sessions: {config.wfo.train_sessions}",
        f"- test_sessions: {config.wfo.test_sessions}",
        f"- step_sessions: {config.wfo.step_sessions}",
        f"- parameter_combinations: {_parameter_combination_count(train_df, test_df)}",
        "",
        "## Test Fold Summary",
    ]
    if not test_df.empty:
        summary = (
            test_df.groupby("variant")
            .agg(
                folds=("fold", "count"),
                avg_return_pct=("total_return_pct", "mean"),
                total_return_pct=("total_return_pct", "sum"),
                avg_sharpe=("sharpe", "mean"),
                avg_max_drawdown_pct=("max_drawdown_pct", "mean"),
                total_trades=("trade_count", "sum"),
                optimized_folds=("optimized", "sum"),
                degraded_folds=("degraded", "sum"),
            )
            .reset_index()
        )
        lines.extend(["", _dataframe_to_markdown(summary)])
    if not stitched_metrics.empty:
        lines.extend(["", "## Stitched OOS Performance", "", _dataframe_to_markdown(stitched_metrics)])
    if not hmm_df.empty:
        hmm_summary = (
            hmm_df.groupby("variant")
            .agg(
                folds=("fold", "count"),
                train_samples=("train_samples", "sum"),
                test_candidates=("test_candidates", "sum"),
                allowed_trades=("allowed_trades", "sum"),
                rejected_trades=("rejected_trades", "sum"),
            )
            .reset_index()
        )
        hmm_summary["allowed_rate_pct"] = (
            hmm_summary["allowed_trades"] / hmm_summary["test_candidates"] * 100
        )
        lines.extend(["", "## HMM Filter Summary", "", _dataframe_to_markdown(hmm_summary)])
    (output_dir / "wfo_report.md").write_text("\n".join(lines) + "\n")


def _parameter_combination_count(train_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    if train_df.empty or "objective_score" not in train_df:
        return 0
    return int(len(train_df["objective_score"].dropna()) // max(len(test_df), 1))


def _compound_equity_curve(
    equity_curve: pd.DataFrame,
    base_cash: float,
    starting_cash: float,
    fold: int,
    variant: str,
) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame()
    compounded = equity_curve.copy()
    compounded["DateTime"] = pd.to_datetime(compounded["DateTime"])
    compounded["Equity"] = starting_cash + (compounded["Equity"] - base_cash)
    compounded["Cash"] = starting_cash + (compounded["Cash"] - base_cash)
    compounded["Fold"] = fold
    compounded["Variant"] = variant
    return compounded


def _write_stitched_outputs(
    output_dir: Path,
    config: BacktestConfig,
    stitched_equity: dict[str, list[pd.DataFrame]],
    stitched_trades: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    metric_rows = []
    for variant, curves in stitched_equity.items():
        if not curves:
            continue
        equity = pd.concat(curves, ignore_index=True).sort_values("DateTime")
        trades = (
            pd.concat(stitched_trades[variant], ignore_index=True)
            if stitched_trades[variant]
            else pd.DataFrame(columns=["NetPnL", "ExitTime"])
        )
        equity.to_csv(output_dir / f"wfo_oos_equity_{variant}.csv", index=False)
        trades.to_csv(output_dir / f"wfo_oos_trades_{variant}.csv", index=False)

        yearly = calculate_yearly_metrics(equity, trades, config.backtest.initial_cash)
        yearly.to_csv(output_dir / f"wfo_oos_yearly_{variant}.csv", index=False)

        metrics = calculate_metrics(equity, trades, config.backtest.initial_cash)
        metric_rows.append({"variant": variant, **metrics})
        _save_wfo_equity_chart(equity, output_dir / f"wfo_oos_equity_{variant}.png", variant)
    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df.to_csv(output_dir / "wfo_oos_metrics.csv", index=False)
    return metrics_df


def _write_wfo_research_outputs(
    output_dir: Path,
    config: BacktestConfig,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stitched_equity: dict[str, list[pd.DataFrame]],
    stitched_trades: dict[str, list[pd.DataFrame]],
) -> list[str]:
    research = config.reports.wfo_research
    if not research.enabled:
        return []

    research_dir = output_dir / "wfo_research"
    research_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = ["", "## WFO Research Diagnostics"]

    if research.comparative_charts_enabled and config.reports.save_charts:
        _save_comparative_oos_charts(research_dir, stitched_equity, test_df)
        summary_lines.append("- Comparative OOS charts saved under `wfo_research/`.")

    if research.parameter_stability_enabled:
        stability = _parameter_stability_summary(test_df)
        if not stability.empty:
            stability.to_csv(research_dir / "parameter_stability.csv", index=False)
            summary_lines.extend(["", "### Parameter Stability", "", _dataframe_to_markdown(stability)])
            if config.reports.save_charts:
                _save_parameter_stability_charts(research_dir, test_df, stability)

    if research.monte_carlo_enabled:
        mc_paths, mc_summary = _simulate_wfo_monte_carlo(stitched_trades, config)
        if not mc_paths.empty:
            mc_paths.to_csv(research_dir / "wfo_monte_carlo_paths.csv", index=False)
        if not mc_summary.empty:
            mc_summary.to_csv(research_dir / "wfo_monte_carlo_summary.csv", index=False)
            mc_agg = _monte_carlo_method_summary(mc_summary)
            mc_agg.to_csv(research_dir / "wfo_monte_carlo_method_summary.csv", index=False)
            summary_lines.extend(["", "### WFO Monte Carlo", "", _dataframe_to_markdown(mc_agg)])
            if config.reports.save_charts:
                _save_wfo_monte_carlo_charts(research_dir, mc_paths, mc_summary, config)

    return summary_lines


def _append_wfo_research_report(output_dir: Path, summary_lines: list[str]) -> None:
    report_path = output_dir / "wfo_report.md"
    with report_path.open("a") as file:
        file.write("\n".join(summary_lines) + "\n")


def _save_comparative_oos_charts(
    output_dir: Path,
    stitched_equity: dict[str, list[pd.DataFrame]],
    test_df: pd.DataFrame,
) -> None:
    _apply_research_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    for variant, curves in stitched_equity.items():
        if not curves:
            continue
        equity = pd.concat(curves, ignore_index=True).sort_values("DateTime")
        ax.plot(pd.to_datetime(equity["DateTime"]), equity["Equity"], linewidth=1.8, label=variant)
    ax.set_title("Comparative WFO OOS Equity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "comparative_oos_equity.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for variant, curves in stitched_equity.items():
        if not curves:
            continue
        equity = pd.concat(curves, ignore_index=True).sort_values("DateTime")
        drawdown = equity["Equity"].astype(float) / equity["Equity"].astype(float).cummax() - 1.0
        ax.plot(pd.to_datetime(equity["DateTime"]), drawdown * 100, linewidth=1.5, label=variant)
    ax.set_title("Comparative WFO OOS Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "comparative_oos_drawdown.png", dpi=160)
    plt.close(fig)

    if test_df.empty or "total_return_pct" not in test_df:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for variant, group in test_df.groupby("variant"):
        ax.plot(group["fold"], group["total_return_pct"], marker="o", linewidth=1.5, label=variant)
    ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.55)
    ax.set_title("OOS Fold Return by Variant")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Return %")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_dir / "comparative_fold_returns.png", dpi=160)
    plt.close(fig)


def _parameter_stability_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    param_cols = [col for col in test_df.columns if col.startswith("param.")]
    if test_df.empty or not param_cols:
        return pd.DataFrame()
    rows = []
    for variant, variant_df in test_df.groupby("variant"):
        folds = max(len(variant_df), 1)
        for col in param_cols:
            if col not in variant_df or variant_df[col].isna().all():
                continue
            values = variant_df[col].dropna()
            if values.empty:
                continue
            top_value = values.mode(dropna=True).iloc[0]
            changes = int(values.astype(str).ne(values.astype(str).shift()).sum() - 1)
            rows.append(
                {
                    "variant": variant,
                    "parameter": col.removeprefix("param."),
                    "unique_values": int(values.nunique(dropna=True)),
                    "most_common_value": top_value,
                    "most_common_pct": float((values == top_value).mean() * 100),
                    "changes": max(changes, 0),
                    "changes_per_fold": float(max(changes, 0) / folds),
                }
            )
    return pd.DataFrame(rows)


def _save_parameter_stability_charts(
    output_dir: Path, test_df: pd.DataFrame, stability: pd.DataFrame
) -> None:
    _apply_research_style()
    param_cols = [col for col in test_df.columns if col.startswith("param.")]
    if not param_cols:
        return
    for variant, group in test_df.groupby("variant"):
        plot_cols = [col for col in param_cols if col in group and not group[col].isna().all()]
        if not plot_cols:
            continue
        fig, axes = plt.subplots(len(plot_cols), 1, figsize=(12, max(3, 2.5 * len(plot_cols))), sharex=True)
        if len(plot_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, plot_cols):
            series = group[col]
            if pd.api.types.is_numeric_dtype(series):
                ax.plot(group["fold"], series.astype(float), marker="o", linewidth=1.5)
            else:
                codes, labels = pd.factorize(series.astype(str))
                ax.step(group["fold"], codes, where="mid", linewidth=1.5)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels)
            ax.set_ylabel(col.removeprefix("param."))
            ax.grid(alpha=0.16)
        axes[-1].set_xlabel("Fold")
        fig.suptitle(f"Parameter Stability: {variant}", y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / f"parameter_stability_{variant}.png", dpi=160)
        plt.close(fig)

    pivot = stability.pivot_table(
        index="parameter", columns="variant", values="most_common_pct", aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(pivot.columns)), max(4, 0.45 * len(pivot))))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Parameter Stability Heatmap (Most Common %)" )
    fig.colorbar(image, ax=ax, label="Most Common %")
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_stability_heatmap.png", dpi=160)
    plt.close(fig)


def _simulate_wfo_monte_carlo(
    stitched_trades: dict[str, list[pd.DataFrame]], config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    research = config.reports.wfo_research
    rng = np.random.default_rng(research.monte_carlo_seed)
    path_rows = []
    summary_rows = []
    for variant, chunks in stitched_trades.items():
        if not chunks:
            continue
        trades = pd.concat(chunks, ignore_index=True)
        if trades.empty or "NetPnL" not in trades:
            continue
        pnls = trades["NetPnL"].to_numpy(dtype=float)
        sample_trades = research.monte_carlo_sample_trades or len(pnls)
        sample_trades = max(1, min(int(sample_trades), len(pnls)))
        for method in research.monte_carlo_methods:
            for simulation in range(research.monte_carlo_simulations):
                sample = _monte_carlo_sample(pnls, method, sample_trades, research.monte_carlo_dropout_pct, rng)
                if len(sample) == 0:
                    continue
                equity = config.backtest.initial_cash + np.cumsum(sample)
                drawdown = _drawdown_from_equity(equity)
                terminal_return = (equity[-1] / config.backtest.initial_cash - 1.0) * 100
                summary_rows.append(
                    {
                        "variant": variant,
                        "method": method,
                        "simulation": simulation,
                        "terminal_equity": float(equity[-1]),
                        "total_return_pct": float(terminal_return),
                        "max_drawdown_pct": float(drawdown.min() * 100),
                        "trade_count": int(len(sample)),
                    }
                )
                for step, value in enumerate(equity, start=1):
                    path_rows.append(
                        {
                            "variant": variant,
                            "method": method,
                            "simulation": simulation,
                            "trade_step": step,
                            "equity": float(value),
                        }
                    )
    return pd.DataFrame(path_rows), pd.DataFrame(summary_rows)


def _monte_carlo_sample(
    pnls: np.ndarray,
    method: str,
    sample_trades: int,
    dropout_pct: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if method == "bootstrap":
        return rng.choice(pnls, size=sample_trades, replace=True)
    if method == "reshuffle":
        return rng.permutation(pnls)[:sample_trades]
    if method == "dropout":
        shuffled = rng.permutation(pnls)
        keep_count = max(1, int(round(sample_trades * (1.0 - dropout_pct))))
        return shuffled[:keep_count]
    raise ValueError(f"Unsupported WFO Monte Carlo method: {method}")


def _drawdown_from_equity(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def _monte_carlo_method_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(["variant", "method"])
        .agg(
            simulations=("simulation", "count"),
            median_return_pct=("total_return_pct", "median"),
            p05_return_pct=("total_return_pct", lambda value: value.quantile(0.05)),
            p95_return_pct=("total_return_pct", lambda value: value.quantile(0.95)),
            worst_return_pct=("total_return_pct", "min"),
            best_return_pct=("total_return_pct", "max"),
            median_max_drawdown_pct=("max_drawdown_pct", "median"),
            worst_max_drawdown_pct=("max_drawdown_pct", "min"),
        )
        .reset_index()
    )


def _save_wfo_monte_carlo_charts(
    output_dir: Path, paths: pd.DataFrame, summary: pd.DataFrame, config: BacktestConfig
) -> None:
    if paths.empty or summary.empty:
        return
    _apply_research_style()
    max_paths = config.reports.wfo_research.monte_carlo_max_paths_plotted
    for (variant, method), group_summary in summary.groupby(["variant", "method"]):
        group_paths = paths[(paths["variant"] == variant) & (paths["method"] == method)]
        highlight = _monte_carlo_highlight_simulations(group_summary)
        plotted_sims = set(group_summary["simulation"].head(max_paths).astype(int)) | set(highlight.values())
        fig, ax = plt.subplots(figsize=(12, 6))
        for simulation, path in group_paths[group_paths["simulation"].isin(plotted_sims)].groupby("simulation"):
            role = next((name for name, sim in highlight.items() if sim == simulation), None)
            if role is None:
                ax.plot(path["trade_step"], path["equity"], color="#64748b", alpha=0.12, linewidth=0.8)
                continue
            color = {"worst": "#dc2626", "median": "#2563eb", "best": "#16a34a"}[role]
            ax.plot(path["trade_step"], path["equity"], color=color, linewidth=2.6, label=role.title())
        ax.axhline(config.backtest.initial_cash, color="#111827", linewidth=1.0, alpha=0.5)
        ax.set_title(f"WFO Monte Carlo Paths: {variant} / {method}")
        ax.set_xlabel("Trade Step")
        ax.set_ylabel("Equity")
        ax.legend(frameon=False)
        ax.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(output_dir / f"monte_carlo_paths_{variant}_{method}.png", dpi=160)
        plt.close(fig)

    labels = []
    values = []
    for (variant, method), group in summary.groupby(["variant", "method"]):
        labels.append(f"{variant}\n{method}")
        values.append(group["total_return_pct"].to_numpy(dtype=float))
    if values:
        fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(values)), 5))
        ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
        ax.axhline(0, color="#111827", linewidth=1.0, alpha=0.55)
        ax.set_title("WFO Monte Carlo Terminal Return Distribution")
        ax.set_ylabel("Total Return %")
        ax.grid(axis="y", alpha=0.18)
        fig.tight_layout()
        fig.savefig(output_dir / "monte_carlo_terminal_return_comparison.png", dpi=160)
        plt.close(fig)


def _monte_carlo_highlight_simulations(summary: pd.DataFrame) -> dict[str, int]:
    ordered = summary.sort_values("total_return_pct").reset_index(drop=True)
    median_index = int(len(ordered) // 2)
    return {
        "worst": int(ordered.iloc[0]["simulation"]),
        "median": int(ordered.iloc[median_index]["simulation"]),
        "best": int(ordered.iloc[-1]["simulation"]),
    }


def _apply_research_style() -> None:
    plt.rcParams.update(
        {
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#111827",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "font.size": 10,
        }
    )


def _save_wfo_equity_chart(equity: pd.DataFrame, path: Path, variant: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(pd.to_datetime(equity["DateTime"]), equity["Equity"], linewidth=1.25)
    ax.set_title(f"WFO OOS Equity: {variant}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _print_wfo_summary(test_df: pd.DataFrame, output_dir: Path) -> None:
    print("\nWFO Summary")
    print("-----------")
    if test_df.empty:
        print("No test folds generated")
    else:
        summary = test_df.groupby("variant").agg(
            folds=("fold", "count"),
            avg_return_pct=("total_return_pct", "mean"),
            total_return_pct=("total_return_pct", "sum"),
            avg_sharpe=("sharpe", "mean"),
            total_trades=("trade_count", "sum"),
        )
        print(summary.to_string(float_format=lambda value: f"{value:.2f}"))
    print(f"Output: {output_dir}")
