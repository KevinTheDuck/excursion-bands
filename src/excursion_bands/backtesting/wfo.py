"""
Walk-forward optimization for ORB variants.
"""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from excursion_bands.backtesting.models import BacktestResult
from excursion_bands.backtesting.metrics import calculate_metrics, calculate_yearly_metrics
from excursion_bands.backtesting.ml import train_ml_filter, write_ml_artifacts
from excursion_bands.backtesting.specification import BacktestConfig, WfoConfig
from excursion_bands.backtesting.strategies.orb import prepare_orb_bars, run_orb_variant
from excursion_bands.paths import resolve_path

type Fold = dict[str, Any]


def validate_parameter_sweep(parameters: Sequence[object], config: WfoConfig) -> None:
    if len(parameters) > config.max_parameter_combinations:
        raise ValueError(
            f"Parameter sweep has {len(parameters)} combinations, exceeding "
            f"configured max_parameter_combinations={config.max_parameter_combinations}."
        )
    if config.max_workers < 1:
        raise ValueError("wfo.max_workers must be >= 1")


def run_wfo(
    intraday: pd.DataFrame,
    bands: pd.DataFrame,
    config: BacktestConfig,
) -> Path:
    if config.strategy is None:
        raise ValueError("WFO requires a strategy config")
    if config.wfo.max_workers != 1:
        raise ValueError("WFO currently supports max_workers: 1 to avoid host overload")

    parameter_sets = _build_parameter_sets(config.wfo)
    validate_parameter_sweep(parameter_sets, config.wfo)

    variants = config.variants
    if not variants:
        raise ValueError("WFO requires at least one active variant")
    intraday = intraday.sort_values("DateTime").reset_index(drop=True).copy()
    intraday["DateTime"] = pd.to_datetime(intraday["DateTime"])
    warmup_start = (
        config.ml.warmup_start_date
        if config.ml is not None and config.ml.warmup_start_date is not None
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
    ml_rows: list[dict[str, Any]] = []
    stitched_equity: dict[str, list[pd.DataFrame]] = {variant.label: [] for variant in variants}
    stitched_trades: dict[str, list[pd.DataFrame]] = {variant.label: [] for variant in variants}
    variant_capital = {variant.label: float(config.backtest.initial_cash) for variant in variants}

    for variant in variants:
        print(f"\nWFO Variant: {variant.label}")
        for fold in folds:
            fold_number = fold["fold"]
            train_sessions = fold["train_sessions"]
            test_sessions = fold["test_sessions"]
            train_data = intraday[intraday["Session"].isin(train_sessions)]
            train_bands = bands[bands["Session"].isin(train_sessions)]
            best_result: BacktestResult | None = None
            best_params: dict[str, Any] | None = None
            best_score: float | None = None

            for params in parameter_sets:
                trial_config = _patch_config(config, params)
                trial_prepared = prepare_orb_bars(train_data, trial_config, train_bands)
                result = run_orb_variant(trial_prepared, trial_config, variant)
                score = _objective_score(result.metrics, config.wfo.objective)
                train_rows.append(
                    _row_from_result(
                        fold_number,
                        "train",
                        variant.label,
                        fold,
                        params,
                        result,
                        score,
                    )
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_result = result
                    best_params = params

            if best_result is None or best_params is None:
                continue

            test_config = _patch_config(config, best_params)
            test_data = intraday[intraday["Session"].isin(test_sessions)]
            test_bands = bands[bands["Session"].isin(test_sessions)]
            test_prepared = prepare_orb_bars(test_data, test_config, test_bands)
            allowed_signal_times = None
            if variant.use_ml_filter:
                ml_train_data, ml_train_bands = _ml_training_window(
                    intraday, bands, fold, config
                )
                ml_train_prepared = prepare_orb_bars(ml_train_data, test_config, ml_train_bands)
                ml_result = train_ml_filter(
                    ml_train_prepared, test_prepared, test_config, variant
                )
                write_ml_artifacts(output_dir, variant.label, fold_number, ml_result)
                allowed_signal_times = ml_result.allowed_signal_times
                ml_rows.append(
                    {
                        "fold": fold_number,
                        "variant": variant.label,
                        "train_start": fold["train_start"],
                        "train_end": fold["train_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                        **ml_result.diagnostics,
                    }
                )
            test_result = run_orb_variant(
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
            test_rows.append(
                _row_from_result(
                    fold_number,
                    "test",
                    variant.label,
                    fold,
                    best_params,
                    test_result,
                    test_score,
                )
            )
            print(
                f"fold={fold_number} variant={variant.label} "
                f"train_{config.wfo.objective}={best_score:.4f} "
                f"test_return={test_result.metrics.get('total_return_pct'):.2f}% "
                f"params={best_params}"
            )

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    ml_df = pd.DataFrame(ml_rows)
    train_df.to_csv(output_dir / "wfo_train_trials.csv", index=False)
    test_df.to_csv(output_dir / "wfo_test_folds.csv", index=False)
    if not ml_df.empty:
        ml_df.to_csv(output_dir / "ml_wfo_diagnostics.csv", index=False)
    stitched_metrics = _write_stitched_outputs(
        output_dir, config, stitched_equity, stitched_trades
    )
    _write_wfo_report(output_dir, config, train_df, test_df, stitched_metrics, ml_df)
    _print_wfo_summary(test_df, output_dir)
    return output_dir


def _build_parameter_sets(wfo: WfoConfig) -> list[dict[str, Any]]:
    grid = wfo.parameter_grid or {}
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [value if isinstance(value, list) else [value] for value in grid.values()]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _ml_training_window(
    intraday: pd.DataFrame, bands: pd.DataFrame, fold: Fold, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_start = pd.Timestamp(fold["test_start"]).date()
    warmup_start = (
        config.ml.warmup_start_date
        if config.ml is not None and config.ml.warmup_start_date is not None
        else config.backtest.start_date
    )
    if config.ml is not None and config.ml.train_lookback_months is not None:
        train_start = (pd.Timestamp(test_start) - pd.DateOffset(months=config.ml.train_lookback_months)).date()
        train_start = max(train_start, warmup_start)
    else:
        train_start = warmup_start
    session_dates = pd.to_datetime(intraday["Session"]).dt.date
    mask = (session_dates >= train_start) & (session_dates < test_start)
    ml_train = intraday[mask]
    train_sessions = set(ml_train["Session"].dropna().unique())
    ml_bands = bands[bands["Session"].isin(train_sessions)]
    return ml_train, ml_bands


def _patch_config(config: BacktestConfig, params: dict[str, Any]) -> BacktestConfig:
    strategy = config.strategy
    if strategy is None:
        raise ValueError("Cannot patch missing strategy config")

    patched_strategy = strategy
    patched_sizing = config.sizing
    patched_execution = config.execution
    patched_ml = config.ml

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
        elif key == "ml.probability_threshold":
            if patched_ml is None:
                raise ValueError("Cannot patch ml.probability_threshold without ml config")
            patched_ml = replace(patched_ml, probability_threshold=float(value))
        elif key == "ml.min_train_samples":
            if patched_ml is None:
                raise ValueError("Cannot patch ml.min_train_samples without ml config")
            patched_ml = replace(patched_ml, min_train_samples=int(value))
        else:
            raise ValueError(f"Unsupported WFO parameter: {key}")

    return replace(
        config,
        strategy=patched_strategy,
        sizing=patched_sizing,
        execution=patched_execution,
        ml=patched_ml,
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


def _row_from_result(
    fold: int,
    split: str,
    variant: str,
    fold_info: Fold,
    params: dict[str, Any],
    result: BacktestResult,
    score: float,
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
    }
    row.update(result.metrics)
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
    ml_df: pd.DataFrame,
) -> None:
    lines = [
        "# Walk-Forward Optimization Report",
        "",
        f"- objective: {config.wfo.objective}",
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
            )
            .reset_index()
        )
        lines.extend(["", _dataframe_to_markdown(summary)])
    if not stitched_metrics.empty:
        lines.extend(["", "## Stitched OOS Performance", "", _dataframe_to_markdown(stitched_metrics)])
    if not ml_df.empty:
        ml_summary = (
            ml_df.groupby("variant")
            .agg(
                folds=("fold", "count"),
                train_samples=("train_samples", "sum"),
                test_candidates=("test_candidates", "sum"),
                allowed_trades=("allowed_trades", "sum"),
                rejected_trades=("rejected_trades", "sum"),
            )
            .reset_index()
        )
        ml_summary["allowed_rate_pct"] = (
            ml_summary["allowed_trades"] / ml_summary["test_candidates"] * 100
        )
        lines.extend(["", "## ML Filter Summary", "", _dataframe_to_markdown(ml_summary)])
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
