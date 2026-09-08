"""
Backtest artifact and terminal reporting.
"""

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from excursion_bands.backtesting.metrics import calculate_yearly_metrics
from excursion_bands.backtesting.models import BacktestResult
from excursion_bands.backtesting.monte_carlo import simulate_trade_bootstrap
from excursion_bands.backtesting.specification import BacktestConfig
from excursion_bands.paths import resolve_path


def _format_metric(value: float | str | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    columns = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        values = [_format_metric(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save_charts(
    result: BacktestResult,
    output_dir: Path,
    monte_carlo_paths: pd.DataFrame,
    monte_carlo_summary: pd.DataFrame,
    max_paths_plotted: int,
) -> None:
    equity = result.equity_curve.copy()
    equity["DateTime"] = pd.to_datetime(equity["DateTime"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(equity["DateTime"], equity["Equity"], linewidth=1.3)
    ax.set_title(f"{result.name} Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=140)
    plt.close(fig)

    initial_equity = float(result.metrics.get("initial_cash", equity["Equity"].iloc[0]))
    path_equity = pd.concat(
        [pd.Series([initial_equity]), equity["Equity"].astype(float).reset_index(drop=True)],
        ignore_index=True,
    )
    drawdown = (path_equity / path_equity.cummax() - 1.0).iloc[1:]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(equity["DateTime"], drawdown * 100, 0, alpha=0.35)
    ax.set_title(f"{result.name} Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "drawdown.png", dpi=140)
    plt.close(fig)

    if not monte_carlo_paths.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        plotted = 0
        for _simulation, path in monte_carlo_paths.groupby("simulation"):
            if plotted >= max_paths_plotted:
                break
            ax.plot(path["trade_step"], path["equity"], linewidth=0.8, alpha=0.18, color="#1f77b4")
            plotted += 1
        ax.axhline(result.metrics["initial_cash"], color="black", linewidth=1.0, alpha=0.55)
        ax.set_title(f"Monte Carlo Equity Paths ({plotted} plotted)")
        ax.set_xlabel("Sampled Trade Step")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "monte_carlo.png", dpi=140)
        plt.close(fig)

    if not monte_carlo_summary.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(monte_carlo_summary["total_return_pct"], bins=40, alpha=0.8)
        ax.set_title("Monte Carlo Terminal Return Distribution")
        ax.set_xlabel("Total Return %")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "monte_carlo_distribution.png", dpi=140)
        plt.close(fig)


def save_report(result: BacktestResult, config: BacktestConfig) -> Path:
    root, _ = resolve_path(config.reports.output_dir)
    run_id = f"{datetime.now(UTC).astimezone().strftime('%Y%m%d_%H%M%S')}_{result.name}"
    output_dir = root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    result.equity_curve.to_csv(output_dir / "equity_curve.csv", index=False)
    result.trades.to_csv(output_dir / "trades.csv", index=False)
    result_initial_cash = float(result.metrics.get("initial_cash", config.backtest.initial_cash))
    yearly = calculate_yearly_metrics(
        result.equity_curve, result.trades, result_initial_cash
    )
    yearly.to_csv(output_dir / "yearly_metrics.csv", index=False)

    with (output_dir / "metrics.yaml").open("w") as f:
        yaml.safe_dump(result.metrics, f, sort_keys=False)

    monte_carlo_paths = pd.DataFrame()
    monte_carlo_summary = pd.DataFrame()
    if config.reports.monte_carlo.enabled:
        monte_carlo_paths, monte_carlo_summary = simulate_trade_bootstrap(
            result.trades,
            float(result.metrics.get("initial_cash", config.backtest.initial_cash)),
            config.reports.monte_carlo.simulations,
            config.reports.monte_carlo.sample_trades,
            config.reports.monte_carlo.seed,
        )
        if not monte_carlo_paths.empty:
            monte_carlo_paths.to_csv(output_dir / "monte_carlo_paths.csv", index=False)
        if not monte_carlo_summary.empty:
            monte_carlo_summary.to_csv(output_dir / "monte_carlo_summary.csv", index=False)

    if config.reports.save_charts:
        _save_charts(
            result,
            output_dir,
            monte_carlo_paths,
            monte_carlo_summary,
            config.reports.monte_carlo.max_paths_plotted,
        )

    lines = [
        f"# Backtest Report: {result.name}",
        "",
        "## Metrics",
    ]
    for key, value in result.metrics.items():
        lines.append(f"- {key}: {_format_metric(value)}")
    lines.extend(["", "## Execution Assumptions"])
    for key, value in result.config_summary.items():
        lines.append(f"- {key}: {_format_metric(value)}")
    if not yearly.empty:
        lines.extend(["", "## Yearly Metrics", "", _dataframe_to_markdown(yearly)])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    return output_dir


def print_receipt(result: BacktestResult, output_dir: Path) -> None:
    m = result.metrics
    print("\nBacktest Receipt")
    print("----------------")
    print(f"Strategy:       {result.name}")
    print(f"Final Equity:   {_format_metric(m.get('final_equity'))}")
    print(f"Total Return:   {_format_metric(m.get('total_return_pct'))}%")
    print(f"CAGR:           {_format_metric(m.get('cagr_pct'))}%")
    print(f"Max Drawdown:   {_format_metric(m.get('max_drawdown_pct'))}%")
    print(f"Sharpe:         {_format_metric(m.get('sharpe'))}")
    print(f"Sortino:        {_format_metric(m.get('sortino'))}")
    print(f"Win Rate:       {_format_metric(m.get('win_rate_pct'))}%")
    print(f"Profit Factor:  {_format_metric(m.get('profit_factor'))}")
    print(f"Trades:         {_format_metric(m.get('trade_count'))}")
    print(f"Exposure:       {_format_metric(m.get('exposure_pct'))}%")
    print(f"Output:         {output_dir}")
