"""
Performance metrics for backtest results.
"""

import math

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> dict[str, float | int | str | None]:
    if equity_curve.empty:
        return {}

    equity = equity_curve["Equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = equity.iloc[-1] / initial_cash - 1.0

    start = pd.to_datetime(equity_curve["DateTime"].iloc[0])
    end = pd.to_datetime(equity_curve["DateTime"].iloc[-1])
    years = max((end - start).total_seconds() / (365.25 * 24 * 60 * 60), 1 / 365.25)
    cagr = (equity.iloc[-1] / initial_cash) ** (1 / years) - 1.0

    periods_per_year = len(returns) / years if years > 0 else 0.0
    sharpe = None
    sortino = None
    if not returns.empty and returns.std(ddof=0) > 0 and periods_per_year > 0:
        sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(periods_per_year))
    downside = returns[returns < 0]
    if not downside.empty and downside.std(ddof=0) > 0 and periods_per_year > 0:
        sortino = float((returns.mean() / downside.std(ddof=0)) * math.sqrt(periods_per_year))

    trade_count = int(len(trades))
    wins = trades[trades["NetPnL"] > 0] if trade_count else trades
    losses = trades[trades["NetPnL"] < 0] if trade_count else trades
    gross_profit = float(wins["NetPnL"].sum()) if trade_count else 0.0
    gross_loss = float(abs(losses["NetPnL"].sum())) if trade_count else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    win_rate = len(wins) / trade_count if trade_count else None
    expectancy = float(trades["NetPnL"].mean()) if trade_count else None

    exposure = float(equity_curve["InPosition"].mean()) if "InPosition" in equity_curve else 0.0

    return {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity.iloc[-1]),
        "total_return_pct": float(total_return * 100),
        "cagr_pct": float(cagr * 100),
        "max_drawdown_pct": float(max_drawdown(equity) * 100),
        "sharpe": sharpe,
        "sortino": sortino,
        "trade_count": trade_count,
        "win_rate_pct": None if win_rate is None else float(win_rate * 100),
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "exposure_pct": float(exposure * 100),
    }
