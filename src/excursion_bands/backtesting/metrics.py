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


def calculate_yearly_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    initial_cash: float,
) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame()

    equity = equity_curve.copy()
    equity["DateTime"] = pd.to_datetime(equity["DateTime"])
    equity["Year"] = equity["DateTime"].dt.year
    trades = trades.copy()
    if not trades.empty:
        trades["ExitTime"] = pd.to_datetime(trades["ExitTime"])
        trades["Year"] = trades["ExitTime"].dt.year

    rows = []
    previous_year_end_equity = initial_cash
    for year, year_equity in equity.groupby("Year", sort=True):
        start_equity = previous_year_end_equity
        end_equity = float(year_equity["Equity"].iloc[-1])
        previous_year_end_equity = end_equity
        year_trades = trades[trades["Year"] == year] if not trades.empty else trades
        wins = year_trades[year_trades["NetPnL"] > 0] if not year_trades.empty else year_trades
        losses = year_trades[year_trades["NetPnL"] < 0] if not year_trades.empty else year_trades
        gross_profit = float(wins["NetPnL"].sum()) if not year_trades.empty else 0.0
        gross_loss = float(abs(losses["NetPnL"].sum())) if not year_trades.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        win_rate = len(wins) / len(year_trades) if len(year_trades) else None

        rows.append(
            {
                "Year": int(year),
                "StartEquity": float(start_equity),
                "EndEquity": end_equity,
                "ReturnPct": float((end_equity / start_equity - 1.0) * 100),
                "MaxDrawdownPct": float(max_drawdown(year_equity["Equity"].astype(float)) * 100),
                "Trades": int(len(year_trades)),
                "WinRatePct": None if win_rate is None else float(win_rate * 100),
                "ProfitFactor": profit_factor,
                "NetPnL": float(year_trades["NetPnL"].sum()) if not year_trades.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)
