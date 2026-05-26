"""
Core backtesting data models.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass(frozen=True)
class Trade:
    entry_time: datetime
    exit_time: datetime
    side: str
    units: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    commission: float
    net_pnl: float
    return_pct: float
    exit_reason: str


@dataclass(frozen=True)
class Position:
    side: str
    units: float
    entry_time: datetime
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    entry_commission: float


@dataclass(frozen=True)
class BacktestResult:
    name: str
    metrics: dict[str, float | int | str | None]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    config_summary: dict[str, float | int | str | None]
