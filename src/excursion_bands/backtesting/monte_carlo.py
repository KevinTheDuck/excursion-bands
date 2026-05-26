"""
Monte Carlo analysis for trade-level outcomes.
"""

import numpy as np
import pandas as pd


def simulate_trade_bootstrap(
    trades: pd.DataFrame,
    initial_cash: float,
    simulations: int,
    sample_trades: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty or simulations <= 0 or sample_trades <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    pnls = trades["NetPnL"].to_numpy(dtype=float)
    path_rows = []
    summary_rows = []
    for simulation in range(simulations):
        sample = rng.choice(pnls, size=sample_trades, replace=True)
        equity = initial_cash + np.cumsum(sample)
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        for step, value in enumerate(equity, start=1):
            path_rows.append(
                {
                    "simulation": simulation,
                    "trade_step": step,
                    "equity": float(value),
                }
            )
        summary_rows.append(
            {
                "simulation": simulation,
                "terminal_equity": float(equity[-1]),
                "total_return_pct": float((equity[-1] / initial_cash - 1.0) * 100),
                "max_drawdown_pct": float(drawdown.min() * 100),
            }
        )
    return pd.DataFrame(path_rows), pd.DataFrame(summary_rows)
