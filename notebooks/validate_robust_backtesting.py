"""Small real-data integration run, not a performance-validation experiment."""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from excursion_bands.backtesting.loader import load_backtest_config, load_core_data
from excursion_bands.backtesting.wfo import run_wfo


def main():
    config = load_backtest_config("configs/strategies/donchian_1/backtest_robust.yaml")
    output = Path("data/output/implementation_validation") / datetime.now(UTC).strftime(
        "%Y%m%d_%H%M%S_robust"
    )
    config = replace(
        config,
        backtest=replace(
            config.backtest, start_date=date(2024, 1, 1), end_date=date(2024, 6, 30)
        ),
        wfo=replace(
            config.wfo,
            n_trials=2,
            checkpoint_dir=str(output),
            max_run_seconds=1800,
            robust=replace(config.wfo.robust, shortlist_size=1),
        ),
    )
    started = perf_counter()
    intraday, bands = load_core_data(config.core_data)
    result = run_wfo(intraday.to_pandas(), bands.to_pandas(), config)
    metrics = pd.read_csv(result / "wfo_oos_metrics.csv")
    decisions = pd.read_csv(result / "session_decisions.csv")
    for label, daily in decisions.groupby("variant", sort=False):
        daily = daily.sort_values("session")
        np.testing.assert_allclose(
            daily.starting_cash.iloc[1:], daily.ending_cash.iloc[:-1]
        )
        trades_path = result / f"wfo_oos_trades_{label}.csv"
        try:
            trades = pd.read_csv(trades_path)
        except pd.errors.EmptyDataError:
            trades = pd.DataFrame()
        net = 0.0 if trades.empty else trades.NetPnL.sum()
        np.testing.assert_allclose(
            daily.ending_cash.iloc[-1] - config.backtest.initial_cash, net, atol=1e-6
        )
        assert not daily.session.duplicated().any()
        assert daily.fold.nunique() == 2
    print(metrics.to_string(index=False))
    print(f"Real-data integration passed in {perf_counter() - started:.1f}s: {result}")


if __name__ == "__main__":
    main()
