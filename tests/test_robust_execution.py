from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from excursion_bands.backtesting.loader import load_backtest_config
from excursion_bands.backtesting.models import BacktestResult
from excursion_bands.backtesting.robust_execution import (
    ParameterEvaluator,
    session_returns,
)
from excursion_bands.backtesting.specification import VariantConfig
from excursion_bands.backtesting.strategies.donchian import (
    prepare_donchian_bars,
    run_donchian_variant,
)
from excursion_bands.backtesting.wfo import _patch_config


def _config(sizing_mode: str = "fixed_units"):
    with redirect_stdout(StringIO()):
        config = load_backtest_config(
            "configs/strategies/donchian_1/backtest_default.yaml"
        )
    strategy = replace(
        config.strategy,
        donchian=replace(config.strategy.donchian, lookback_bars=3),
        atr=replace(config.strategy.atr, lookback_sessions=2),
        atr_stop=replace(config.strategy.atr_stop, length=2, multiplier=2.0),
    )
    return replace(
        config,
        strategy=strategy,
        execution=replace(
            config.execution,
            commission_per_contract_side=0.0,
            slippage_ticks_per_side=0.0,
        ),
        sizing=replace(
            config.sizing,
            mode=sizing_mode,
            fixed_units=1.0,
            risk_pct=0.01,
        ),
    )


def _market(session_count: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = pd.date_range("2025-01-02", periods=session_count, freq="D")
    rows: list[dict] = []
    for number, session in enumerate(sessions):
        base = 100.0 + number * 3.0
        timestamps = [
            session - pd.Timedelta(days=1) + pd.Timedelta(hours=18),
            session + pd.Timedelta(hours=8, minutes=30),
            session + pd.Timedelta(hours=10),
            session + pd.Timedelta(hours=16),
        ]
        closes = [base, base + 1.0, base + 2.0, base + 3.0]
        for timestamp, close in zip(timestamps, closes, strict=True):
            rows.append(
                {
                    "DateTime": timestamp,
                    "Session": session,
                    "Open": close - 0.2,
                    "High": close + 0.5,
                    "Low": close - 0.5,
                    "Close": close,
                    "Volume": 10.0,
                }
            )
    intraday = pd.DataFrame(rows)
    bands = pd.DataFrame(
        {
            "Session": sessions,
            "Band_AE_Pos_Upper": 0.0,
            "Band_FE_Pos_Lower": 10_000.0,
            "Band_AE_Neg_Lower": 0.0,
            "Band_FE_Neg_Upper": -10_000.0,
        }
    )
    return intraday, bands


class ParameterEvaluatorTests(unittest.TestCase):
    def test_matches_standard_runner_including_band_lookback(self):
        intraday, bands = _market()
        config = _config()
        variant = VariantConfig(
            label="bands", side_mode="long_only", use_band_filter=True
        )
        params = {
            "strategy.donchian.lookback_bars": 5,
            "features.bands.lookback_window": 10,
            "strategy.atr_stop.multiplier": 1.25,
            "strategy.take_profit.rr": 2.5,
            "strategy.break_even.enabled": True,
        }
        recalculated = bands.copy()

        def band_cache(_bands, _parameter_sets, _config, _variant):
            return {None: bands, 10: recalculated}

        with patch(
            "excursion_bands.backtesting.wfo._build_band_cache",
            side_effect=band_cache,
        ):
            evaluator = ParameterEvaluator(intraday, bands, config, variant)
            actual = evaluator.run(params, intraday["Session"].unique())

            expected_config = _patch_config(config, params)
            expected_config = replace(
                expected_config,
                backtest=replace(
                    expected_config.backtest,
                    start_date=intraday["DateTime"].min().date(),
                    end_date=None,
                ),
            )
            prepared = prepare_donchian_bars(
                intraday, expected_config, recalculated
            )
            expected = run_donchian_variant(
                prepared,
                expected_config,
                variant,
                starting_cash=config.backtest.initial_cash,
                session_filter=set(pd.to_datetime(intraday["Session"]).dt.date),
            )

        pd.testing.assert_frame_equal(
            actual.equity_curve.reset_index(drop=True),
            expected.equity_curve.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            actual.trades.drop(columns="Session").reset_index(drop=True),
            expected.trades.reset_index(drop=True),
        )
        self.assertTrue(actual.trades["Session"].notna().all())
        self.assertEqual(actual.metrics, expected.metrics)

    def test_session_by_session_matches_full_run(self):
        intraday, bands = _market()
        config = _config("risk_current_equity")
        variant = VariantConfig(label="raw", side_mode="long_only")
        evaluator = ParameterEvaluator(intraday, bands, config, variant)
        sessions = list(pd.to_datetime(intraday["Session"].unique()).date)
        full = evaluator.run({}, sessions)

        capital = config.backtest.initial_cash
        net_pnl = 0.0
        trade_count = 0
        for session in sessions:
            result = evaluator.run({}, [session], cash=capital)
            capital = float(result.metrics["final_equity"])
            net_pnl += float(result.trades["NetPnL"].sum())
            trade_count += len(result.trades)

        self.assertAlmostEqual(capital, full.metrics["final_equity"])
        self.assertAlmostEqual(net_pnl, full.trades["NetPnL"].sum())
        self.assertEqual(trade_count, len(full.trades))

    def test_initial_and_current_equity_sizing_keep_correct_anchor(self):
        intraday, bands = _market()
        variant = VariantConfig(label="raw", side_mode="long_only")
        sessions = list(pd.to_datetime(intraday["Session"].unique()).date)

        initial_evaluator = ParameterEvaluator(
            intraday, bands, _config("risk_initial_equity"), variant
        )
        initial_100 = initial_evaluator.run({}, sessions, cash=100_000.0)
        initial_200 = initial_evaluator.run({}, sessions, cash=200_000.0)
        self.assertGreater(len(initial_100.trades), 0)
        self.assertAlmostEqual(
            initial_100.trades["Units"].iloc[0], initial_200.trades["Units"].iloc[0]
        )

        current_evaluator = ParameterEvaluator(
            intraday, bands, _config("risk_current_equity"), variant
        )
        current_100 = current_evaluator.run({}, sessions, cash=100_000.0)
        current_200 = current_evaluator.run({}, sessions, cash=200_000.0)
        self.assertAlmostEqual(
            current_200.trades["Units"].iloc[0],
            current_100.trades["Units"].iloc[0] * 2.0,
        )

    def test_future_prices_do_not_change_earlier_features(self):
        intraday, bands = _market()
        config = _config()
        variant = VariantConfig(label="raw", side_mode="long_only")
        original = ParameterEvaluator(intraday, bands, config, variant)._prepare(
            {"strategy.donchian.lookback_bars": 5}, variant
        )
        changed = intraday.copy()
        last_session = changed["Session"].max()
        future = changed["Session"] == last_session
        changed.loc[future, ["High", "Low", "Close"]] *= 3.0
        revised = ParameterEvaluator(changed, bands, config, variant)._prepare(
            {"strategy.donchian.lookback_bars": 5}, variant
        )
        earlier = original["Session"] < pd.Timestamp(last_session).date()
        pd.testing.assert_frame_equal(
            original.loc[earlier, ["Donchian_High", "Donchian_Low", "ATR_Stop"]],
            revised.loc[earlier, ["Donchian_High", "Donchian_Low", "ATR_Stop"]],
        )

    def test_nonpositive_cash_and_one_bar_are_flat(self):
        intraday, bands = _market()
        config = _config()
        variant = VariantConfig(label="raw", side_mode="long_only")
        evaluator = ParameterEvaluator(intraday, bands, config, variant)
        session = pd.Timestamp(intraday["Session"].iloc[0]).date()
        bankrupt = evaluator.run({}, [session], cash=-1.0)
        self.assertEqual(bankrupt.metrics["final_equity"], -1.0)
        self.assertTrue(bankrupt.trades.empty)

        one_bar = ParameterEvaluator(intraday.iloc[:1], bands, config, variant).run(
            {}, [session]
        )
        self.assertEqual(one_bar.metrics["total_return_pct"], 0.0)
        self.assertTrue(one_bar.trades.empty)


class SessionReturnTests(unittest.TestCase):
    def test_uses_session_mapping_across_midnight_and_includes_zero_days(self):
        session_1 = pd.Timestamp("2025-01-02")
        session_2 = pd.Timestamp("2025-01-03")
        intraday = pd.DataFrame(
            {
                "DateTime": pd.to_datetime(
                    [
                        "2025-01-01 18:00",
                        "2025-01-02 16:00",
                        "2025-01-02 18:00",
                        "2025-01-03 16:00",
                    ]
                ),
                "Session": [session_1, session_1, session_2, session_2],
            }
        )
        equity = pd.DataFrame(
            {
                "DateTime": intraday["DateTime"],
                "Equity": [100.0, 110.0, 110.0, 99.0],
            }
        )
        result = BacktestResult("test", {}, equity, pd.DataFrame(), {})
        returns = session_returns(
            result,
            intraday,
            [session_1, session_2, pd.Timestamp("2025-01-04")],
            100.0,
        )
        np.testing.assert_allclose(returns.to_numpy(), [0.1, -0.1, 0.0])
        self.assertEqual(
            returns.index.tolist(),
            [session_1.date(), session_2.date(), pd.Timestamp("2025-01-04").date()],
        )


if __name__ == "__main__":
    unittest.main()
