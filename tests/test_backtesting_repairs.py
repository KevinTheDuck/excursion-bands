from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import polars as pl

from excursion_bands.backtesting.engine import _exit_from_bar, run_backtest
from excursion_bands.backtesting.hmm import (
    FEATURE_COLUMNS,
    _simulate_candidate,
    train_hmm_filter,
)
from excursion_bands.backtesting.loader import load_backtest_config
from excursion_bands.backtesting.metrics import (
    calculate_metrics,
    calculate_yearly_metrics,
)
from excursion_bands.backtesting.models import Position, Signal
from excursion_bands.backtesting.monte_carlo import simulate_trade_bootstrap
from excursion_bands.backtesting.specification import VariantConfig, WfoConfig
from excursion_bands.backtesting.strategies.donchian import prepare_donchian_bars
from excursion_bands.backtesting.wfo import (
    _build_session_folds,
    _validate_continuous_folds,
    run_wfo,
)
from excursion_bands.data.cache import cache_fingerprint


def _config():
    with redirect_stdout(StringIO()):
        return load_backtest_config("configs/strategies/donchian_1/backtest_default.yaml")


def _position(**overrides):
    values = {
        "side": "long",
        "units": 1.0,
        "entry_time": pd.Timestamp("2025-01-01"),
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "take_profit": 120.0,
        "force_exit_time": None,
        "entry_commission": 0.0,
    }
    values.update(overrides)
    return Position(**values)


class AccountingTests(unittest.TestCase):
    def test_drawdown_includes_opening_equity(self):
        equity = pd.DataFrame(
            {
                "DateTime": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "Equity": [90_000.0, 95_000.0],
                "InPosition": [1, 0],
            }
        )
        trades = pd.DataFrame({"NetPnL": [-10_000.0], "ExitTime": equity.DateTime[:1]})
        metrics = calculate_metrics(equity, trades, 100_000.0)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], -10.0)

    def test_yearly_drawdown_uses_year_start_balance(self):
        equity = pd.DataFrame(
            {
                "DateTime": pd.to_datetime(["2025-12-31", "2026-01-01", "2026-01-02"]),
                "Equity": [95_000.0, 80_000.0, 100_000.0],
                "InPosition": [0, 1, 0],
            }
        )
        trades = pd.DataFrame({"NetPnL": [-15_000.0, 20_000.0], "ExitTime": equity.DateTime[1:]})
        yearly = calculate_yearly_metrics(equity, trades, 100_000.0)
        self.assertAlmostEqual(yearly.loc[yearly.Year == 2026, "MaxDrawdownPct"].iloc[0], -15.7894736842)

    def test_monte_carlo_drawdown_includes_opening_equity(self):
        _, summary = simulate_trade_bootstrap(
            pd.DataFrame({"NetPnL": [-10_000.0]}), 100_000.0, 1, 1, 42
        )
        self.assertAlmostEqual(summary["max_drawdown_pct"].iloc[0], -10.0)

    def test_engine_uses_supplied_starting_cash(self):
        config = _config()
        config = config.__class__(
            **{
                **config.__dict__,
                "backtest": config.backtest.__class__(
                    start_date=pd.Timestamp("2025-01-01").date(),
                    end_date=None,
                    initial_cash=100_000.0,
                ),
                "sizing": config.sizing.__class__(
                    mode="fixed_units", fixed_units=1.0, risk_pct=config.sizing.risk_pct
                ),
                "execution": config.execution.__class__(
                    commission_per_contract_side=0.0,
                    slippage_ticks_per_side=0.0,
                    fill_on="next_open",
                    same_bar_exit_priority="stop",
                ),
            }
        )
        bars = pd.DataFrame(
            {
                "DateTime": pd.date_range("2025-01-01", periods=3, freq="h"),
                "Session": [1, 1, 1],
                "Open": [100.0, 100.0, 110.0],
                "High": [100.0, 101.0, 111.0],
                "Low": [100.0, 99.0, 109.0],
                "Close": [100.0, 100.0, 110.0],
            }
        )
        result = run_backtest(
            "test",
            bars,
            config,
            lambda i, bar, _bars, _position: Signal(
                "long", stop_loss=90.0, session=1
            )
            if i == 0
            else None,
            starting_cash=200_000.0,
        )
        self.assertEqual(result.metrics["initial_cash"], 200_000.0)
        self.assertAlmostEqual(result.metrics["final_equity"], 200_010.0)


class ExecutionTests(unittest.TestCase):
    def test_opening_target_precedes_later_stop(self):
        bar = SimpleNamespace(DateTime=pd.Timestamp("2025-01-02"), Open=125.0,
                              High=126.0, Low=90.0, Close=95.0)
        self.assertEqual(_exit_from_bar(_position(), bar, "stop"), (120.0, "take_profit"))

    def test_hmm_rejects_entry_at_cutoff(self):
        bars = pd.DataFrame({"DateTime": pd.date_range("2025-01-01", periods=2, freq="h"),
                             "Session": [1, 1], "Open": [100.0, 100.0]})
        signal = Signal("long", stop_loss=90.0, force_exit_time=bars.DateTime.iloc[1], session=1)
        self.assertIsNone(_simulate_candidate(bars, 0, signal, _config()))

    def test_gap_through_stop_fills_at_open(self):
        bar = SimpleNamespace(
            DateTime=pd.Timestamp("2025-01-02"), Open=90.0, High=94.0, Low=89.0, Close=92.0
        )
        self.assertEqual(_exit_from_bar(_position(), bar, "stop"), (90.0, "stop_loss"))

    def test_position_without_stop_is_supported(self):
        bar = SimpleNamespace(
            DateTime=pd.Timestamp("2025-01-02"), Open=100.0, High=104.0, Low=98.0, Close=101.0
        )
        self.assertIsNone(_exit_from_bar(_position(stop_loss=None, take_profit=None), bar, "stop"))

    def test_break_even_activation_is_deferred(self):
        position = _position(break_even_trigger=105.0, break_even_stop=100.0)
        trigger_bar = SimpleNamespace(
            DateTime=pd.Timestamp("2025-01-02"), Open=100.0, High=106.0, Low=96.0, Close=104.0
        )
        self.assertIsNone(_exit_from_bar(position, trigger_bar, "stop"))
        self.assertTrue(position.break_even_moved)
        self.assertEqual(position.stop_loss, 100.0)

        exit_bar = SimpleNamespace(
            DateTime=pd.Timestamp("2025-01-03"), Open=101.0, High=102.0, Low=99.0, Close=100.0
        )
        self.assertEqual(_exit_from_bar(position, exit_bar, "stop"), (100.0, "stop_loss"))

    def test_force_exit_uses_cutoff_open(self):
        config = _config()
        config = config.__class__(
            **{
                **config.__dict__,
                "backtest": config.backtest.__class__(
                    start_date=pd.Timestamp("2025-01-01").date(),
                    end_date=None,
                    initial_cash=100_000.0,
                ),
                "sizing": config.sizing.__class__(
                    mode="fixed_units", fixed_units=1.0, risk_pct=config.sizing.risk_pct
                ),
                "execution": config.execution.__class__(
                    commission_per_contract_side=0.0,
                    slippage_ticks_per_side=0.0,
                    fill_on="next_open",
                    same_bar_exit_priority="stop",
                ),
            }
        )
        bars = pd.DataFrame(
            {
                "DateTime": pd.date_range("2025-01-01", periods=4, freq="h"),
                "Session": [1, 1, 1, 1],
                "Open": [100.0, 100.0, 100.0, 90.0],
                "High": [100.0, 101.0, 101.0, 110.0],
                "Low": [100.0, 99.0, 99.0, 89.0],
                "Close": [100.0, 100.0, 100.0, 110.0],
            }
        )
        force_exit = pd.Timestamp("2025-01-01 03:00")
        result = run_backtest(
            "test",
            bars,
            config,
            lambda i, _bar, _bars, _position: Signal(
                "long", stop_loss=80.0, force_exit_time=force_exit, session=1
            )
            if i == 0
            else None,
        )
        self.assertAlmostEqual(result.trades["ExitPrice"].iloc[0], 90.0)
        self.assertEqual(result.trades["ExitReason"].iloc[0], "force_exit")


class WfoTests(unittest.TestCase):
    def test_hmm_does_not_fit_on_validation_when_training_is_small(self):
        config = _config()
        config = replace(config, hmm=replace(config.hmm, min_train_samples=8,
                         min_validation_trades=3, validation_fraction=.3))
        times = pd.date_range("2025-01-01", periods=10, freq="D")
        candidates = pd.DataFrame({"SignalTime": times, "Session": times.date,
                                   "Side": "long", "net_r": 1.0,
                                   **{column: 0.0 for column in FEATURE_COLUMNS}})
        with patch("excursion_bands.backtesting.hmm._add_hmm_features", side_effect=lambda data, **kw: data), \
             patch("excursion_bands.backtesting.hmm.build_candidate_dataset", side_effect=[candidates, candidates.iloc[:2]]), \
             patch("excursion_bands.backtesting.hmm._fit_hmm") as fit:
            result = train_hmm_filter(pd.DataFrame(), pd.DataFrame(), config, VariantConfig("test"))
        fit.assert_not_called()
        self.assertEqual(result.diagnostics["status"], "insufficient_fit_samples")
        self.assertTrue(result.predictions.Allowed.all())

    def test_real_engine_capital_and_stitched_accounting(self):
        config = _config()
        dates = pd.date_range("2025-01-01", periods=8, freq="D")
        bars = pd.DataFrame({
            "DateTime": [d + pd.Timedelta(hours=h) for d in dates for h in (9, 10, 11)],
            "Session": [d.date() for d in dates for _ in range(3)],
            "Open": [100.0, 100.0, 110.0] * 8,
            "High": [101.0, 101.0, 111.0] * 8,
            "Low": [99.0, 99.0, 109.0] * 8,
            "Close": [100.0, 100.0, 110.0] * 8,
        })
        def prepare(data, _config, _bands):
            return data
        def runner(data, cfg, variant, _allowed=None, **kwargs):
            def signal(_i, bar, _bars, position):
                if bar.DateTime.hour == 9 and position is None:
                    return Signal("long", stop_loss=90.0,
                                  force_exit_time=bar.DateTime + pd.Timedelta(hours=2))
                return None
            return run_backtest(variant.label, data, cfg, signal, **kwargs)
        with TemporaryDirectory() as directory:
            config = replace(config,
                backtest=replace(config.backtest, start_date=dates[0].date()),
                sizing=replace(config.sizing, mode="risk_current_equity", risk_pct=.01),
                execution=replace(config.execution, commission_per_contract_side=0, slippage_ticks_per_side=0),
                variants=(VariantConfig(label="test"),),
                wfo=replace(config.wfo, mode="sessions", train_sessions=2, test_sessions=2,
                            step_sessions=2, optimizer="grid", parameter_grid={},
                            objective="total_return_pct", reoptimization_mode="always"),
                reports=replace(config.reports, output_dir=directory, save_charts=False,
                                wfo_research=replace(config.reports.wfo_research, enabled=False)))
            with patch("excursion_bands.backtesting.wfo._strategy_functions", return_value=(prepare, runner)), redirect_stdout(StringIO()):
                output = run_wfo(bars, pd.DataFrame({"Session": dates}), config)
            folds = pd.read_csv(output / "wfo_test_folds.csv")
            metrics = pd.read_csv(output / "wfo_oos_metrics.csv").iloc[0]
            trades = pd.read_csv(output / "wfo_oos_trades_test.csv")
            self.assertEqual(len(folds), 3)
            self.assertAlmostEqual(folds.initial_cash.iloc[1], folds.final_equity.iloc[0])
            self.assertAlmostEqual(metrics.final_equity, 100000 * 1.01**6)
            self.assertAlmostEqual(metrics.final_equity - 100000, trades.NetPnL.sum())
            self.assertAlmostEqual(metrics.total_return_pct, (1.01**6 - 1) * 100)
            self.assertEqual(list(Path(output).glob("*.png")), [])

    def test_price_correction_invalidates_cache(self):
        frame = pl.DataFrame({"Close": [100.0, 101.0]})
        changed = pl.DataFrame({"Close": [100.0, 102.0]})
        self.assertNotEqual(cache_fingerprint(frames=[frame]), cache_fingerprint(frames=[changed]))

    def test_overlapping_oos_folds_are_rejected(self):
        folds = _build_session_folds([1, 2, 3, 4], 1, 2, 1)
        config = WfoConfig(
            enabled=True,
            max_parameter_combinations=10,
            max_workers=1,
            mode="sessions",
            train_sessions=1,
            test_sessions=2,
            step_sessions=1,
        )
        with self.assertRaises(ValueError):
            _validate_continuous_folds(folds, config)

    def test_session_normalisation_allows_band_merge(self):
        config = _config()
        config = config.__class__(
            **{
                **config.__dict__,
                "strategy": config.strategy.__class__(
                    **{
                        **config.strategy.__dict__,
                        "donchian": config.strategy.donchian.__class__(lookback_bars=2),
                    }
                ),
            }
        )
        bars = pd.DataFrame(
            {
                "DateTime": pd.date_range("2025-01-01", periods=4, freq="h"),
                "Session": pd.to_datetime(["2025-01-01"] * 4),
                "Open": [100.0] * 4,
                "High": [101.0] * 4,
                "Low": [99.0] * 4,
                "Close": [100.0] * 4,
                "Volume": [1] * 4,
            }
        )
        bands = pd.DataFrame(
            {
                "Session": [pd.Timestamp("2025-01-01")],
                "Band_AE_Pos_Upper": [90.0],
                "Band_FE_Pos_Lower": [110.0],
                "Band_AE_Neg_Lower": [90.0],
                "Band_FE_Neg_Upper": [110.0],
            }
        )
        prepared = prepare_donchian_bars(bars, config, bands)
        self.assertEqual(prepared["Session"].iloc[0], pd.Timestamp("2025-01-01").date())


if __name__ == "__main__":
    unittest.main()
