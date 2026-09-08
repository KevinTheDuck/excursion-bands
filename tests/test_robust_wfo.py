import json
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from test_robust_execution import _config, _market

from excursion_bands.backtesting.checkpoints import (
    CheckpointStore,
    backtest_fingerprint,
)
from excursion_bands.backtesting.robust_wfo import (
    _ResearchRun,
    robust_folds,
    run_robust_wfo,
)
from excursion_bands.backtesting.specification import RobustWfoConfig, VariantConfig
from excursion_bands.backtesting.wfo import run_wfo


def settings(directory, protocol="robust"):
    config = _config()
    robust = RobustWfoConfig(
        search_months=2,
        search_blocks=2,
        calibration_months=1,
        gate_months=1,
        shortlist_size=1,
        min_total_trades=0,
        min_block_trades=0,
        min_regime_sessions=2,
        min_regime_trades=0,
        gate_min_trades=0,
        bootstrap_simulations=10,
        bootstrap_block_sessions=2,
        workers=1,
        include_legacy=False,
        search_space={
            "strategy.donchian.lookback_bars": {
                "type": "int",
                "low": 3,
                "high": 5,
                "step": 1,
            }
        },
    )
    return replace(
        config,
        variants=(VariantConfig(label="raw", side_mode="long_only"),),
        backtest=replace(
            config.backtest,
            start_date=pd.Timestamp("2025-05-02").date(),
            end_date=pd.Timestamp("2025-06-01").date(),
        ),
        reports=replace(
            config.reports,
            save_charts=False,
            wfo_research=replace(config.reports.wfo_research, enabled=False),
        ),
        wfo=replace(
            config.wfo,
            protocol=protocol,
            checkpoint_dir=str(directory),
            robust=robust,
            n_trials=2,
            test_months=1,
            step_months=1,
            train_months=1,
            optimizer="optuna",
            reoptimization_mode="always",
            parameter_grid={"strategy.donchian.lookback_bars": [3, 4, 5]},
        ),
    )


class RobustIntegrationTests(TestCase):
    def test_legacy_degradation_resume_after_selection(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            cfg = settings(root / "resumed", "legacy")
            cfg = replace(
                cfg,
                backtest=replace(
                    cfg.backtest, start_date=pd.Timestamp("2025-04-02").date()
                ),
                wfo=replace(cfg.wfo, reoptimization_mode="degradation"),
            )
            from excursion_bands.backtesting.wfo import _LegacyCheckpoint

            original = _LegacyCheckpoint.save

            def interrupt(checkpoint, **kwargs):
                original(checkpoint, **kwargs)
                if (
                    checkpoint.state.get("selected_folds")
                    and not checkpoint.committed_folds
                ):
                    raise KeyboardInterrupt()

            with (
                patch.object(_LegacyCheckpoint, "save", interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_wfo(intraday, bands, cfg)
            run_wfo(intraday, bands, replace(cfg, wfo=replace(cfg.wfo, resume=True)))
            run_wfo(
                intraday,
                bands,
                replace(cfg, wfo=replace(cfg.wfo, checkpoint_dir=str(root / "full"))),
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(root / "resumed" / "wfo_test_folds.csv"),
                pd.read_csv(root / "full" / "wfo_test_folds.csv"),
            )

    def test_boundaries_and_future_data(self):
        intraday, _ = _market(160)
        cfg = settings("unused")
        fold = robust_folds(intraday, cfg)[0]
        self.assertLess(max(fold["search"]), min(fold["calibration"]))
        self.assertLess(max(fold["calibration"]), min(fold["gate"]))
        self.assertLess(max(fold["gate"]), min(fold["test"]))

    def test_interrupt_resume_is_identical_and_preserves_capital(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            cfg = settings(root / "resumed")
            original = _ResearchRun.commit
            interrupted = False

            def interrupt(run):
                nonlocal interrupted
                original(run)
                if (
                    run.state.get("current", {}).get("oos_index", 0) == 2
                    and not interrupted
                ):
                    interrupted = True
                    raise KeyboardInterrupt()

            with (
                patch.object(_ResearchRun, "commit", interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_robust_wfo(intraday, bands, cfg)
            self.assertTrue(interrupted)
            run_robust_wfo(
                intraday, bands, replace(cfg, wfo=replace(cfg.wfo, resume=True))
            )
            run_robust_wfo(intraday, bands, settings(root / "full"))
            for filename in (
                "session_decisions.csv",
                "wfo_oos_metrics.csv",
                "wfo_train_trials.csv",
            ):
                pd.testing.assert_frame_equal(
                    pd.read_csv(root / "resumed" / filename),
                    pd.read_csv(root / "full" / filename),
                )
            decisions = pd.read_csv(root / "full" / "session_decisions.csv")
            for _, group in decisions.groupby("variant"):
                self.assertEqual(
                    group.starting_cash.iloc[1:].tolist(),
                    group.ending_cash.iloc[:-1].tolist(),
                )

    def test_timeout_checkpoint_and_resume_flags_do_not_change_identity(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            cfg = settings(temp)
            paused = replace(cfg, wfo=replace(cfg.wfo, max_run_seconds=0.000001))
            run_robust_wfo(intraday, bands, paused)
            resumed = replace(cfg, wfo=replace(cfg.wfo, resume=True))
            self.assertEqual(
                backtest_fingerprint(paused, intraday, bands),
                backtest_fingerprint(resumed, intraday, bands),
            )
            run_robust_wfo(intraday, bands, resumed)
            normalized = intraday.copy()
            normalized["Session"] = pd.to_datetime(normalized.Session).dt.date
            normalized_bands = bands.copy()
            normalized_bands["Session"] = pd.to_datetime(
                normalized_bands.Session
            ).dt.date
            with CheckpointStore(
                Path(temp),
                backtest_fingerprint(cfg, normalized, normalized_bands),
                resume=True,
            ) as store:
                self.assertTrue(store.completed)

    def test_legacy_resume_retains_optuna_trials(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            cfg = settings(Path(temp) / "legacy", "legacy")
            cfg = replace(
                cfg,
                backtest=replace(
                    cfg.backtest, start_date=pd.Timestamp("2025-04-02").date()
                ),
            )
            from excursion_bands.backtesting import wfo

            original = wfo._checkpoint_search_trial
            calls = 0

            def interrupt(*args, **kwargs):
                nonlocal calls
                original(*args, **kwargs)
                calls += 1
                if calls == 1:
                    raise KeyboardInterrupt()

            with (
                patch.object(wfo, "_checkpoint_search_trial", interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_wfo(intraday, bands, cfg)
            run_wfo(intraday, bands, replace(cfg, wfo=replace(cfg.wfo, resume=True)))
            reference = replace(
                cfg, wfo=replace(cfg.wfo, checkpoint_dir=str(Path(temp) / "reference"))
            )
            run_wfo(intraday, bands, reference)
            for filename in ("wfo_oos_metrics.csv", "wfo_test_folds.csv"):
                pd.testing.assert_frame_equal(
                    pd.read_csv(Path(temp) / "legacy" / filename),
                    pd.read_csv(Path(temp) / "reference" / filename),
                )

    def test_search_interruption_replays_same_sampler_sequence(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            cfg = settings(root / "resumed")
            original = _ResearchRun.commit

            def interrupt(run):
                original(run)
                if run.state.get("current", {}).get("trials") == 1:
                    raise KeyboardInterrupt()

            with (
                patch.object(_ResearchRun, "commit", interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_robust_wfo(intraday, bands, cfg)
            run_robust_wfo(
                intraday, bands, replace(cfg, wfo=replace(cfg.wfo, resume=True))
            )
            run_robust_wfo(intraday, bands, settings(root / "full"))
            pd.testing.assert_frame_equal(
                pd.read_csv(root / "resumed" / "wfo_train_trials.csv"),
                pd.read_csv(root / "full" / "wfo_train_trials.csv"),
            )

    def test_raw_bands_ablation_and_legacy_comparison(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            cfg = settings(temp)
            cfg = replace(
                cfg,
                variants=(
                    *cfg.variants,
                    VariantConfig(
                        label="bands", use_band_filter=True, side_mode="long_only"
                    ),
                ),
                wfo=replace(
                    cfg.wfo,
                    robust=replace(cfg.wfo.robust, include_legacy=True, workers=2),
                ),
            )
            run_robust_wfo(intraday, bands, cfg)
            self.assertTrue((Path(temp) / "wfo_report.md").exists())
            self.assertTrue(
                (Path(temp) / "legacy_comparison" / "wfo_report.md").exists()
            )
            decisions = pd.read_csv(Path(temp) / "session_decisions.csv")
            for kind in ("general", "regime"):
                band = decisions[decisions.variant == "bands_" + kind]
                ablation = decisions[decisions.variant == "bands_" + kind + "_no_bands"]
                self.assertEqual(
                    band.parameter_id.tolist(), ablation.parameter_id.tolist()
                )

    def test_outer_test_prices_cannot_change_selection(self):
        intraday, bands = _market(160)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            cfg = settings(root / "before")
            run_robust_wfo(intraday, bands, cfg)
            changed = intraday.copy()
            mask = pd.to_datetime(changed.Session).dt.date >= cfg.backtest.start_date
            changed.loc[mask, ["Open", "High", "Low", "Close"]] *= 1.5
            run_robust_wfo(changed, bands, settings(root / "after"))
            before = json.loads(
                (root / "before" / "parameter_libraries.json").read_text()
            )
            after = json.loads(
                (root / "after" / "parameter_libraries.json").read_text()
            )
            self.assertEqual(before[0]["plan"], after[0]["plan"])

    def test_legacy_committed_fold_is_not_repeated(self):
        intraday, bands = _market(210)
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            cfg = settings(root / "resumed", "legacy")
            cfg = replace(
                cfg,
                backtest=replace(
                    cfg.backtest,
                    start_date=pd.Timestamp("2025-04-02").date(),
                    end_date=pd.Timestamp("2025-07-01").date(),
                ),
            )
            from excursion_bands.backtesting.wfo import _LegacyCheckpoint

            original = _LegacyCheckpoint.save

            def interrupt(checkpoint, **kwargs):
                original(checkpoint, **kwargs)
                if len(checkpoint.committed_folds) == 1:
                    raise KeyboardInterrupt()

            with (
                patch.object(_LegacyCheckpoint, "save", interrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                run_wfo(intraday, bands, cfg)
            run_wfo(intraday, bands, replace(cfg, wfo=replace(cfg.wfo, resume=True)))
            run_wfo(
                intraday,
                bands,
                replace(cfg, wfo=replace(cfg.wfo, checkpoint_dir=str(root / "full"))),
            )
            for filename in ("wfo_oos_metrics.csv", "wfo_test_folds.csv"):
                pd.testing.assert_frame_equal(
                    pd.read_csv(root / "resumed" / filename),
                    pd.read_csv(root / "full" / filename),
                )
