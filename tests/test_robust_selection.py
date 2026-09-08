from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from excursion_bands.backtesting.robust_selection import (
    build_routing_schedule,
    calibrate_library,
    classify_regimes,
    fit_regime_thresholds,
    gate_router,
    numeric_neighbors,
    score_candidate,
    session_volatility,
)


class CandidateScoringTests(unittest.TestCase):
    def test_numeric_session_identifiers_remain_distinct(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.01], index=[1, 2, 3, 4])
        trades = pd.DataFrame({"Session": [1, 2, 3, 4]})
        result = score_candidate(
            returns,
            trades,
            [[1, 2], [3, 4]],
            min_total_trades=4,
            min_block_trades=2,
        )
        self.assertTrue(result["eligible"])

    def test_zero_return_sessions_are_included_in_sharpe(self):
        sessions = pd.date_range("2020-01-01", periods=8, freq="D")
        returns = pd.Series([0.01, 0.0, -0.005, 0.0] * 2, index=sessions)
        trades = pd.DataFrame({"Session": sessions[[0, 2, 4, 6]]})
        blocks = [sessions[:4], sessions[4:]]

        result = score_candidate(
            returns, trades, blocks, min_total_trades=4, min_block_trades=2
        )

        expected = returns.iloc[:4].mean() / returns.iloc[:4].std(ddof=0) * np.sqrt(252)
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["block_sharpes"][0], expected)
        self.assertEqual(result["total_trades"], 4)
        self.assertLessEqual(result["max_drawdown"], 0.0)

    def test_insufficient_block_trades_are_ineligible(self):
        sessions = pd.date_range("2020-01-01", periods=8, freq="D")
        returns = pd.Series([0.01, -0.01] * 4, index=sessions)
        trades = pd.DataFrame({"EntryTime": sessions[:3]})
        result = score_candidate(
            returns,
            trades,
            [sessions[:4], sessions[4:]],
            min_total_trades=3,
            min_block_trades=1,
        )
        self.assertFalse(result["eligible"])
        self.assertEqual(result["score"], float("-inf"))
        self.assertEqual(result["reason"], "block_1_insufficient_trades")

    def test_bankrupt_or_zero_variance_candidate_is_ineligible(self):
        sessions = pd.date_range("2020-01-01", periods=4, freq="D")
        trades = pd.DataFrame({"Session": sessions})
        bankrupt = score_candidate(
            pd.Series([0.1, -1.0, 0.1, -0.1], index=sessions),
            trades,
            [sessions],
            min_total_trades=1,
            min_block_trades=1,
        )
        constant = score_candidate(
            pd.Series(0.0, index=sessions),
            trades,
            [sessions],
            min_total_trades=1,
            min_block_trades=1,
        )
        self.assertEqual(bankrupt["reason"], "bankrupt_return")
        self.assertEqual(constant["reason"], "block_0_zero_variance")


class NeighborTests(unittest.TestCase):
    def test_numeric_neighbors_respect_steps_bounds_and_categories(self):
        params = {"lookback": 10, "atr": 1.0, "break_even": False}
        search_space = {
            "lookback": {"low": 10, "high": 20, "step": 5},
            "atr": {"low": 0.75, "high": 1.25, "step": 0.25},
            "break_even": {"choices": [False, True]},
        }
        self.assertEqual(
            numeric_neighbors(params, search_space),
            [
                {"lookback": 15, "atr": 1.0, "break_even": False},
                {"lookback": 10, "atr": 0.75, "break_even": False},
                {"lookback": 10, "atr": 1.25, "break_even": False},
            ],
        )


class VolatilityTests(unittest.TestCase):
    @staticmethod
    def _bars(closes: np.ndarray) -> pd.DataFrame:
        sessions = pd.date_range("2020-01-01", periods=len(closes), freq="D")
        rows = []
        for session, close in zip(sessions, closes, strict=True):
            rows.extend(
                [
                    {
                        "Session": session,
                        "DateTime": session + pd.Timedelta(hours=9),
                        "Close": close - 0.1,
                    },
                    {
                        "Session": session,
                        "DateTime": session + pd.Timedelta(hours=16),
                        "Close": close,
                    },
                ]
            )
        return pd.DataFrame(rows)

    def test_future_price_change_does_not_change_earlier_volatility(self):
        closes = 100.0 * np.exp(
            np.linspace(0.0, 0.3, 30) + np.sin(np.arange(30)) * 0.01
        )
        original = session_volatility(self._bars(closes))
        changed = closes.copy()
        changed[-1] *= 2.0
        revised = session_volatility(self._bars(changed))
        pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])
        # The current session's value is also fixed before that session closes.
        self.assertEqual(original.iloc[-1], revised.iloc[-1])

    def test_thresholds_and_classification_handle_unknowns(self):
        values = pd.Series([1.0, 2.0, 3.0, np.nan])
        thresholds = fit_regime_thresholds(values)
        labels = classify_regimes(values, thresholds)
        self.assertEqual(labels.tolist(), ["low", "normal", "high", "unknown"])
        self.assertIsNone(fit_regime_thresholds(pd.Series([np.nan])))


class CalibrationTests(unittest.TestCase):
    def test_sparse_regime_falls_back_and_shrinkage_selects_stable_candidate(self):
        sessions = pd.date_range("2021-01-01", periods=12, freq="D")
        regimes = pd.Series(["low"] * 5 + ["normal"] * 5 + ["high"] * 2, index=sessions)
        candidate_returns = {
            "general": pd.Series([0.002, -0.001] * 6, index=sessions),
            "stable": pd.Series(
                [0.004, -0.001, 0.003, -0.001, 0.004] + [0.001, -0.001] * 3 + [0.001],
                index=sessions,
            ),
        }
        candidate_trades = {
            candidate: pd.DataFrame({"Session": sessions})
            for candidate in candidate_returns
        }
        result = calibrate_library(
            candidate_returns,
            candidate_trades,
            regimes,
            "general",
            min_sessions=3,
            min_trades=3,
            shrinkage_sessions=10,
        )
        self.assertEqual(result["mapping"]["low"], "stable")
        self.assertEqual(result["mapping"]["high"], "general")
        self.assertEqual(
            result["diagnostics"]["high"]["reason"], "fallback_no_eligible_candidate"
        )

    def test_ties_are_resolved_by_candidate_id(self):
        sessions = pd.date_range("2021-01-01", periods=6, freq="D")
        returns = pd.Series([0.01, -0.005] * 3, index=sessions)
        regimes = pd.Series(["low"] * 6, index=sessions)
        result = calibrate_library(
            {"general": returns, "alpha": returns.copy()},
            {
                "general": pd.DataFrame({"Session": sessions}),
                "alpha": pd.DataFrame({"Session": sessions}),
            },
            regimes,
            "general",
            min_sessions=3,
            min_trades=3,
        )
        self.assertEqual(result["mapping"]["low"], "alpha")


class RoutingTests(unittest.TestCase):
    def test_confirmation_and_unknown_reset(self):
        regimes = pd.Series(["low", "low", "high", "high", "unknown", "high", "high"])
        schedule = build_routing_schedule(
            regimes, {"low": "slow", "high": "fast"}, "general", 2
        )
        self.assertEqual(
            schedule.tolist(),
            ["general", "slow", "slow", "fast", "general", "general", "fast"],
        )


class RouterGateTests(unittest.TestCase):
    def test_constant_positive_improvement_passes(self):
        sessions = pd.date_range("2022-01-01", periods=60, freq="D")
        general = pd.Series([0.001, -0.001] * 30, index=sessions)
        router = general + 0.001
        result = gate_router(
            router, general, 40, 40, simulations=100, block_length=10, seed=7
        )
        self.assertTrue(result["accepted"])
        self.assertGreater(result["ci_lower"], 0.0)

    def test_negative_improvement_fails(self):
        sessions = pd.date_range("2022-01-01", periods=60, freq="D")
        general = pd.Series([0.001, -0.001] * 30, index=sessions)
        result = gate_router(
            general - 0.001,
            general,
            40,
            40,
            simulations=100,
            block_length=10,
            seed=7,
        )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "bootstrap_improvement_not_positive")


if __name__ == "__main__":
    unittest.main()
