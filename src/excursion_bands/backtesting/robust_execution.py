"""Efficient execution helpers for robust Donchian walk-forward research."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import replace
from threading import Lock
from typing import Any

import pandas as pd

from excursion_bands.backtesting.metrics import calculate_metrics
from excursion_bands.backtesting.models import BacktestResult
from excursion_bands.backtesting.specification import BacktestConfig, VariantConfig
from excursion_bands.backtesting.strategies.donchian import (
    prepare_donchian_bars,
    run_donchian_variant,
)

_BAND_COLUMNS = (
    "Band_AE_Pos_Upper",
    "Band_FE_Pos_Lower",
    "Band_AE_Neg_Lower",
    "Band_FE_Neg_Upper",
)
_TRADE_COLUMNS = (
    "EntryTime",
    "ExitTime",
    "Side",
    "Units",
    "EntryPrice",
    "ExitPrice",
    "GrossPnL",
    "Commission",
    "NetPnL",
    "ReturnPct",
    "ExitReason",
    "Source",
    "Probability",
    "Session",
)


def _normal_session(value: object) -> object:
    if isinstance(value, (pd.Timestamp, str)):
        try:
            return pd.Timestamp(value).date()
        except (TypeError, ValueError):
            return value
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        try:
            return pd.Timestamp(value).date()
        except (TypeError, ValueError):
            return value
    return value


def _normal_sessions(sessions: Sequence[object]) -> list[object]:
    unique: list[object] = []
    seen: set[object] = set()
    for value in sessions:
        session = _normal_session(value)
        if session not in seen:
            seen.add(session)
            unique.append(session)
    return unique


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_TRADE_COLUMNS))


def _flat_metrics(cash: float) -> dict[str, float | int | str | None]:
    return {
        "initial_cash": float(cash),
        "final_equity": float(cash),
        "total_return_pct": 0.0,
        "cagr_pct": 0.0 if cash > 0 else None,
        "max_drawdown_pct": 0.0,
        "sharpe": None,
        "sortino": None,
        "trade_count": 0,
        "win_rate_pct": None,
        "profit_factor": None,
        "expectancy": None,
        "exposure_pct": 0.0,
    }


class ParameterEvaluator:
    """Evaluate Donchian parameters while retaining only reusable components.

    The static strategy columns are prepared once.  Donchian rolling arrays use
    a small LRU cache, band frames are session-sized, and fully prepared trial
    frames are transient so trial count cannot multiply full-history memory.
    """

    _rolling_cache_size = 8

    def __init__(
        self,
        intraday: pd.DataFrame,
        bands: pd.DataFrame,
        config: BacktestConfig,
        variant: VariantConfig,
    ) -> None:
        if config.strategy is None or config.strategy.name != "donchian":
            raise ValueError("ParameterEvaluator requires a Donchian strategy")
        if config.strategy.donchian is None:
            raise ValueError("ParameterEvaluator requires strategy.donchian")
        if variant.use_hmm_filter:
            raise ValueError("robust parameter evaluation excludes HMM variants")
        if intraday.empty:
            raise ValueError("intraday data cannot be empty")

        self.config = config
        self.variant = variant
        self._intraday = intraday.copy()
        self._intraday["DateTime"] = pd.to_datetime(
            self._intraday["DateTime"], errors="raise"
        )
        self._intraday = self._intraday.sort_values("DateTime").reset_index(drop=True)
        if "Session" not in self._intraday:
            self._intraday["Session"] = self._intraday["DateTime"].dt.date
        else:
            self._intraday["Session"] = pd.to_datetime(
                self._intraday["Session"], errors="raise"
            ).dt.date
        self._earliest_date = self._intraday["DateTime"].iloc[0].date()

        self._bands_source = bands.copy()
        if "Session" not in self._bands_source:
            raise ValueError("bands data must contain Session")
        self._bands_source["Session"] = pd.to_datetime(
            self._bands_source["Session"], errors="raise"
        ).dt.date

        prepared = prepare_donchian_bars(self._intraday, config, None)
        static_columns = [
            "DateTime",
            "Session",
            "Open",
            "High",
            "Low",
            "Close",
            "ForceExitTime",
            "ATR_Session",
            "VWAP",
        ]
        self._base = prepared[static_columns].copy()
        self._session_indices = self._base.groupby("Session", sort=False).indices
        default_multiplier = float(config.strategy.atr_stop.multiplier)
        if default_multiplier != 0.0:
            self._atr_stop_base = (
                prepared["ATR_Stop"].astype(float) / default_multiplier
            )
        else:
            from excursion_bands.backtesting.strategies.orb import _rma

            previous_close = self._base["Close"].shift(1)
            true_range = pd.concat(
                [
                    self._base["High"] - self._base["Low"],
                    (self._base["High"] - previous_close).abs(),
                    (self._base["Low"] - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            self._atr_stop_base = _rma(true_range, config.strategy.atr_stop.length)

        self._rolling_cache: OrderedDict[int, tuple[pd.Series, pd.Series]] = (
            OrderedDict()
        )
        self._band_cache: dict[int | None, pd.DataFrame] = {}
        self._cache_lock = Lock()

    def _donchian(self, lookback: int) -> tuple[pd.Series, pd.Series]:
        if lookback <= 0:
            raise ValueError("Donchian lookback must be positive")
        with self._cache_lock:
            cached = self._rolling_cache.get(lookback)
            if cached is not None:
                self._rolling_cache.move_to_end(lookback)
                return cached
        high = self._base["High"].rolling(lookback, min_periods=lookback).max().shift(1)
        low = self._base["Low"].rolling(lookback, min_periods=lookback).min().shift(1)
        with self._cache_lock:
            existing = self._rolling_cache.get(lookback)
            if existing is not None:
                self._rolling_cache.move_to_end(lookback)
                return existing
            self._rolling_cache[lookback] = (high, low)
            while len(self._rolling_cache) > self._rolling_cache_size:
                self._rolling_cache.popitem(last=False)
        return high, low

    @staticmethod
    def _band_columns(frame: pd.DataFrame) -> pd.DataFrame:
        required = {"Session", *_BAND_COLUMNS}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"bands data missing required columns: {sorted(missing)}")
        result = frame[["Session", *_BAND_COLUMNS]].copy()
        result["Session"] = pd.to_datetime(result["Session"], errors="raise").dt.date
        return result.drop_duplicates("Session", keep="last")

    def _bands(self, lookback: int | None, variant: VariantConfig) -> pd.DataFrame:
        with self._cache_lock:
            cached = self._band_cache.get(lookback)
            if cached is not None:
                return cached
        if lookback is None:
            calculated = self._band_columns(self._bands_source)
        else:
            # Keep band-generation behavior identical to the legacy WFO while
            # retaining only the four session-sized columns used by Donchian.
            from excursion_bands.backtesting.wfo import _build_band_cache

            built = _build_band_cache(
                self._bands_source,
                [{"features.bands.lookback_window": int(lookback)}],
                self.config,
                variant,
            )
            calculated = self._band_columns(built[int(lookback)])
        with self._cache_lock:
            existing = self._band_cache.setdefault(lookback, calculated)
        return existing

    def _prepare(
        self,
        params: dict[str, Any],
        variant: VariantConfig,
        sessions: Sequence[object] | None = None,
    ) -> pd.DataFrame:
        from excursion_bands.backtesting.wfo import _patch_config

        trial_config = _patch_config(self.config, params)
        strategy = trial_config.strategy
        if strategy is None or strategy.donchian is None:
            raise ValueError("patched configuration is missing Donchian strategy")
        high, low = self._donchian(strategy.donchian.lookback_bars)
        if sessions is None:
            frame = self._base.copy()
        else:
            indices = sorted(
                i
                for session in _normal_sessions(sessions)
                for i in self._session_indices.get(session, [])
            )
            frame = self._base.iloc[indices].copy()
        frame["Donchian_High"] = high.reindex(frame.index)
        frame["Donchian_Low"] = low.reindex(frame.index)
        frame["Donchian_Mid"] = (frame["Donchian_High"] + frame["Donchian_Low"]) / 2.0
        frame["ATR_Stop"] = (
            self._atr_stop_base.reindex(frame.index) * strategy.atr_stop.multiplier
        )

        if variant.use_band_filter:
            lookback_value = params.get("features.bands.lookback_window")
            lookback = None if lookback_value is None else int(lookback_value)
            band_data = self._bands(lookback, variant).set_index("Session")
            for column in _BAND_COLUMNS:
                frame[column] = frame["Session"].map(band_data[column])

        frame["EntryAllowed"] = (
            frame["Donchian_High"].notna()
            & frame["Donchian_Low"].notna()
            & frame["ForceExitTime"].notna()
        )
        frame["OR_High"] = frame["Donchian_High"]
        frame["OR_Low"] = frame["Donchian_Low"]
        frame["OR_Mid"] = frame["Donchian_Mid"]
        frame["EntryIndexAfterOR"] = 1.0
        return frame

    def _trial_config(
        self, params: dict[str, Any], cost_multiplier: float
    ) -> BacktestConfig:
        from excursion_bands.backtesting.wfo import _patch_config

        if not math.isfinite(cost_multiplier) or cost_multiplier <= 0.0:
            raise ValueError("cost_multiplier must be finite and positive")
        trial = _patch_config(self.config, params)
        return replace(
            trial,
            backtest=replace(
                trial.backtest,
                start_date=self._earliest_date,
                end_date=None,
            ),
            execution=replace(
                trial.execution,
                commission_per_contract_side=(
                    trial.execution.commission_per_contract_side * cost_multiplier
                ),
                slippage_ticks_per_side=(
                    trial.execution.slippage_ticks_per_side * cost_multiplier
                ),
            ),
        )

    def _selected_bars(
        self, prepared: pd.DataFrame, sessions: Sequence[object]
    ) -> pd.DataFrame:
        selected = set(_normal_sessions(sessions))
        return prepared[prepared["Session"].isin(selected)]

    def _flat_result(
        self,
        prepared: pd.DataFrame,
        sessions: Sequence[object],
        cash: float,
        variant: VariantConfig,
        cost_multiplier: float,
    ) -> BacktestResult:
        selected = self._selected_bars(prepared, sessions)
        equity = pd.DataFrame(
            {
                "DateTime": selected["DateTime"],
                "Cash": float(cash),
                "Equity": float(cash),
                "InPosition": 0,
            }
        ).reset_index(drop=True)
        metrics = (
            calculate_metrics(equity, _empty_trades(), cash)
            if cash > 0.0 and not equity.empty
            else _flat_metrics(cash)
        )
        return BacktestResult(
            name=variant.label,
            metrics=metrics,
            equity_curve=equity,
            trades=_empty_trades(),
            config_summary={
                "symbol": self.config.instrument.symbol,
                "starting_cash": float(cash),
                "cost_multiplier": float(cost_multiplier),
            },
        )

    def run(
        self,
        params: dict[str, Any],
        sessions: Sequence[object],
        cash: float | None = None,
        *,
        variant: VariantConfig | None = None,
        cost_multiplier: float = 1.0,
    ) -> BacktestResult:
        """Run selected sessions with full-history causal indicators."""
        selected_variant = self.variant if variant is None else variant
        if selected_variant.use_hmm_filter:
            raise ValueError("robust parameter evaluation excludes HMM variants")
        starting_cash = (
            self.config.backtest.initial_cash if cash is None else float(cash)
        )
        if not math.isfinite(starting_cash):
            raise ValueError("cash must be finite")
        trial_config = self._trial_config(params, cost_multiplier)
        prepared = self._prepare(params, selected_variant, sessions)
        selected = self._selected_bars(prepared, sessions)
        if starting_cash <= 0.0 or len(selected) < 2:
            return self._flat_result(
                prepared, sessions, starting_cash, selected_variant, cost_multiplier
            )

        result = run_donchian_variant(
            prepared,
            trial_config,
            selected_variant,
            starting_cash=starting_cash,
            session_filter=set(_normal_sessions(sessions)),
        )
        lookup = prepared[["DateTime", "Session"]].drop_duplicates(
            "DateTime", keep="last"
        )
        trades = result.trades.merge(
            lookup,
            how="left",
            left_on="EntryTime",
            right_on="DateTime",
            validate="many_to_one",
        ).drop(columns="DateTime")
        return replace(result, trades=trades)


def session_returns(
    result: BacktestResult,
    intraday: pd.DataFrame,
    sessions: Sequence[object],
    initial_cash: float,
) -> pd.Series:
    """Convert bar equity to session returns using the source session labels."""
    ordered_sessions = _normal_sessions(sessions)
    output = pd.Series(
        0.0,
        index=pd.Index(ordered_sessions, name="Session"),
        dtype=float,
        name="Return",
    )
    if not math.isfinite(float(initial_cash)):
        raise ValueError("initial_cash must be finite")
    if initial_cash <= 0.0 or result.equity_curve.empty or not ordered_sessions:
        return output
    required = {"DateTime", "Session"}
    missing = required.difference(intraday.columns)
    if missing:
        raise ValueError(f"intraday is missing required columns: {sorted(missing)}")

    lookup = intraday[["DateTime", "Session"]].copy()
    lookup["DateTime"] = pd.to_datetime(lookup["DateTime"], errors="raise")
    lookup["Session"] = lookup["Session"].map(_normal_session)
    lookup = lookup.drop_duplicates("DateTime", keep="last")
    equity = result.equity_curve[["DateTime", "Equity"]].copy()
    equity["DateTime"] = pd.to_datetime(equity["DateTime"], errors="raise")
    equity = equity.merge(lookup, on="DateTime", how="left", validate="many_to_one")
    equity = equity[equity["Session"].isin(ordered_sessions)].sort_values("DateTime")
    ending_equity = equity.groupby("Session", sort=False)["Equity"].last()

    previous = float(initial_cash)
    for session in ordered_sessions:
        if session not in ending_equity.index:
            continue
        current = float(ending_equity.loc[session])
        if previous == 0.0 or not math.isfinite(current):
            output.loc[session] = 0.0
        else:
            output.loc[session] = current / previous - 1.0
        previous = current
    return output
