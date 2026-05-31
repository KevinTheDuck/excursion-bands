"""
Donchian channel breakout strategy variants.
"""

from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd

from excursion_bands.backtesting.engine import run_backtest
from excursion_bands.backtesting.models import BacktestResult, Position, Signal
from excursion_bands.backtesting.specification import BacktestConfig, VariantConfig
from excursion_bands.backtesting.strategies.orb import _force_exit_timestamp, _rma


def _parse_time(value: str) -> time:
    return time.fromisoformat(value)


def prepare_donchian_bars(
    bars: pd.DataFrame, config: BacktestConfig, bands: pd.DataFrame | None = None
) -> pd.DataFrame:
    if config.strategy is None or config.strategy.donchian is None:
        raise ValueError("Donchian strategy requires strategy.donchian config")

    strategy = config.strategy
    df = bars.sort_values("DateTime").reset_index(drop=True).copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["BarTime"] = df["DateTime"].dt.time
    if "Session" not in df.columns:
        df["Session"] = df["DateTime"].dt.date

    force_exit = _parse_time(strategy.force_exit_time)
    session_to_force_exit = {
        session: _force_exit_timestamp(session, force_exit, df)
        for session in df["Session"].dropna().unique()
    }
    df["ForceExitTime"] = df["Session"].map(session_to_force_exit)

    lookback = strategy.donchian.lookback_bars
    df["Donchian_High"] = df["High"].rolling(lookback, min_periods=lookback).max().shift(1)
    df["Donchian_Low"] = df["Low"].rolling(lookback, min_periods=lookback).min().shift(1)
    df["Donchian_Mid"] = (df["Donchian_High"] + df["Donchian_Low"]) / 2

    daily = (
        df.groupby("Session", sort=True)
        .agg(Session_High=("High", "max"), Session_Low=("Low", "min"))
        .reset_index()
    )
    daily["Daily_Range"] = daily["Session_High"] - daily["Session_Low"]
    daily["ATR_Session"] = daily["Daily_Range"].rolling(
        strategy.atr.lookback_sessions, min_periods=strategy.atr.lookback_sessions
    ).mean().shift(1)

    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["_PV"] = typical * df["Volume"]
    df["VWAP"] = df.groupby("Session")["_PV"].cumsum() / df.groupby("Session")["Volume"].cumsum()

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR_Stop"] = _rma(true_range, strategy.atr_stop.length) * strategy.atr_stop.multiplier
    df = df.merge(daily[["Session", "ATR_Session"]], on="Session", how="left")

    if bands is not None:
        band_cols = [
            "Session",
            "Band_AE_Pos_Upper",
            "Band_FE_Pos_Lower",
            "Band_AE_Neg_Lower",
            "Band_FE_Neg_Upper",
        ]
        missing = set(band_cols) - set(bands.columns)
        if missing:
            raise ValueError(f"Bands data missing required columns: {sorted(missing)}")
        df = df.merge(bands[band_cols].copy(), on="Session", how="left")
    df = df.drop(columns=["_PV"])

    df["EntryAllowed"] = (
        df["Donchian_High"].notna()
        & df["Donchian_Low"].notna()
        & df["ForceExitTime"].notna()
    )
    df["OR_High"] = df["Donchian_High"]
    df["OR_Low"] = df["Donchian_Low"]
    df["OR_Mid"] = df["Donchian_Mid"]
    df["EntryIndexAfterOR"] = 1.0
    return df


def build_donchian_signal(
    bar: pd.Series, config: BacktestConfig, variant: VariantConfig
) -> Signal | None:
    strategy = config.strategy
    if strategy is None or not bool(bar.EntryAllowed):
        return None

    if variant.use_atr_buffer and pd.isna(bar.ATR_Session):
        return None

    buffer_points = strategy.atr.buffer_mult * float(bar.ATR_Session) if variant.use_atr_buffer else 0.0
    long_breakout = float(bar.Close) > float(bar.Donchian_High) + buffer_points
    short_breakout = float(bar.Close) < float(bar.Donchian_Low) - buffer_points

    if variant.use_vwap_filter:
        long_breakout = long_breakout and float(bar.Close) > float(bar.VWAP)
        short_breakout = short_breakout and float(bar.Close) < float(bar.VWAP)

    if variant.use_band_filter:
        required = [
            bar.Band_AE_Pos_Upper,
            bar.Band_FE_Pos_Lower,
            bar.Band_AE_Neg_Lower,
            bar.Band_FE_Neg_Upper,
        ]
        if any(pd.isna(value) for value in required):
            return None
        price = float(bar.Close)
        long_breakout = long_breakout and (
            float(bar.Band_AE_Pos_Upper) < price < float(bar.Band_FE_Pos_Lower)
        )
        short_breakout = short_breakout and (
            float(bar.Band_FE_Neg_Upper) < price < float(bar.Band_AE_Neg_Lower)
        )

    side_mode = variant.side_mode.lower().strip()
    if side_mode in {"long", "long_only"}:
        short_breakout = False
    elif side_mode in {"short", "short_only"}:
        long_breakout = False
    elif side_mode != "both":
        raise ValueError(f"Unsupported side_mode: {variant.side_mode}")

    if not long_breakout and not short_breakout:
        return None

    side = "long" if long_breakout else "short"
    stop_mode = "atr" if strategy.atr_stop.enabled else strategy.stop.mode
    if stop_mode == "donchian_boundary":
        stop_loss = float(bar.Donchian_Low) if side == "long" else float(bar.Donchian_High)
    elif stop_mode == "atr":
        if pd.isna(bar.ATR_Stop):
            return None
        stop_loss = float(bar.Low - bar.ATR_Stop) if side == "long" else float(bar.High + bar.ATR_Stop)
    else:
        raise ValueError(f"Unsupported Donchian stop mode: {stop_mode}")

    entry_reference = float(bar.Close)
    risk = abs(entry_reference - stop_loss)
    if risk <= 0:
        return None
    take_profit = (
        entry_reference + strategy.take_profit.rr * risk
        if side == "long"
        else entry_reference - strategy.take_profit.rr * risk
    )
    break_even_trigger = None
    break_even_stop = None
    if strategy.break_even.enabled:
        break_even_trigger = (
            entry_reference + strategy.break_even.trigger_rr * risk
            if side == "long"
            else entry_reference - strategy.break_even.trigger_rr * risk
        )
        break_even_stop = (
            entry_reference + strategy.break_even.offset_points
            if side == "long"
            else entry_reference - strategy.break_even.offset_points
        )

    return Signal(
        side=side,
        stop_loss=stop_loss,
        take_profit=float(take_profit),
        force_exit_time=pd.to_datetime(bar.ForceExitTime).to_pydatetime(),
        break_even_trigger=None if break_even_trigger is None else float(break_even_trigger),
        break_even_stop=None if break_even_stop is None else float(break_even_stop),
    )


def run_donchian_variant(
    bars: pd.DataFrame,
    config: BacktestConfig,
    variant: VariantConfig,
    allowed_signal_times: set[pd.Timestamp] | None = None,
) -> BacktestResult:
    traded_sessions: set[object] = set()

    def signal_func(
        _index: int, bar: pd.Series, _bars: pd.DataFrame, position: Position | None
    ) -> Signal | None:
        session = bar.Session
        if position is not None or session in traded_sessions:
            return None
        signal = build_donchian_signal(bar, config, variant)
        if signal is not None and allowed_signal_times is not None:
            if pd.Timestamp(bar.DateTime) not in allowed_signal_times:
                return None
            source = "band_hmm" if variant.use_band_filter else "raw_hmm"
            signal = replace(signal, source=source)
        elif signal is not None and allowed_signal_times is None:
            signal = replace(signal, source="band" if variant.use_band_filter else "raw")
        if signal is not None:
            traded_sessions.add(session)
        return signal

    return run_backtest(variant.label, bars, config, signal_func)


def default_donchian_variants() -> tuple[VariantConfig, ...]:
    return (
        VariantConfig(label="donchian_raw", side_mode="long_only"),
        VariantConfig(label="donchian_bands", side_mode="long_only", use_band_filter=True),
    )
