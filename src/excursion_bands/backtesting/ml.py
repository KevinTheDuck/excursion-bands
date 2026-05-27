"""
ML meta-label filter for strategy candidates.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from excursion_bands.backtesting.costs import (
    apply_entry_slippage,
    apply_exit_slippage,
    commission_for_units,
    slippage_points,
)
from excursion_bands.backtesting.models import Signal
from excursion_bands.backtesting.specification import BacktestConfig, VariantConfig
from excursion_bands.backtesting.strategies.donchian import build_donchian_signal
from excursion_bands.backtesting.strategies.orb import build_orb_signal


FEATURE_COLUMNS = [
    "side_long",
    "side_short",
    "entry_index_after_or",
    "or_width",
    "or_width_atr_ratio",
    "breakout_distance",
    "breakout_distance_atr_ratio",
    "close_position_in_or_range",
    "close_vs_vwap",
    "close_vs_vwap_atr_ratio",
    "atr_session",
    "atr_stop",
    "bar_range",
    "bar_body",
    "bar_body_to_range",
    "rolling_return_3",
    "rolling_return_6",
    "rolling_volatility_12",
    "day_of_week",
    "month",
    "band_width_side",
    "price_in_active_band",
    "dist_to_active_ae",
    "dist_to_active_fe",
    "dist_to_active_ae_atr",
    "dist_to_active_fe_atr",
    "breakout_x_band_position",
    "or_width_x_band_width",
    "vwap_x_band_position",
    "atr_stop_x_band_width",
]


@dataclass(frozen=True)
class MLFilterResult:
    allowed_signal_times: set[pd.Timestamp]
    train_dataset: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    diagnostics: dict[str, float | int | str]


@dataclass(frozen=True)
class ExpandingMLResult:
    allowed_signal_times: set[pd.Timestamp]
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    diagnostics: dict[str, float | int | str]


def train_ml_filter(
    train_prepared: pd.DataFrame,
    test_prepared: pd.DataFrame,
    config: BacktestConfig,
    variant: VariantConfig,
) -> MLFilterResult:
    if config.ml is None or not config.ml.enabled:
        return MLFilterResult(set(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    train_dataset = build_candidate_dataset(train_prepared, config, variant)
    test_dataset = build_candidate_dataset(test_prepared, config, variant)

    fallback_allowed = _fallback_allowed(test_dataset, config.ml.fallback)
    if len(train_dataset) < config.ml.min_train_samples or test_dataset.empty:
        return MLFilterResult(
            fallback_allowed,
            train_dataset,
            _fallback_predictions(test_dataset, "insufficient_samples"),
            pd.DataFrame(),
            {"status": "insufficient_samples", "train_samples": len(train_dataset)},
        )

    if train_dataset["label"].nunique() < 2:
        return MLFilterResult(
            fallback_allowed,
            train_dataset,
            _fallback_predictions(test_dataset, "single_class_train"),
            pd.DataFrame(),
            {"status": "single_class_train", "train_samples": len(train_dataset)},
        )

    x_train = _feature_matrix(train_dataset)
    y_train = train_dataset["label"].astype(int)
    x_test = _feature_matrix(test_dataset)

    xgb = config.ml.xgboost
    model = XGBClassifier(
        n_estimators=xgb.n_estimators,
        max_depth=xgb.max_depth,
        learning_rate=xgb.learning_rate,
        subsample=xgb.subsample,
        colsample_bytree=xgb.colsample_bytree,
        random_state=xgb.random_state,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = test_dataset[["SignalTime", "Session", "Side", "label", "net_r"]].copy()
    predictions["Probability"] = probabilities
    predictions["Allowed"] = predictions["Probability"] >= config.ml.probability_threshold
    allowed = set(pd.to_datetime(predictions.loc[predictions["Allowed"], "SignalTime"]))

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    diagnostics = {
        "status": "trained",
        "train_samples": int(len(train_dataset)),
        "test_candidates": int(len(test_dataset)),
        "allowed_trades": int(predictions["Allowed"].sum()),
        "rejected_trades": int((~predictions["Allowed"]).sum()),
        "threshold": float(config.ml.probability_threshold),
    }
    return MLFilterResult(allowed, train_dataset, predictions, importance, diagnostics)


def expanding_ml_filter(
    prepared: pd.DataFrame, config: BacktestConfig, variant: VariantConfig
) -> ExpandingMLResult:
    if config.ml is None or not config.ml.enabled:
        return ExpandingMLResult(set(), pd.DataFrame(), pd.DataFrame(), {"status": "disabled"})

    start_date = config.ml.warmup_start_date or config.backtest.start_date
    filtered = prepared[pd.to_datetime(prepared["DateTime"]).dt.date >= start_date]
    if config.backtest.end_date is not None:
        filtered = filtered[pd.to_datetime(filtered["DateTime"]).dt.date <= config.backtest.end_date]
    dataset = build_candidate_dataset(filtered, config, variant)
    if dataset.empty:
        return ExpandingMLResult(set(), pd.DataFrame(), pd.DataFrame(), {"status": "no_candidates"})

    dataset = dataset.sort_values("SignalTime").reset_index(drop=True)
    allowed: set[pd.Timestamp] = set()
    rows = []
    model: XGBClassifier | None = None
    feature_importance_rows = []
    last_fit_index: int | None = None

    for index, row in dataset.iterrows():
        signal_time = pd.Timestamp(row.SignalTime)
        train = dataset.iloc[:index]
        if config.ml.train_lookback_months is not None:
            cutoff = signal_time - pd.DateOffset(months=config.ml.train_lookback_months)
            train = train[pd.to_datetime(train["SignalTime"]) >= cutoff]

        should_refit = _should_refit_ml(model, last_fit_index, index, config.ml.refit_frequency_sessions, train)
        if should_refit:
            model = _fit_xgb(train, config)
            last_fit_index = index if model is not None else last_fit_index
            if model is not None:
                feature_importance_rows.append(
                    pd.DataFrame(
                        {
                            "SignalTime": [signal_time] * len(FEATURE_COLUMNS),
                            "feature": FEATURE_COLUMNS,
                            "importance": model.feature_importances_,
                        }
                    )
                )

        probability = np.nan
        reason = "trained"
        is_allowed = False
        if model is None:
            is_allowed = config.ml.fallback == "allow_all"
            reason = "fallback"
        else:
            x = _feature_matrix(pd.DataFrame([row]))
            probability = float(model.predict_proba(x)[:, 1][0])
            is_allowed = probability >= config.ml.probability_threshold

        is_trade_window = signal_time.date() >= config.backtest.start_date
        if is_allowed and is_trade_window:
            allowed.add(signal_time)
        rows.append(
            {
                "SignalTime": row.SignalTime,
                "Session": row.Session,
                "Side": row.Side,
                "label": int(row.label),
                "net_r": float(row.net_r),
                "Probability": probability,
                "Allowed": bool(is_allowed and is_trade_window),
                "Reason": reason,
                "TrainSamples": int(len(train)),
                "IsTradeWindow": bool(is_trade_window),
                "entry_price": float(row.entry_price),
                "exit_price": float(row.exit_price),
                "gross_r": float(row.gross_r),
                "exit_reason": row.exit_reason,
            }
        )

    predictions = pd.DataFrame(rows)
    feature_importance = (
        pd.concat(feature_importance_rows, ignore_index=True)
        if feature_importance_rows
        else pd.DataFrame()
    )
    diagnostics = {
        "status": "expanding",
        "candidates": int(len(dataset)),
        "allowed_trades": int(predictions["Allowed"].sum()),
        "rejected_trades": int((~predictions["Allowed"]).sum()),
        "threshold": float(config.ml.probability_threshold),
    }
    return ExpandingMLResult(allowed, predictions, feature_importance, diagnostics)


def write_ml_artifacts(
    output_dir: Path,
    variant_label: str,
    fold: int,
    result: MLFilterResult,
) -> None:
    if result.train_dataset.empty and result.predictions.empty:
        return
    prefix = f"ml_{variant_label}_fold_{fold}"
    result.train_dataset.to_csv(output_dir / f"{prefix}_train_dataset.csv", index=False)
    result.predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)
    if not result.feature_importance.empty:
        result.feature_importance.to_csv(output_dir / f"{prefix}_feature_importance.csv", index=False)


def write_expanding_ml_artifacts(
    output_dir: Path, variant_label: str, result: ExpandingMLResult, trades: pd.DataFrame | None = None
) -> None:
    if not result.predictions.empty:
        result.predictions.to_csv(output_dir / f"ml_{variant_label}_expanding_predictions.csv", index=False)
        summary = summarize_expanding_ml(result.predictions, trades)
        summary.to_csv(output_dir / f"ml_{variant_label}_summary.csv", index=False)
    if not result.feature_importance.empty:
        importance = (
            result.feature_importance.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        importance.to_csv(output_dir / f"ml_{variant_label}_feature_importance.csv", index=False)


def summarize_expanding_ml(predictions: pd.DataFrame, trades: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    trade_window = predictions[predictions["IsTradeWindow"].astype(bool)].copy()
    rows.append(_prediction_summary("all_candidates", trade_window))
    rows.append(_prediction_summary("accepted_candidates", trade_window[trade_window["Allowed"].astype(bool)]))
    rows.append(_prediction_summary("rejected_candidates", trade_window[~trade_window["Allowed"].astype(bool)]))
    if trades is not None and not trades.empty and "Source" in trades.columns:
        for source, group in trades.groupby("Source", dropna=False):
            rows.append(_trade_summary(f"executed_{source}", group))
    return pd.DataFrame(rows)


def _prediction_summary(label: str, data: pd.DataFrame) -> dict[str, float | int | str | None]:
    if data.empty:
        return {"segment": label, "count": 0}
    wins = data[data["net_r"] > 0]
    losses = data[data["net_r"] < 0]
    gross_profit = float(wins["net_r"].sum())
    gross_loss = float(abs(losses["net_r"].sum()))
    return {
        "segment": label,
        "count": int(len(data)),
        "win_rate_pct": float((data["net_r"] > 0).mean() * 100),
        "avg_net_r": float(data["net_r"].mean()),
        "median_net_r": float(data["net_r"].median()),
        "total_net_r": float(data["net_r"].sum()),
        "profit_factor_r": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_probability": None if data["Probability"].isna().all() else float(data["Probability"].mean()),
    }


def _trade_summary(label: str, data: pd.DataFrame) -> dict[str, float | int | str | None]:
    wins = data[data["NetPnL"] > 0]
    losses = data[data["NetPnL"] < 0]
    gross_profit = float(wins["NetPnL"].sum())
    gross_loss = float(abs(losses["NetPnL"].sum()))
    return {
        "segment": label,
        "count": int(len(data)),
        "win_rate_pct": float((data["NetPnL"] > 0).mean() * 100),
        "avg_net_pnl": float(data["NetPnL"].mean()),
        "median_net_pnl": float(data["NetPnL"].median()),
        "total_net_pnl": float(data["NetPnL"].sum()),
        "profit_factor_pnl": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_probability": None
        if "Probability" not in data.columns or data["Probability"].isna().all()
        else float(data["Probability"].mean()),
    }


def build_candidate_dataset(
    prepared: pd.DataFrame, config: BacktestConfig, variant: VariantConfig
) -> pd.DataFrame:
    rows = []
    traded_sessions: set[object] = set()
    prepared = prepared.sort_values("DateTime").reset_index(drop=True).copy()
    prepared["DateTime"] = pd.to_datetime(prepared["DateTime"])

    for index, bar in prepared.iterrows():
        if index + 1 >= len(prepared) or bar.Session in traded_sessions:
            continue
        signal = _build_strategy_signal(bar, config, variant)
        if signal is None:
            continue
        outcome = _simulate_candidate(prepared, index, signal, config)
        if outcome is None:
            continue
        features = _candidate_features(prepared, index, bar, signal)
        rows.append(
            {
                "SignalTime": bar.DateTime,
                "Session": bar.Session,
                "Side": signal.side,
                **features,
                **outcome,
                "label": int(outcome["net_r"] > 0),
            }
        )
        traded_sessions.add(bar.Session)
    return pd.DataFrame(rows)


def _should_refit_ml(
    model: XGBClassifier | None,
    last_fit_index: int | None,
    current_index: int,
    frequency_sessions: int,
    train: pd.DataFrame,
) -> bool:
    if len(train) == 0:
        return False
    if model is None:
        return True
    if last_fit_index is None:
        return True
    return current_index - last_fit_index >= frequency_sessions


def _build_strategy_signal(
    bar: pd.Series, config: BacktestConfig, variant: VariantConfig
) -> Signal | None:
    if config.strategy is None:
        return None
    if config.strategy.name == "orb":
        return build_orb_signal(bar, config, variant)
    if config.strategy.name == "donchian":
        return build_donchian_signal(bar, config, variant)
    raise ValueError(f"Unsupported ML strategy: {config.strategy.name}")


def _fit_xgb(train: pd.DataFrame, config: BacktestConfig) -> XGBClassifier | None:
    if config.ml is None:
        return None
    if len(train) < config.ml.min_train_samples or train["label"].nunique() < 2:
        return None
    x_train = _feature_matrix(train)
    y_train = train["label"].astype(int)
    xgb = config.ml.xgboost
    model = XGBClassifier(
        n_estimators=xgb.n_estimators,
        max_depth=xgb.max_depth,
        learning_rate=xgb.learning_rate,
        subsample=xgb.subsample,
        colsample_bytree=xgb.colsample_bytree,
        random_state=xgb.random_state,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)
    return model


def _feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
    )


def _simulate_candidate(
    prepared: pd.DataFrame, signal_index: int, signal: Signal, config: BacktestConfig
) -> dict[str, float | str] | None:
    entry_index = signal_index + 1
    entry_bar = prepared.iloc[entry_index]
    session = entry_bar.Session
    slip = slippage_points(config.execution, config.instrument)
    entry_price = apply_entry_slippage(signal.side, float(entry_bar.Open), slip)
    if signal.stop_loss is None:
        return None
    risk_points = abs(entry_price - signal.stop_loss)
    if risk_points <= 0:
        return None

    stop_loss = signal.stop_loss
    break_even_moved = False
    exit_price = None
    exit_reason = "final_bar"
    session_bars = prepared.iloc[entry_index:]
    session_bars = session_bars[session_bars["Session"] == session]
    for _, bar in session_bars.iterrows():
        if signal.force_exit_time is not None and bar.DateTime >= signal.force_exit_time:
            exit_price = float(bar.Close)
            exit_reason = "force_exit"
            break

        high = float(bar.High)
        low = float(bar.Low)
        if not break_even_moved and signal.break_even_trigger is not None:
            if signal.side == "long" and high >= signal.break_even_trigger:
                stop_loss = signal.break_even_stop
                break_even_moved = True
            elif signal.side == "short" and low <= signal.break_even_trigger:
                stop_loss = signal.break_even_stop
                break_even_moved = True

        if signal.side == "long":
            stop_hit = stop_loss is not None and low <= stop_loss
            tp_hit = signal.take_profit is not None and high >= signal.take_profit
        else:
            stop_hit = stop_loss is not None and high >= stop_loss
            tp_hit = signal.take_profit is not None and low <= signal.take_profit
        if stop_hit and tp_hit:
            exit_price = stop_loss
            exit_reason = "stop_loss"
            break
        if stop_hit:
            exit_price = stop_loss
            exit_reason = "stop_loss"
            break
        if tp_hit:
            exit_price = signal.take_profit
            exit_reason = "take_profit"
            break
    if exit_price is None:
        exit_price = float(session_bars.iloc[-1].Close) if not session_bars.empty else float(entry_bar.Close)

    fill_exit = apply_exit_slippage(signal.side, float(exit_price), slip)
    direction = 1 if signal.side == "long" else -1
    gross_pnl = (fill_exit - entry_price) * direction * config.instrument.cfd_point_value
    commission = commission_for_units(1.0, config.execution, config.instrument) * 2
    net_pnl = gross_pnl - commission
    return {
        "entry_price": float(entry_price),
        "exit_price": float(fill_exit),
        "gross_r": float(gross_pnl / risk_points),
        "net_r": float(net_pnl / risk_points),
        "exit_reason": exit_reason,
    }


def _candidate_features(
    prepared: pd.DataFrame, index: int, bar: pd.Series, signal: Signal) -> dict[str, float]:
    close = float(bar.Close)
    high = float(bar.High)
    low = float(bar.Low)
    open_ = float(bar.Open)
    atr_session = _safe_float(bar.get("ATR_Session"))
    or_high = float(bar.OR_High)
    or_low = float(bar.OR_Low)
    or_width = max(or_high - or_low, 1e-9)
    breakout_distance = close - or_high if signal.side == "long" else or_low - close
    close_position = (close - or_low) / or_width
    close_vs_vwap = close - _safe_float(bar.get("VWAP"))
    bar_range = max(high - low, 1e-9)
    closes = prepared["Close"].astype(float)

    band = _band_features(bar, signal.side, close, atr_session)
    breakout_atr = _divide(breakout_distance, atr_session)
    or_width_atr = _divide(or_width, atr_session)
    vwap_atr = _divide(close_vs_vwap, atr_session)

    return {
        "side_long": float(signal.side == "long"),
        "side_short": float(signal.side == "short"),
        "entry_index_after_or": _safe_float(bar.get("EntryIndexAfterOR")),
        "or_width": or_width,
        "or_width_atr_ratio": or_width_atr,
        "breakout_distance": breakout_distance,
        "breakout_distance_atr_ratio": breakout_atr,
        "close_position_in_or_range": close_position,
        "close_vs_vwap": close_vs_vwap,
        "close_vs_vwap_atr_ratio": vwap_atr,
        "atr_session": atr_session,
        "atr_stop": _safe_float(bar.get("ATR_Stop")),
        "bar_range": bar_range,
        "bar_body": abs(close - open_),
        "bar_body_to_range": abs(close - open_) / bar_range,
        "rolling_return_3": _rolling_return(closes, index, 3),
        "rolling_return_6": _rolling_return(closes, index, 6),
        "rolling_volatility_12": _rolling_volatility(closes, index, 12),
        "day_of_week": float(pd.Timestamp(bar.DateTime).dayofweek),
        "month": float(pd.Timestamp(bar.DateTime).month),
        **band,
        "breakout_x_band_position": breakout_atr * band["price_in_active_band"],
        "or_width_x_band_width": or_width_atr * band["band_width_side"],
        "vwap_x_band_position": vwap_atr * band["price_in_active_band"],
        "atr_stop_x_band_width": _safe_float(bar.get("ATR_Stop")) * band["band_width_side"],
    }


def _band_features(bar: pd.Series, side: str, close: float, atr_session: float) -> dict[str, float]:
    if side == "long":
        ae = _safe_float(bar.get("Band_AE_Pos_Upper"))
        fe = _safe_float(bar.get("Band_FE_Pos_Lower"))
        width = fe - ae
        dist_ae = close - ae
        dist_fe = fe - close
        position = _divide(close - ae, width)
    else:
        ae = _safe_float(bar.get("Band_AE_Neg_Lower"))
        fe = _safe_float(bar.get("Band_FE_Neg_Upper"))
        width = ae - fe
        dist_ae = ae - close
        dist_fe = close - fe
        position = _divide(ae - close, width)
    return {
        "band_width_side": width,
        "price_in_active_band": position,
        "dist_to_active_ae": dist_ae,
        "dist_to_active_fe": dist_fe,
        "dist_to_active_ae_atr": _divide(dist_ae, atr_session),
        "dist_to_active_fe_atr": _divide(dist_fe, atr_session),
    }


def _fallback_allowed(dataset: pd.DataFrame, fallback: str) -> set[pd.Timestamp]:
    if fallback == "allow_all" and not dataset.empty:
        return set(pd.to_datetime(dataset["SignalTime"]))
    if fallback == "reject_all" or dataset.empty:
        return set()
    raise ValueError("ml.fallback must be allow_all or reject_all")


def _fallback_predictions(dataset: pd.DataFrame, reason: str) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    predictions = dataset[["SignalTime", "Session", "Side", "label", "net_r"]].copy()
    predictions["Probability"] = np.nan
    predictions["Allowed"] = True
    predictions["Reason"] = reason
    return predictions


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(result) or np.isinf(result):
        return 0.0
    return result


def _divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return 0.0
    return float(numerator / denominator)


def _rolling_return(closes: pd.Series, index: int, periods: int) -> float:
    if index - periods < 0:
        return 0.0
    previous = float(closes.iloc[index - periods])
    if previous == 0:
        return 0.0
    return float(closes.iloc[index] / previous - 1.0)


def _rolling_volatility(closes: pd.Series, index: int, periods: int) -> float:
    start = max(0, index - periods + 1)
    returns = closes.iloc[start : index + 1].pct_change().dropna()
    return 0.0 if returns.empty else float(returns.std(ddof=0))
