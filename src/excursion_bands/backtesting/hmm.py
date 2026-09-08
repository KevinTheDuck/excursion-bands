"""
Hidden Markov regime filter for strategy candidates.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from excursion_bands.backtesting.costs import (
    apply_entry_slippage,
    apply_exit_slippage,
    commission_for_units,
    slippage_points,
)
from excursion_bands.backtesting.engine import _BarView, _exit_from_bar
from excursion_bands.backtesting.models import Position, Signal
from excursion_bands.backtesting.specification import BacktestConfig, VariantConfig
from excursion_bands.backtesting.strategies.donchian import build_donchian_signal
from excursion_bands.backtesting.strategies.orb import build_orb_signal

FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "volatility_12",
    "bar_range_atr",
    "body_to_range",
    "close_vs_vwap_atr",
    "atr_stop_atr_session",
]


@dataclass(frozen=True)
class HMMFilterResult:
    allowed_signal_times: set[pd.Timestamp]
    train_dataset: pd.DataFrame
    predictions: pd.DataFrame
    diagnostics: dict[str, float | int | str]


@dataclass
class GaussianHMM:
    start_prob: np.ndarray
    trans_prob: np.ndarray
    means: np.ndarray
    variances: np.ndarray


def train_hmm_filter(
    train_prepared: pd.DataFrame,
    test_prepared: pd.DataFrame,
    config: BacktestConfig,
    variant: VariantConfig,
) -> HMMFilterResult:
    if config.hmm is None or not config.hmm.enabled:
        return HMMFilterResult(set(), pd.DataFrame(), pd.DataFrame(), {"status": "disabled"})

    train_prepared = _add_hmm_features(train_prepared)
    # Feature windows for the first test bars must retain the final training
    # observations.  Recomputing pct-change/rolling features on test-only
    # rows silently resets those windows and changes the HMM state sequence.
    test_prepared = _add_hmm_features(test_prepared, history=train_prepared)
    train_dataset = build_candidate_dataset(train_prepared, config, variant)
    test_dataset = build_candidate_dataset(test_prepared, config, variant)
    fallback_allowed = _fallback_allowed(test_dataset, config.hmm.fallback)
    if len(train_dataset) < config.hmm.min_train_samples or test_dataset.empty:
        return HMMFilterResult(
            fallback_allowed,
            train_dataset,
            _fallback_predictions(test_dataset, "insufficient_samples", fallback_allowed),
            {"status": "insufficient_samples", "train_samples": len(train_dataset)},
        )

    # HMM observations are strategy candidates, rather than every market bar.
    # This is both the causal unit being filtered and several orders of
    # magnitude smaller than a multi-year intraday frame.
    train_features = train_dataset.dropna(subset=FEATURE_COLUMNS).copy()
    test_features = test_dataset.dropna(subset=FEATURE_COLUMNS).copy()
    if len(train_features) < config.hmm.min_train_samples or test_features.empty:
        return HMMFilterResult(
            fallback_allowed,
            train_dataset,
            _fallback_predictions(test_dataset, "insufficient_bar_samples", fallback_allowed),
            {"status": "insufficient_bar_samples", "train_samples": len(train_features)},
        )

    x_train = _feature_matrix(train_features)
    x_test = _feature_matrix(test_features)
    # Reserve the most recent candidate observations for chronological state
    # validation.  The HMM parameters are fitted only on the earlier feature
    # sequence so validation outcomes cannot influence the regime model.
    state_train_dataset, preliminary_validation = _split_state_selection_data(
        train_dataset, config
    )
    if preliminary_validation.empty:
        fit_features = train_features
    else:
        validation_start = pd.to_datetime(preliminary_validation["SignalTime"].iloc[0])
        fit_features = train_features[train_features["SignalTime"] < validation_start]
        if len(fit_features) < config.hmm.min_train_samples:
            return HMMFilterResult(
                fallback_allowed, train_dataset,
                _fallback_predictions(test_dataset, "insufficient_fit_samples", fallback_allowed),
                {"status": "insufficient_fit_samples", "train_samples": len(fit_features)},
            )
    x_fit = _feature_matrix(fit_features)
    model = _fit_hmm(x_fit, config.hmm.n_states, config.hmm.max_iter, config.hmm.random_state)
    # Forward filtering is causal; Viterbi would use future observations when
    # assigning a state to an earlier candidate.
    train_dataset = train_features.copy()
    test_dataset = test_features.copy()
    train_dataset["HMMState"] = _filter_states(model, x_train)
    # Continue the forward-filtering posterior from the final training
    # candidate.  Resetting to ``start_prob`` at the fold boundary would make
    # the first test state depend on an artificial new sequence.
    test_states = _filter_states(model, np.vstack([x_train, x_test]))[-len(x_test) :]
    test_dataset["HMMState"] = test_states
    train_dataset["HMMState"] = train_dataset["HMMState"].astype(int)
    test_dataset["HMMState"] = test_dataset["HMMState"].astype(int)
    state_train_dataset, state_validation_dataset = _split_state_selection_data(train_dataset, config)
    tradable_states = _select_tradable_states(state_train_dataset, config)
    validation = _validate_tradable_states(state_validation_dataset, tradable_states, config)
    if not validation["use_filter"]:
        tradable_states = set(test_dataset["HMMState"].unique())

    predictions = test_dataset[["SignalTime", "Session", "Side", "net_r"]].copy()
    predictions["HMMState"] = test_dataset["HMMState"].to_numpy()
    predictions["Allowed"] = predictions["HMMState"].isin(tradable_states)
    allowed = set(pd.to_datetime(predictions.loc[predictions["Allowed"], "SignalTime"]))
    diagnostics = {
        "status": "trained",
        "train_samples": len(train_dataset),
        "test_candidates": len(test_dataset),
        "allowed_trades": int(predictions["Allowed"].sum()),
        "rejected_trades": int((~predictions["Allowed"]).sum()),
        "tradable_states": ",".join(str(state) for state in sorted(tradable_states)),
        "filter_active": int(validation["use_filter"]),
        "validation_trades": int(validation["trades"]),
        "validation_allowed_trades": int(validation["allowed_trades"]),
        "validation_allow_rate": float(validation["allow_rate"]),
        "validation_baseline_net_r": float(validation["baseline_net_r"]),
        "validation_filtered_net_r": float(validation["filtered_net_r"]),
        "validation_net_r_improvement": float(validation["net_r_improvement"]),
        "validation_reason": str(validation["reason"]),
    }
    return HMMFilterResult(allowed, train_dataset, predictions, diagnostics)


def write_hmm_artifacts(
    output_dir: Path, variant_label: str, fold: int, result: HMMFilterResult
) -> None:
    if result.train_dataset.empty and result.predictions.empty:
        return
    prefix = f"hmm_{variant_label}_fold_{fold}"
    result.train_dataset.to_csv(output_dir / f"{prefix}_train_dataset.csv", index=False)
    result.predictions.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)


def build_candidate_dataset(
    prepared: pd.DataFrame, config: BacktestConfig, variant: VariantConfig
) -> pd.DataFrame:
    rows = []
    traded_sessions: set[object] = set()
    prepared = prepared.sort_values("DateTime").reset_index(drop=True).copy()
    prepared["DateTime"] = pd.to_datetime(prepared["DateTime"])
    for index, row in enumerate(prepared.itertuples(index=False)):
        bar = _BarView(row, index)
        if index + 1 >= len(prepared) or bar.Session in traded_sessions:
            continue
        signal = _build_strategy_signal(bar, config, variant)
        if signal is None:
            continue
        outcome = _simulate_candidate(prepared, index, signal, config)
        if outcome is None:
            continue
        rows.append(
            {
                "SignalTime": bar.DateTime,
                "Session": bar.Session,
                "Side": signal.side,
                **_candidate_features(prepared, index, bar),
                **outcome,
            }
        )
        traded_sessions.add(bar.Session)
    return pd.DataFrame(rows)


def _add_hmm_features(
    prepared: pd.DataFrame, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    target = prepared.sort_values("DateTime").reset_index(drop=True).copy()
    target["DateTime"] = pd.to_datetime(target["DateTime"])
    if history is not None and not history.empty:
        history = history.sort_values("DateTime").copy()
        history["DateTime"] = pd.to_datetime(history["DateTime"])
        base_columns = [
            column
            for column in (
                "DateTime", "Session", "Open", "High", "Low", "Close", "Volume",
                "ATR_Session", "VWAP", "ATR_Stop",
            )
            if column in target.columns and column in history.columns
        ]
        combined = pd.concat(
            [history.reindex(columns=base_columns), target.reindex(columns=base_columns)],
            ignore_index=True,
        ).drop_duplicates("DateTime", keep="last").sort_values("DateTime").reset_index(drop=True)
    else:
        combined = target

    prepared = combined.copy()
    closes = prepared["Close"].astype(float)
    returns = closes.pct_change()
    high = prepared["High"].astype(float)
    low = prepared["Low"].astype(float)
    open_ = prepared["Open"].astype(float)
    close = prepared["Close"].astype(float)
    bar_range = (high - low).clip(lower=1e-9)
    atr_session = pd.to_numeric(prepared.get("ATR_Session", 0.0), errors="coerce").replace(0, np.nan)
    vwap = pd.to_numeric(prepared.get("VWAP", close), errors="coerce")
    atr_stop = pd.to_numeric(prepared.get("ATR_Stop", 0.0), errors="coerce")
    prepared["return_1"] = returns.fillna(0.0)
    prepared["return_3"] = closes.pct_change(3).fillna(0.0)
    prepared["return_6"] = closes.pct_change(6).fillna(0.0)
    prepared["volatility_12"] = returns.rolling(12, min_periods=3).std().fillna(0.0)
    prepared["bar_range_atr"] = (bar_range / atr_session).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prepared["body_to_range"] = ((close - open_).abs() / bar_range).fillna(0.0)
    prepared["close_vs_vwap_atr"] = ((close - vwap) / atr_session).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prepared["atr_stop_atr_session"] = (atr_stop / atr_session).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features = prepared[["DateTime", *FEATURE_COLUMNS]]
    return target.drop(columns=FEATURE_COLUMNS, errors="ignore").merge(
        features[features["DateTime"].isin(set(target["DateTime"]))],
        on="DateTime",
        how="left",
        validate="one_to_one",
    )


def _build_strategy_signal(
    bar: pd.Series, config: BacktestConfig, variant: VariantConfig
) -> Signal | None:
    if config.strategy is None:
        return None
    if config.strategy.name == "orb":
        return build_orb_signal(bar, config, variant)
    if config.strategy.name == "donchian":
        return build_donchian_signal(bar, config, variant)
    raise ValueError(f"Unsupported HMM strategy: {config.strategy.name}")


def _candidate_features(prepared: pd.DataFrame, index: int, bar: pd.Series) -> dict[str, float]:
    if all(column in bar.index for column in FEATURE_COLUMNS):
        values = {
            column: _safe_float(bar[column])
            for column in FEATURE_COLUMNS
        }
        if all(np.isfinite(value) for value in values.values()):
            return values
    close = float(bar.Close)
    open_ = float(bar.Open)
    high = float(bar.High)
    low = float(bar.Low)
    closes = prepared["Close"].astype(float)
    atr_session = _safe_float(bar.get("ATR_Session"))
    bar_range = max(high - low, 1e-9)
    return {
        "return_1": _rolling_return(closes, index, 1),
        "return_3": _rolling_return(closes, index, 3),
        "return_6": _rolling_return(closes, index, 6),
        "volatility_12": _rolling_volatility(closes, index, 12),
        "bar_range_atr": _divide(bar_range, atr_session),
        "body_to_range": abs(close - open_) / bar_range,
        "close_vs_vwap_atr": _divide(close - _safe_float(bar.get("VWAP")), atr_session),
        "atr_stop_atr_session": _divide(_safe_float(bar.get("ATR_Stop")), atr_session),
    }


def _feature_matrix(data: pd.DataFrame) -> np.ndarray:
    return (
        data[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(float)
        .to_numpy()
    )


def _fit_hmm(x: np.ndarray, n_states: int, max_iter: int, random_state: int) -> GaussianHMM:
    rng = np.random.default_rng(random_state)
    n_obs, _n_features = x.shape
    quantiles = np.linspace(0, 1, n_states + 2)[1:-1]
    order_feature = x[:, 0]
    means = []
    for q in quantiles:
        center = np.quantile(order_feature, q)
        means.append(x[np.argmin(np.abs(order_feature - center))])
    means = np.asarray(means, dtype=float)
    if len(means) < n_states:
        means = x[rng.choice(n_obs, size=n_states, replace=n_obs < n_states)]
    variances = np.tile(np.var(x, axis=0) + 1e-6, (n_states, 1))
    start_prob = np.full(n_states, 1.0 / n_states)
    trans_prob = np.full((n_states, n_states), 1.0 / n_states)

    for _ in range(max_iter):
        log_emit = _log_emissions(x, means, variances)
        gamma, xi_sum, _log_likelihood = _forward_backward(log_emit, start_prob, trans_prob)
        weights = gamma.sum(axis=0) + 1e-12
        means = (gamma.T @ x) / weights[:, None]
        for state in range(n_states):
            diff = x - means[state]
            variances[state] = (gamma[:, state][:, None] * diff * diff).sum(axis=0) / weights[state]
        variances = np.maximum(variances, 1e-6)
        start_prob = gamma[0] / gamma[0].sum()
        trans_prob = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
    return GaussianHMM(start_prob, trans_prob, means, variances)


def _log_emissions(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    log_probs = []
    for mean, variance in zip(means, variances):
        log_det = np.log(2 * np.pi * variance).sum()
        quad = (((x - mean) ** 2) / variance).sum(axis=1)
        log_probs.append(-0.5 * (log_det + quad))
    return np.column_stack(log_probs)


def _forward_backward(
    log_emit: np.ndarray, start_prob: np.ndarray, trans_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    n_obs, n_states = log_emit.shape
    log_start = np.log(start_prob + 1e-12)
    log_trans = np.log(trans_prob + 1e-12)
    alpha = np.zeros((n_obs, n_states))
    beta = np.zeros((n_obs, n_states))
    alpha[0] = log_start + log_emit[0]
    for t in range(1, n_obs):
        alpha[t] = log_emit[t] + _logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
    for t in range(n_obs - 2, -1, -1):
        beta[t] = _logsumexp(log_trans + log_emit[t + 1] + beta[t + 1], axis=1)
    log_likelihood = float(_logsumexp(alpha[-1], axis=0))
    gamma_log = alpha + beta - log_likelihood
    gamma = np.exp(gamma_log)
    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_obs - 1):
        xi_log = alpha[t][:, None] + log_trans + log_emit[t + 1] + beta[t + 1] - log_likelihood
        xi_sum += np.exp(xi_log)
    return gamma, xi_sum, log_likelihood


def _viterbi(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    log_emit = _log_emissions(x, model.means, model.variances)
    log_start = np.log(model.start_prob + 1e-12)
    log_trans = np.log(model.trans_prob + 1e-12)
    n_obs, n_states = log_emit.shape
    delta = np.zeros((n_obs, n_states))
    psi = np.zeros((n_obs, n_states), dtype=int)
    delta[0] = log_start + log_emit[0]
    for t in range(1, n_obs):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_emit[t] + np.max(scores, axis=0)
    states = np.zeros(n_obs, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(n_obs - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def _filter_states(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    log_emit = _log_emissions(x, model.means, model.variances)
    log_trans = np.log(model.trans_prob + 1e-12)
    alpha = np.log(model.start_prob + 1e-12) + log_emit[0]
    states = [int(np.argmax(alpha))]
    for t in range(1, len(x)):
        alpha = log_emit[t] + _logsumexp(alpha[:, None] + log_trans, axis=0)
        alpha = alpha - _logsumexp(alpha, axis=0)
        states.append(int(np.argmax(alpha)))
    return np.asarray(states, dtype=int)


def _split_state_selection_data(
    dataset: pd.DataFrame, config: BacktestConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert config.hmm is not None
    if dataset.empty:
        return dataset, dataset
    validation_fraction = min(max(float(config.hmm.validation_fraction), 0.0), 0.9)
    if validation_fraction <= 0.0:
        return dataset, pd.DataFrame(columns=dataset.columns)
    data = dataset.sort_values("SignalTime").reset_index(drop=True)
    validation_size = max(config.hmm.min_validation_trades, round(len(data) * validation_fraction))
    if validation_size >= len(data):
        return data, pd.DataFrame(columns=data.columns)
    split = len(data) - validation_size
    return data.iloc[:split].copy(), data.iloc[split:].copy()


def _validate_tradable_states(
    dataset: pd.DataFrame, tradable_states: set[int], config: BacktestConfig
) -> dict[str, float | int | str | bool]:
    assert config.hmm is not None
    if dataset.empty:
        return _validation_result(False, 0, 0, 0.0, 0.0, 0.0, "no_validation_split")
    net_r = pd.to_numeric(dataset["net_r"], errors="coerce").fillna(0.0)
    allowed_mask = dataset["HMMState"].isin(tradable_states)
    trades = len(dataset)
    allowed_trades = int(allowed_mask.sum())
    allow_rate = allowed_trades / trades if trades else 0.0
    baseline_net_r = float(net_r.sum())
    filtered_net_r = float(net_r[allowed_mask].sum())
    improvement = filtered_net_r - baseline_net_r
    if trades < config.hmm.min_validation_trades:
        reason = "insufficient_validation_trades"
        use_filter = False
    elif allowed_trades == 0:
        reason = "no_allowed_validation_trades"
        use_filter = False
    elif allow_rate < config.hmm.min_validation_allow_rate:
        reason = "validation_allow_rate_too_low"
        use_filter = False
    elif improvement < config.hmm.min_validation_net_r_improvement:
        reason = "validation_improvement_too_low"
        use_filter = False
    else:
        reason = "validation_passed"
        use_filter = True
    return _validation_result(
        use_filter,
        trades,
        allowed_trades,
        allow_rate,
        baseline_net_r,
        filtered_net_r,
        reason,
    )


def _validation_result(
    use_filter: bool,
    trades: int,
    allowed_trades: int,
    allow_rate: float,
    baseline_net_r: float,
    filtered_net_r: float,
    reason: str,
) -> dict[str, float | int | str | bool]:
    return {
        "use_filter": use_filter,
        "trades": trades,
        "allowed_trades": allowed_trades,
        "allow_rate": allow_rate,
        "baseline_net_r": baseline_net_r,
        "filtered_net_r": filtered_net_r,
        "net_r_improvement": filtered_net_r - baseline_net_r,
        "reason": reason,
    }


def _select_tradable_states(dataset: pd.DataFrame, config: BacktestConfig) -> set[int]:
    assert config.hmm is not None
    qualified_states: list[tuple[float, float, float, int, int]] = []
    ranked_states: list[tuple[float, float, float, int, int]] = []
    for state, group in dataset.groupby("HMMState"):
        net_r = pd.to_numeric(group["net_r"], errors="coerce").fillna(0.0)
        if len(group) < config.hmm.min_state_trades:
            continue
        net_sum = float(net_r.sum())
        expectancy = float(net_r.mean())
        win_rate = float((net_r > 0).mean())
        ranked_states.append((expectancy, net_sum, win_rate, len(group), int(state)))
        if net_sum < config.hmm.min_state_net_r:
            continue
        if win_rate < config.hmm.min_state_win_rate:
            continue
        qualified_states.append((expectancy, net_sum, win_rate, len(group), int(state)))
    max_states = max(0, int(config.hmm.top_states))
    if max_states == 0:
        return set()
    qualified_states.sort(reverse=True)
    ranked_states.sort(reverse=True)
    selected = {state for _, _, _, _, state in qualified_states[:max_states]}
    for _, _, _, _, state in ranked_states:
        selected.add(state)
        if len(selected) >= max_states:
            break
    return selected


def _simulate_candidate(
    prepared: pd.DataFrame, signal_index: int, signal: Signal, config: BacktestConfig
) -> dict[str, float | str] | None:
    entry_index = signal_index + 1
    entry_bar = prepared.iloc[entry_index]
    if signal.force_exit_time is not None and entry_bar.DateTime >= signal.force_exit_time:
        return None
    session = entry_bar.Session
    if signal.session is not None and session != signal.session:
        return None
    slip = slippage_points(config.execution, config.instrument)
    entry_price = apply_entry_slippage(signal.side, float(entry_bar.Open), slip)
    if signal.stop_loss is None:
        return None
    risk_points = abs(entry_price - signal.stop_loss)
    if risk_points <= 0:
        return None
    stop_loss = signal.stop_loss
    position = Position(
        side=signal.side,
        units=1.0,
        entry_time=entry_bar.DateTime,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=signal.take_profit,
        force_exit_time=signal.force_exit_time,
        entry_commission=commission_for_units(1.0, config.execution, config.instrument),
        break_even_trigger=signal.break_even_trigger,
        break_even_stop=signal.break_even_stop,
        initial_stop_loss=stop_loss,
    )
    exit_price = None
    exit_reason = "final_bar"
    end_index = len(prepared)
    if signal.force_exit_time is not None:
        end_index = min(
            len(prepared),
            int(prepared["DateTime"].searchsorted(signal.force_exit_time, side="left")) + 1,
        )
    session_bars = prepared.iloc[entry_index:end_index]
    session_bars = session_bars[session_bars["Session"] == session]
    for _, bar in session_bars.iterrows():
        exit_candidate = _exit_from_bar(
            position, bar, config.execution.same_bar_exit_priority
        )
        if exit_candidate is not None:
            exit_price, exit_reason = exit_candidate
            break
    if exit_price is None:
        exit_price = float(session_bars.iloc[-1].Close) if not session_bars.empty else float(entry_bar.Close)
    fill_exit = apply_exit_slippage(signal.side, float(exit_price), slip)
    direction = 1 if signal.side == "long" else -1
    gross_pnl = (fill_exit - entry_price) * direction * config.instrument.cfd_point_value
    commission = commission_for_units(1.0, config.execution, config.instrument) * 2
    net_pnl = gross_pnl - commission
    risk_amount = risk_points * config.instrument.cfd_point_value
    return {
        "entry_price": float(entry_price),
        "exit_price": float(fill_exit),
        "gross_r": float(gross_pnl / risk_amount) if risk_amount > 0 else 0.0,
        "net_r": float(net_pnl / risk_amount) if risk_amount > 0 else 0.0,
        "exit_reason": exit_reason,
    }


def _fallback_allowed(dataset: pd.DataFrame, fallback: str) -> set[pd.Timestamp]:
    if fallback == "allow_all" and not dataset.empty:
        return set(pd.to_datetime(dataset["SignalTime"]))
    if fallback == "reject_all" or dataset.empty:
        return set()
    raise ValueError("hmm.fallback must be allow_all or reject_all")


def _fallback_predictions(
    dataset: pd.DataFrame, reason: str, allowed_signal_times: set[pd.Timestamp] | None = None
) -> pd.DataFrame:
    if dataset.empty:
        return pd.DataFrame()
    predictions = dataset[["SignalTime", "Session", "Side", "net_r"]].copy()
    predictions["HMMState"] = -1
    allowed = set(allowed_signal_times or set())
    predictions["Allowed"] = pd.to_datetime(predictions["SignalTime"]).isin(allowed)
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
    return 0.0 if returns.empty else float(returns.std())


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    result = max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)
