"""Pure selection primitives for robust, regime-aware walk-forward research.

The functions in this module do not run backtests or mutate optimizer state.  They
operate on already-computed daily returns and trades so that the WFO orchestrator
can keep search, calibration, validation, and out-of-sample periods separate.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import pandas as pd

REGIMES = ("low", "normal", "high")


def _session_key(value: object) -> object:
    """Return a date-like value in the same canonical form as engine sessions."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return value
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return value


def _daily_returns(values: pd.Series) -> tuple[pd.Series, str | None]:
    if not isinstance(values, pd.Series) or values.empty:
        return pd.Series(dtype=float), "no_daily_returns"
    result = pd.to_numeric(values, errors="coerce").astype(float)
    if not np.isfinite(result.to_numpy()).all():
        return result, "nonfinite_returns"
    result.index = pd.Index(
        [_session_key(value) for value in result.index], name="Session"
    )
    if result.index.has_duplicates:
        return result, "duplicate_return_sessions"
    return result.sort_index(), None


def _trade_sessions(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=object)
    if "Session" in trades:
        source = trades["Session"]
    elif "EntryTime" in trades:
        source = trades["EntryTime"]
    else:
        raise ValueError("trades must contain Session or EntryTime")
    return source.map(_session_key)


def _drawdown_from_returns(returns: pd.Series) -> float:
    values = returns.to_numpy(dtype=float)
    equity = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    peaks = np.maximum.accumulate(equity)
    return float(np.min(equity / peaks - 1.0))


def _ineligible_score(
    reason: str,
    total_trades: int,
    block_sharpes: list[float] | None = None,
    max_drawdown: float | None = None,
) -> dict[str, Any]:
    return {
        "score": float("-inf"),
        "eligible": False,
        "reason": reason,
        "block_sharpes": block_sharpes or [],
        "max_drawdown": max_drawdown,
        "total_trades": int(total_trades),
    }


def score_candidate(
    daily_returns: pd.Series,
    trades: pd.DataFrame,
    blocks: list[Collection[object]],
    min_total_trades: int = 60,
    min_block_trades: int = 10,
) -> dict[str, Any]:
    """Score a candidate on stable performance across chronological blocks.

    ``daily_returns`` must have one value for every session, including sessions
    with no trades.  ``max_drawdown`` follows the rest of the backtesting package
    and is returned as a non-positive fraction.
    """
    if min_total_trades < 0 or min_block_trades < 0:
        raise ValueError("minimum trade counts cannot be negative")
    if not blocks:
        return _ineligible_score("no_blocks", len(trades))

    returns, error = _daily_returns(daily_returns)
    total_trades = len(trades)
    if error is not None:
        return _ineligible_score(error, total_trades)
    if (returns <= -1.0).any():
        return _ineligible_score("bankrupt_return", total_trades)
    if total_trades < min_total_trades:
        return _ineligible_score("insufficient_total_trades", total_trades)

    try:
        trade_sessions = _trade_sessions(trades)
    except ValueError as exc:
        return _ineligible_score(str(exc), total_trades)

    block_sharpes: list[float] = []
    for index, block in enumerate(blocks):
        sessions = {_session_key(value) for value in block}
        block_returns = returns[returns.index.isin(sessions)]
        if block_returns.empty:
            return _ineligible_score(
                f"block_{index}_has_no_returns", total_trades, block_sharpes
            )
        block_trade_count = int(trade_sessions.isin(sessions).sum())
        if block_trade_count < min_block_trades:
            return _ineligible_score(
                f"block_{index}_insufficient_trades", total_trades, block_sharpes
            )
        standard_deviation = float(block_returns.std(ddof=0))
        if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
            return _ineligible_score(
                f"block_{index}_zero_variance", total_trades, block_sharpes
            )
        sharpe = float(block_returns.mean() / standard_deviation * math.sqrt(252.0))
        if not math.isfinite(sharpe):
            return _ineligible_score(
                f"block_{index}_invalid_sharpe", total_trades, block_sharpes
            )
        block_sharpes.append(sharpe)

    max_drawdown = _drawdown_from_returns(returns)
    if not math.isfinite(max_drawdown):
        return _ineligible_score(
            "invalid_equity_path", total_trades, block_sharpes, max_drawdown
        )
    quartile_25, quartile_75 = np.percentile(block_sharpes, [25.0, 75.0])
    score = float(
        np.median(block_sharpes) - 0.5 * (quartile_75 - quartile_25) - abs(max_drawdown)
    )
    return {
        "score": score,
        "eligible": True,
        "reason": "eligible",
        "block_sharpes": block_sharpes,
        "max_drawdown": max_drawdown,
        "total_trades": int(total_trades),
    }


def _numeric_spec(specification: object) -> tuple[float, float, float] | None:
    if not isinstance(specification, Mapping):
        return None
    if not {"low", "high", "step"}.issubset(specification):
        return None
    low = specification["low"]
    high = specification["high"]
    step = specification["step"]
    if any(isinstance(value, (bool, np.bool_)) for value in (low, high, step)):
        return None
    try:
        numeric = (float(low), float(high), float(step))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in numeric) or numeric[2] <= 0:
        return None
    if numeric[0] > numeric[1]:
        return None
    return numeric


def numeric_neighbors(
    params: Mapping[str, Any], search_space: Mapping[str, object]
) -> list[dict[str, Any]]:
    """Return one-step numeric neighbors, changing exactly one parameter.

    Numeric search-space entries use ``{"low": ..., "high": ..., "step": ...}``.
    Categorical or malformed entries are ignored.  Results follow parameter-key
    order and list the lower neighbor before the upper neighbor.
    """
    neighbors: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for key in params:
        if key not in search_space:
            continue
        space_entry = search_space[key]
        if not isinstance(space_entry, Mapping):
            continue
        specification = _numeric_spec(space_entry)
        current = params[key]
        if specification is None or isinstance(current, (bool, np.bool_)):
            continue
        try:
            current_number = float(current)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(current_number):
            continue
        low, high, step = specification
        integer_parameter = all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in (
                current,
                space_entry["low"],
                space_entry["high"],
                space_entry["step"],
            )
        )
        for candidate_number in (current_number - step, current_number + step):
            tolerance = max(1.0, abs(low), abs(high)) * 1e-12
            if (
                candidate_number < low - tolerance
                or candidate_number > high + tolerance
            ):
                continue
            candidate_number = min(high, max(low, candidate_number))
            candidate_value: int | float
            candidate_value = (
                round(candidate_number) if integer_parameter else candidate_number
            )
            if candidate_value == current:
                continue
            candidate = dict(params)
            candidate[key] = candidate_value
            fingerprint = tuple(
                sorted((name, repr(value)) for name, value in candidate.items())
            )
            if fingerprint not in seen:
                seen.add(fingerprint)
                neighbors.append(candidate)
    return neighbors


def session_volatility(intraday: pd.DataFrame) -> pd.Series:
    """Calculate causal 20-session close-to-close realized volatility.

    The value assigned to a session uses only returns from the 20 completed
    sessions preceding it.  No annualization is applied because only relative
    regime thresholds are needed.
    """
    required = {"Session", "Close"}
    missing = required.difference(intraday.columns)
    if missing:
        raise ValueError(f"intraday is missing required columns: {sorted(missing)}")
    frame = intraday.copy()
    frame["Session"] = frame["Session"].map(_session_key)
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    if "DateTime" in frame:
        frame["DateTime"] = pd.to_datetime(frame["DateTime"], errors="coerce")
        frame = frame.sort_values(["Session", "DateTime"])
    else:
        frame = frame.sort_values("Session", kind="stable")
    closes = frame.groupby("Session", sort=True, dropna=True)["Close"].last()
    if (
        closes.empty
        or not np.isfinite(closes.to_numpy(dtype=float)).all()
        or (closes <= 0).any()
    ):
        raise ValueError("session closes must be finite and positive")
    log_returns = np.log(closes).diff()
    volatility = log_returns.rolling(20, min_periods=20).std(ddof=0).shift(1)
    volatility.name = "Volatility20"
    return volatility


def fit_regime_thresholds(volatility: pd.Series) -> tuple[float, float] | None:
    """Fit low/normal/high tercile thresholds on finite observations."""
    values = pd.to_numeric(volatility, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    lower, upper = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    return float(lower), float(upper)


def classify_regimes(
    volatility: pd.Series, thresholds: tuple[float, float] | None
) -> pd.Series:
    """Classify volatility as low, normal, high, or unknown."""
    result = pd.Series("unknown", index=volatility.index, dtype=object, name="Regime")
    if thresholds is None:
        return result
    lower, upper = thresholds
    if not all(math.isfinite(float(value)) for value in thresholds) or lower > upper:
        raise ValueError("regime thresholds must be finite and ordered")
    numeric = pd.to_numeric(volatility, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float))
    result.loc[finite & (numeric <= lower)] = "low"
    result.loc[finite & (numeric > lower) & (numeric <= upper)] = "normal"
    result.loc[finite & (numeric > upper)] = "high"
    return result


def calibrate_library(
    candidate_returns: dict[str, pd.Series],
    candidate_trades: dict[str, pd.DataFrame],
    regimes: pd.Series,
    general_id: str,
    min_sessions: int = 60,
    min_trades: int = 20,
    shrinkage_sessions: int = 126,
) -> dict[str, Any]:
    """Select a candidate per regime using shrunk daily-return moments.

    Returns ``{"mapping": {regime: candidate_id}, "diagnostics": ...}``.
    Candidates without enough regime observations or trades are recorded but
    cannot displace the general fallback.
    """
    if min_sessions < 1 or min_trades < 0 or shrinkage_sessions < 0:
        raise ValueError("invalid calibration thresholds")
    if general_id not in candidate_returns:
        raise ValueError("general_id must be present in candidate_returns")

    regime_values = regimes.copy()
    regime_values.index = pd.Index(
        [_session_key(value) for value in regime_values.index], name="Session"
    )
    if regime_values.index.has_duplicates:
        raise ValueError("regimes must contain at most one label per session")
    regime_values = regime_values.astype(object)
    mapping: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}

    prepared: dict[str, tuple[pd.Series, pd.Series] | None] = {}
    for candidate_id in sorted(candidate_returns):
        returns, error = _daily_returns(candidate_returns[candidate_id])
        if error is not None or (returns <= -1.0).any():
            prepared[candidate_id] = None
            continue
        returns = returns[returns.index.isin(regime_values.index)]
        try:
            trades = candidate_trades.get(candidate_id, pd.DataFrame())
            trade_sessions = _trade_sessions(trades)
        except ValueError:
            prepared[candidate_id] = None
            continue
        prepared[candidate_id] = returns, trade_sessions

    for regime in REGIMES:
        regime_sessions = set(regime_values.index[regime_values == regime])
        candidate_diagnostics: dict[str, dict[str, Any]] = {}
        eligible_candidates: list[tuple[float, str]] = []
        for candidate_id in sorted(candidate_returns):
            candidate = prepared[candidate_id]
            if candidate is None:
                candidate_diagnostics[candidate_id] = {
                    "eligible": False,
                    "reason": "invalid_candidate_data",
                    "sessions": 0,
                    "trades": 0,
                    "score": None,
                }
                continue
            returns, trade_sessions = candidate
            regime_returns = returns[returns.index.isin(regime_sessions)]
            session_count = len(regime_returns)
            trade_count = int(trade_sessions.isin(regime_sessions).sum())
            reason = "eligible"
            score: float | None = None
            weight: float | None = None
            if session_count < min_sessions:
                reason = "insufficient_sessions"
            elif trade_count < min_trades:
                reason = "insufficient_trades"
            elif returns.empty:
                reason = "no_candidate_returns"
            else:
                weight = session_count / (session_count + shrinkage_sessions)
                global_mean = float(returns.mean())
                global_second = float((returns * returns).mean())
                regime_mean = float(regime_returns.mean())
                regime_second = float((regime_returns * regime_returns).mean())
                shrunk_mean = weight * regime_mean + (1.0 - weight) * global_mean
                shrunk_second = weight * regime_second + (1.0 - weight) * global_second
                variance = max(0.0, shrunk_second - shrunk_mean * shrunk_mean)
                if variance <= 0.0:
                    reason = "zero_variance"
                else:
                    score = float(shrunk_mean / math.sqrt(variance) * math.sqrt(252.0))
                    if not math.isfinite(score):
                        score = None
                        reason = "invalid_score"
            eligible = score is not None
            candidate_diagnostics[candidate_id] = {
                "eligible": eligible,
                "reason": reason,
                "sessions": int(session_count),
                "trades": int(trade_count),
                "weight": weight,
                "score": score,
            }
            if score is not None:
                eligible_candidates.append((score, candidate_id))

        if eligible_candidates:
            selected_id = min(
                eligible_candidates, key=lambda item: (-item[0], item[1])
            )[1]
            reason = "selected"
        else:
            selected_id = general_id
            reason = "fallback_no_eligible_candidate"
        mapping[regime] = selected_id
        diagnostics[regime] = {
            "selected_id": selected_id,
            "reason": reason,
            "candidates": candidate_diagnostics,
        }
    return {"mapping": mapping, "diagnostics": diagnostics}


def build_routing_schedule(
    regimes: pd.Series,
    mapping: Mapping[str, str],
    general_id: str,
    confirmation_sessions: int = 2,
) -> pd.Series:
    """Build a deterministic regime route with confirmation hysteresis."""
    if confirmation_sessions < 1:
        raise ValueError("confirmation_sessions must be positive")
    active = general_id
    pending_regime: str | None = None
    pending_count = 0
    selected: list[str] = []
    for value in regimes:
        regime = value if isinstance(value, str) and value in REGIMES else "unknown"
        if regime == "unknown":
            active = general_id
            pending_regime = None
            pending_count = 0
            selected.append(active)
            continue
        target = mapping.get(regime, general_id)
        if target == active:
            pending_regime = None
            pending_count = 0
        else:
            if pending_regime == regime:
                pending_count += 1
            else:
                pending_regime = regime
                pending_count = 1
            if pending_count >= confirmation_sessions:
                active = target
                pending_regime = None
                pending_count = 0
        selected.append(active)
    return pd.Series(selected, index=regimes.index, dtype=object, name="CandidateId")


def gate_router(
    router_returns: pd.Series,
    general_returns: pd.Series,
    router_trade_count: int,
    general_trade_count: int,
    simulations: int = 1000,
    block_length: int = 20,
    seed: int = 42,
    min_trades: int = 30,
) -> dict[str, Any]:
    """Gate a router with a paired moving-block bootstrap and drawdown check."""
    base = {
        "accepted": False,
        "reason": "",
        "ci_lower": None,
        "ci_upper": None,
        "mean_difference": None,
        "router_max_drawdown": None,
        "general_max_drawdown": None,
        "observations": 0,
        "router_trade_count": int(router_trade_count),
        "general_trade_count": int(general_trade_count),
    }
    if simulations < 1 or block_length < 1 or min_trades < 0:
        raise ValueError("invalid router gate settings")
    if router_trade_count < min_trades or general_trade_count < min_trades:
        return {**base, "reason": "insufficient_trades"}

    router, router_error = _daily_returns(router_returns)
    general, general_error = _daily_returns(general_returns)
    if router_error is not None or general_error is not None:
        return {**base, "reason": "invalid_returns"}
    paired = pd.concat(
        [router.rename("router"), general.rename("general")], axis=1, join="inner"
    )
    if not router.index.equals(general.index):
        return {**base, "reason": "return_sessions_mismatch"}
    if len(paired) < block_length:
        return {
            **base,
            "reason": "insufficient_observations",
            "observations": len(paired),
        }
    if (paired <= -1.0).any().any():
        return {**base, "reason": "bankrupt_return", "observations": len(paired)}

    difference = (paired["router"] - paired["general"]).to_numpy(dtype=float)
    observations = len(difference)
    possible_starts = observations - block_length + 1
    blocks_per_sample = math.ceil(observations / block_length)
    generator = np.random.default_rng(seed)
    means = np.empty(simulations, dtype=float)
    for simulation in range(simulations):
        starts = generator.integers(0, possible_starts, size=blocks_per_sample)
        sampled = np.concatenate(
            [difference[start : start + block_length] for start in starts]
        )[:observations]
        means[simulation] = sampled.mean()

    ci_lower, ci_upper = np.percentile(means, [2.5, 97.5])
    router_drawdown = _drawdown_from_returns(paired["router"])
    general_drawdown = _drawdown_from_returns(paired["general"])
    result = {
        **base,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "mean_difference": float(difference.mean()),
        "router_max_drawdown": router_drawdown,
        "general_max_drawdown": general_drawdown,
        "observations": observations,
    }
    if ci_lower <= 0.0:
        return {**result, "reason": "bootstrap_improvement_not_positive"}
    if router_drawdown < general_drawdown - 1e-12:
        return {**result, "reason": "drawdown_worse"}
    return {**result, "accepted": True, "reason": "accepted"}
