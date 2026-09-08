"""Chronological, resumable robust WFO. No outer-test result trains its router."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Event
from time import monotonic

import numpy as np
import optuna
import pandas as pd

from excursion_bands.backtesting.checkpoints import (
    CheckpointStore,
    backtest_fingerprint,
)
from excursion_bands.backtesting.metrics import (
    calculate_metrics,
    calculate_yearly_metrics,
)
from excursion_bands.backtesting.robust_execution import (
    ParameterEvaluator,
    session_returns,
)
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
from excursion_bands.backtesting.specification import BacktestConfig
from excursion_bands.data import load_yaml
from excursion_bands.paths import resolve_path


def _clean(value):
    """JSON primitives only, with undefined scores represented explicitly as null."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _identity(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:20]


def _score(record):
    return record["score"] if record.get("score") is not None else -math.inf


def robust_folds(intraday, config):
    """The configured start is the first requested OOS date, not training start."""
    settings = config.wfo.robust
    if config.wfo.test_months <= 0:
        raise ValueError("test_months must be positive")
    sessions = sorted(pd.to_datetime(intraday.Session).dt.date.unique())
    if not sessions:
        raise ValueError("No sessions available")
    end = min(sessions[-1], config.backtest.end_date or sessions[-1])
    cursor = pd.Timestamp(config.backtest.start_date)
    folds = []
    while cursor.date() <= end:
        gate_start = cursor - pd.DateOffset(months=settings.gate_months)
        calibration_start = gate_start - pd.DateOffset(
            months=settings.calibration_months
        )
        search_start = calibration_start - pd.DateOffset(months=settings.search_months)
        next_cursor = cursor + pd.DateOffset(months=config.wfo.test_months)
        if next_cursor > pd.Timestamp(end) + pd.Timedelta(days=1):
            break

        def between(start, stop):
            return [s for s in sessions if start.date() <= s < stop.date() and s <= end]

        search = between(search_start, calibration_start)
        calibration = between(calibration_start, gate_start)
        gate = between(gate_start, cursor)
        test = between(cursor, next_cursor)
        if sessions[0] <= search_start.date() and all(
            (search, calibration, gate, test)
        ):
            width = settings.search_months // settings.search_blocks
            blocks = [
                between(
                    search_start + pd.DateOffset(months=i * width),
                    search_start + pd.DateOffset(months=(i + 1) * width),
                )
                for i in range(settings.search_blocks)
            ]
            folds.append(
                {
                    "fold": len(folds) + 1,
                    "search": search,
                    "calibration": calibration,
                    "gate": gate,
                    "test": test,
                    "blocks": blocks,
                    "available_at": cursor.date().isoformat(),
                }
            )
        cursor = next_cursor
    if not folds:
        raise ValueError(
            "No complete robust training windows; supply more history or later OOS dates"
        )
    return folds


def _fingerprint(config, intraday, bands):
    return backtest_fingerprint(config, intraday, bands)


class _BudgetExpired(Exception):
    pass


class _ResearchRun:
    def __init__(
        self, store, config, variant, intraday, bands, folds, deadline, stopped
    ):
        self.store, self.config, self.variant = store, config, variant
        self.intraday, self.folds = intraday, folds
        self.deadline, self.stopped = deadline, stopped
        self.settings = config.wfo.robust
        self.evaluator = ParameterEvaluator(intraday, bands, config, variant)
        self.volatility = session_volatility(intraday)
        self.state = store.load() or {
            "status": "running",
            "fold_index": 0,
            "folds": [],
            "streams": {},
            "artifacts": [],
            "decisions": [],
        }

    def commit(self):
        self.state = _clean(self.state)
        self.store.save(self.state)
        if self.stopped.is_set() or monotonic() >= self.deadline:
            raise _BudgetExpired()

    def evaluate(self, params, sessions, phase):
        """Evaluation outputs are immutable and committed before moving on."""
        current = self.state["current"]
        key = f"{phase}_{_identity(params)}"
        if key not in current["evaluations"]:
            result = self.evaluator.run(params, sessions)
            returns = session_returns(
                result, self.intraday, sessions, self.config.backtest.initial_cash
            )
            prefix = f"fold{self.state['fold_index']}_{key}"
            self.store.write_frame(
                prefix + "_returns", returns.rename("Return").to_frame()
            )
            self.store.write_frame(prefix + "_trades", result.trades)
            record = {
                "params": params,
                "returns": prefix + "_returns",
                "trades": prefix + "_trades",
            }
            if phase == "search":
                record.update(
                    score_candidate(
                        returns,
                        result.trades,
                        self.folds[self.state["fold_index"]]["blocks"],
                        self.settings.min_total_trades,
                        self.settings.min_block_trades,
                    )
                )
            current["evaluations"][key] = _clean(record)
            self.commit()
        # commit normalizes the state object; never retain mutable nested references across it.
        record = self.state["current"]["evaluations"][key]
        returns = self.store.read_frame(record["returns"])["Return"]
        returns.index = pd.to_datetime(returns.index).date
        return record, returns, self.store.read_frame(record["trades"])

    def discover(self, fold):
        space = {
            k: v
            for k, v in self.settings.search_space.items()
            if self.variant.use_band_filter or not k.startswith("features.bands.")
        }
        if "study" in self.state["current"]:
            # Only load a checksum-verified checkpoint produced by this trusted local run.
            study = pickle.loads(self.store.load_blob(self.state["current"]["study"]))
        else:
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.config.wfo.seed, n_startup_trials=16
                ),
            )
            initial = {}
            for key, spec in space.items():
                if key.startswith("features.bands."):
                    path, _ = resolve_path(self.config.core_data.bands_config)
                    value = load_yaml(path)["lookback_window"]
                else:
                    value = self.config
                    for part in key.split("."):
                        value = getattr(value, part)
                if spec["type"] == "categorical":
                    valid = value in spec["choices"]
                else:
                    valid = spec["low"] <= value <= spec["high"] and math.isclose(
                        (value - spec["low"]) / spec["step"],
                        round((value - spec["low"]) / spec["step"]),
                    )
                if valid:
                    initial[key] = value
            study.enqueue_trial(initial)
        while (
            len([t for t in study.trials if t.state.is_finished()])
            < self.config.wfo.n_trials
        ):
            trial = study.ask()
            params = {}
            for key, spec in space.items():
                kind = spec["type"]
                if kind == "categorical":
                    params[key] = trial.suggest_categorical(key, spec["choices"])
                elif kind == "int":
                    params[key] = trial.suggest_int(
                        key, spec["low"], spec["high"], step=spec["step"]
                    )
                else:
                    params[key] = trial.suggest_float(
                        key, spec["low"], spec["high"], step=spec["step"]
                    )
            record, _, _ = self.evaluate(params, fold["search"], "search")
            study.tell(trial, _score(record))
            study_name = f"study_{self.state['fold_index']}"
            self.store.save_blob(
                study_name, pickle.dumps(study, protocol=pickle.HIGHEST_PROTOCOL)
            )
            self.state["current"]["study"] = study_name
            self.state["current"]["trials"] = len(
                [t for t in study.trials if t.state.is_finished()]
            )
            self.commit()
        current = self.state["current"]
        if "shortlist" not in current:
            # Only sampled candidates, not later neighborhood probes, become seed finalists.
            sampled = {
                _identity(t.params): t.params
                for t in study.trials
                if t.state.is_finished()
            }
            eligible = [
                current["evaluations"]["search_" + key]
                for key in sampled
                if current["evaluations"]["search_" + key].get("eligible")
            ]
            eligible.sort(key=lambda x: (-_score(x), _identity(x["params"])))
            current["shortlist"] = [
                x["params"] for x in eligible[: self.settings.shortlist_size]
            ]
            self.commit()
        ranked = []
        for params in self.state["current"]["shortlist"]:
            neighborhood = [params, *numeric_neighbors(params, space)]
            scores = [
                _score(self.evaluate(p, fold["search"], "search")[0])
                for p in neighborhood
            ]
            value = float(np.median(scores))
            if math.isfinite(value):
                ranked.append((value, _identity(params), params))
        ranked.sort(key=lambda x: (-x[0], x[1]))
        return ranked

    def select(self, fold):
        if "plan" in self.state["current"]:
            return self.state["current"]["plan"]
        ranked = self.discover(fold)
        if not ranked:
            plan = {
                "general": "cash",
                "params": {"cash": {}},
                "mapping": {},
                "gate": {"accepted": False, "reason": "no_eligible_candidates"},
                "thresholds": None,
            }
        else:
            general = ranked[0][1]
            params = {key: p for _, key, p in ranked}
            thresholds = fit_regime_thresholds(self.volatility.reindex(fold["search"]))
            calibration_regimes = classify_regimes(
                self.volatility.reindex(fold["calibration"]), thresholds
            )
            returns, trades = {}, {}
            for key, candidate in params.items():
                _, returns[key], trades[key] = self.evaluate(
                    candidate, fold["calibration"], "calibration"
                )
            library = calibrate_library(
                returns,
                trades,
                calibration_regimes,
                general,
                self.settings.min_regime_sessions,
                self.settings.min_regime_trades,
                self.settings.shrinkage_sessions,
            )
            regimes = classify_regimes(
                self.volatility.reindex(fold["gate"]), thresholds
            )
            schedule = build_routing_schedule(
                regimes,
                library["mapping"],
                general,
                self.settings.confirmation_sessions,
            )
            # Replay real capital/sizing, not a splice of independently compounded returns.
            gate_series, counts = self.replay_gate(fold, params, general, schedule)
            gate = gate_router(
                gate_series["router"],
                gate_series["general"],
                counts["router"],
                counts["general"],
                self.settings.bootstrap_simulations,
                self.settings.bootstrap_block_sessions,
                self.config.wfo.seed,
                self.settings.gate_min_trades,
            )
            bar_drawdowns = self.state["current"]["gate_replay"]["drawdowns"]
            gate["router_bar_drawdown"] = bar_drawdowns["router"]
            gate["general_bar_drawdown"] = bar_drawdowns["general"]
            if (
                gate["accepted"]
                and bar_drawdowns["router"] < bar_drawdowns["general"] - 1e-12
            ):
                gate.update(accepted=False, reason="worse_close_marked_drawdown")
            plan = {
                "general": general,
                "params": params,
                "thresholds": thresholds,
                "mapping": library["mapping"],
                "calibration": library["diagnostics"],
                "gate": gate,
                "neighborhood_scores": {key: score for score, key, _ in ranked},
            }
        self.state["current"]["plan"] = _clean(plan)
        self.commit()
        return self.state["current"]["plan"]

    def replay_gate(self, fold, params, general, schedule):
        current = self.state["current"]
        if "gate_replay" not in current:
            current["gate_replay"] = {
                "index": 0,
                "cash": {
                    k: self.config.backtest.initial_cash for k in ("router", "general")
                },
                "returns": {k: [] for k in ("router", "general")},
                "counts": {k: 0 for k in ("router", "general")},
                "peaks": {
                    k: self.config.backtest.initial_cash for k in ("router", "general")
                },
                "drawdowns": {k: 0.0 for k in ("router", "general")},
            }
        for session in fold["gate"][current["gate_replay"]["index"] :]:
            replay = self.state["current"]["gate_replay"]
            for kind, key in (("router", schedule.loc[session]), ("general", general)):
                cash = replay["cash"][kind]
                result = self.evaluator.run(params[key], [session], cash)
                closing = float(result.equity_curve.Equity.iloc[-1])
                values = np.r_[
                    replay["peaks"][kind], result.equity_curve.Equity.to_numpy()
                ]
                peaks = np.maximum.accumulate(values)
                replay["peaks"][kind] = float(peaks[-1])
                replay["drawdowns"][kind] = min(
                    replay["drawdowns"][kind], float(np.min(values / peaks - 1))
                )
                replay["returns"][kind].append(closing / cash - 1 if cash > 0 else 0.0)
                replay["cash"][kind] = closing
                replay["counts"][kind] += len(result.trades)
            replay["index"] += 1
            self.commit()
        replay = self.state["current"]["gate_replay"]
        return {
            k: pd.Series(v, index=fold["gate"]) for k, v in replay["returns"].items()
        }, replay["counts"]

    def execute_oos(self, fold, plan):
        regimes = classify_regimes(
            self.volatility.reindex(fold["test"]), plan["thresholds"]
        )
        schedule = build_routing_schedule(
            regimes,
            plan["mapping"],
            plan["general"],
            self.settings.confirmation_sessions,
        )
        policies = [
            ("general", False, 1.0),
            ("regime", False, 1.0),
            ("general_stress", False, self.settings.stress_cost_multiplier),
            ("regime_stress", False, self.settings.stress_cost_multiplier),
        ]
        if self.variant.use_band_filter:
            policies += [
                ("general_no_bands", True, 1.0),
                ("regime_no_bands", True, 1.0),
            ]
        for session in fold["test"][self.state["current"].get("oos_index", 0) :]:
            equity_parts, trade_parts = [], []
            for policy, remove_bands, costs in policies:
                label = self.variant.label + "_" + policy
                routed = policy.startswith("regime") and plan["gate"]["accepted"]
                key = schedule.loc[session] if routed else plan["general"]
                cash = self.state["streams"].get(
                    label, self.config.backtest.initial_cash
                )
                variant = (
                    replace(self.variant, use_band_filter=False)
                    if remove_bands
                    else self.variant
                )
                if key == "cash":
                    # No selection evidence: explicitly do not trade.
                    bars = self.intraday[self.intraday.Session == session]
                    equity = pd.DataFrame(
                        {"DateTime": bars.DateTime, "Equity": cash, "InPosition": 0}
                    )
                    trades = pd.DataFrame()
                else:
                    result = self.evaluator.run(
                        plan["params"][key],
                        [session],
                        cash,
                        variant=variant,
                        cost_multiplier=costs,
                    )
                    equity, trades = result.equity_curve.copy(), result.trades.copy()
                closing = float(equity.Equity.iloc[-1])
                self.state["streams"][label] = closing
                equity["Variant"], equity["Session"], equity["Fold"] = (
                    label,
                    session,
                    fold["fold"],
                )
                equity_parts.append(equity)
                if not trades.empty:
                    trades["Variant"], trades["Fold"], trades["ParameterId"] = (
                        label,
                        fold["fold"],
                        key,
                    )
                    trade_parts.append(trades)
                self.state["decisions"].append(
                    {
                        "variant": label,
                        "session": session,
                        "fold": fold["fold"],
                        "regime": regimes.loc[session],
                        "parameter_id": key,
                        "routed": bool(routed),
                        "gate_reason": plan["gate"]["reason"],
                        "starting_cash": cash,
                        "ending_cash": closing,
                        "daily_return": closing / cash - 1 if cash > 0 else 0.0,
                    }
                )
            name = f"oos_{fold['fold']}_{session}"
            self.store.write_frame(
                name + "_equity", pd.concat(equity_parts, ignore_index=True)
            )
            self.store.write_frame(
                name + "_trades",
                pd.concat(trade_parts, ignore_index=True)
                if trade_parts
                else pd.DataFrame(),
            )
            self.state["artifacts"].append(name)
            self.state["current"]["oos_index"] = (
                self.state["current"].get("oos_index", 0) + 1
            )
            self.commit()

    def run(self):
        if self.store.completed:
            return True
        try:
            self.commit()
            while self.state["fold_index"] < len(self.folds):
                fold = self.folds[self.state["fold_index"]]
                self.state.setdefault("current", {"evaluations": {}})
                print(
                    f"Robust WFO {self.variant.label} fold {fold['fold']}/{len(self.folds)}",
                    flush=True,
                )
                plan = self.select(fold)
                self.execute_oos(fold, plan)
                self.state["folds"].append(
                    {
                        "fold": fold["fold"],
                        "available_at": fold["available_at"],
                        "plan": plan,
                        "evaluations": self.state["current"]["evaluations"],
                        "trials": self.state["current"].get("trials", 0),
                    }
                )
                del self.state["current"]
                self.state["fold_index"] += 1
                self.commit()
            self.state["status"] = "complete"
            self.store.mark_complete(_clean(self.state))
            return True
        except _BudgetExpired:
            self.state["status"] = "paused"
            self.store.save(_clean(self.state))
            return False


def _daily_summary(equity, trades, daily, initial_cash):
    summary = calculate_metrics(equity, trades, initial_cash)
    summary["bar_sharpe"] = summary.pop("sharpe")
    summary["sharpe"] = (
        float(daily.mean() / daily.std(ddof=0) * np.sqrt(252))
        if daily.std(ddof=0) > 0
        else None
    )
    downside = np.sqrt(np.mean(np.minimum(daily, 0) ** 2))
    summary["sortino"] = (
        float(daily.mean() / downside * np.sqrt(252)) if downside > 0 else None
    )
    return summary


def _write_reports(run_dir, states, config):
    metrics, decisions, trials, plans, optimizer_trials, fold_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    reference_sessions = None
    for store, state in states:
        decisions.extend(state["decisions"])
        equity_parts = [
            store.read_frame(name + "_equity") for name in state["artifacts"]
        ]
        trade_parts = [
            store.read_frame(name + "_trades") for name in state["artifacts"]
        ]
        if not equity_parts:
            continue
        all_equity = pd.concat(equity_parts, ignore_index=True)
        if reference_sessions is None:
            reference_sessions = all_equity[["DateTime", "Session"]].drop_duplicates(
                "DateTime"
            )
            reference_sessions["DateTime"] = pd.to_datetime(
                reference_sessions.DateTime, utc=True
            )
        all_trades = (
            pd.concat([t for t in trade_parts if not t.empty], ignore_index=True)
            if any(not t.empty for t in trade_parts)
            else pd.DataFrame()
        )
        for label, equity in all_equity.groupby("Variant"):
            trades = (
                all_trades[all_trades.Variant == label]
                if not all_trades.empty
                else pd.DataFrame()
            )
            equity = equity.sort_values("DateTime")
            daily = pd.DataFrame(
                [d for d in state["decisions"] if d["variant"] == label]
            ).daily_return
            summary = _daily_summary(
                equity, trades, daily, config.backtest.initial_cash
            )
            summary["variant"] = label
            metrics.append(summary)
            for fold_number, fold_equity in equity.groupby("Fold", sort=True):
                rows = [
                    d
                    for d in state["decisions"]
                    if d["variant"] == label and d["fold"] == fold_number
                ]
                fold_trades = (
                    trades[trades.Fold == fold_number] if not trades.empty else trades
                )
                fold_summary = _daily_summary(
                    fold_equity,
                    fold_trades,
                    pd.Series([d["daily_return"] for d in rows]),
                    rows[0]["starting_cash"],
                )
                fold_rows.append(
                    dict(
                        variant=label,
                        fold=fold_number,
                        test_start=rows[0]["session"],
                        test_end=rows[-1]["session"],
                        **fold_summary,
                    )
                )
            equity.to_csv(run_dir / f"wfo_oos_equity_{label}.csv", index=False)
            trades.to_csv(run_dir / f"wfo_oos_trades_{label}.csv", index=False)
            calculate_yearly_metrics(
                equity, trades, config.backtest.initial_cash
            ).to_csv(run_dir / f"wfo_oos_yearly_{label}.csv", index=False)
        for fold in state["folds"]:
            plans.append(dict(variant=store.run_dir.name, **fold))
            study = pickle.loads(store.load_blob(f"study_{fold['fold'] - 1}"))
            for trial in study.trials:
                optimizer_trials.append(
                    dict(
                        variant=store.run_dir.name,
                        fold=fold["fold"],
                        trial_number=trial.number,
                        state=trial.state.name,
                        score=trial.value,
                        **{"param." + k: v for k, v in trial.params.items()},
                    )
                )
            for key, record in fold["evaluations"].items():
                trials.append(
                    dict(
                        variant=store.run_dir.name,
                        fold=fold["fold"],
                        evaluation=key,
                        score=record.get("score"),
                        eligible=record.get("eligible"),
                        reason=record.get("reason"),
                        **{"param." + k: v for k, v in record["params"].items()},
                    )
                )
    legacy_dir = run_dir / "legacy_comparison"
    if config.wfo.robust.include_legacy and reference_sessions is not None:
        for variant in config.variants:
            if variant.use_hmm_filter:
                continue
            equity = pd.read_csv(legacy_dir / f"wfo_oos_equity_{variant.label}.csv")
            equity["DateTime"] = pd.to_datetime(equity.DateTime, utc=True)
            # Map exact engine timestamps, never use the calendar date of overnight bars.
            equity = equity.drop(columns="Session", errors="ignore").merge(
                reference_sessions, on="DateTime", how="left", validate="many_to_one"
            )
            if equity.Session.isna().any() or set(equity.Session) != set(
                reference_sessions.Session
            ):
                raise ValueError(
                    "Legacy and robust OOS sessions differ; refusing an unaligned comparison"
                )
            try:
                trades = pd.read_csv(legacy_dir / f"wfo_oos_trades_{variant.label}.csv")
            except pd.errors.EmptyDataError:
                trades = pd.DataFrame()
            closing = (
                equity.sort_values("DateTime")
                .groupby("Session", sort=True)
                .Equity.last()
            )
            daily = (
                pd.concat(
                    [
                        pd.Series([config.backtest.initial_cash]),
                        closing.reset_index(drop=True),
                    ]
                )
                .pct_change()
                .iloc[1:]
            )
            summary = _daily_summary(
                equity, trades, daily, config.backtest.initial_cash
            )
            metrics.append(dict(variant="legacy_" + variant.label, **summary))
    pd.DataFrame(metrics).to_csv(run_dir / "wfo_oos_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "wfo_test_folds.csv", index=False)
    pd.DataFrame(optimizer_trials).to_csv(run_dir / "optimizer_trials.csv", index=False)
    pd.DataFrame(decisions).to_csv(run_dir / "session_decisions.csv", index=False)
    pd.DataFrame(trials).to_csv(run_dir / "wfo_train_trials.csv", index=False)
    (run_dir / "parameter_libraries.json").write_text(
        json.dumps(_clean(plans), indent=2), encoding="utf-8"
    )
    lines = [
        "# Robust WFO research",
        "",
        "Status: complete",
        "",
        "Sharpe and Sortino use net session returns, including flat sessions. Drawdown uses close-marked bar equity.",
        "Reviewed historical data is development evidence, not an untouched holdout.",
        "",
        "| Model | Return % | Drawdown % | Daily Sharpe | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        sharpe = "n/a" if row["sharpe"] is None else f"{row['sharpe']:.2f}"
        lines.append(
            f"| {row['variant']} | {row['total_return_pct']:.2f} | {row['max_drawdown_pct']:.2f} | {sharpe} | {row['trade_count']} |"
        )
    lines += [
        "",
        "`*_stress` doubles configured costs without retuning. `*_no_bands` removes only the band filter from the same parameter schedule.",
        "",
        "See session_decisions.csv for routing/fallbacks and parameter_libraries.json for calibration, gates, and every evaluated candidate.",
    ]
    (run_dir / "wfo_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_robust_wfo(
    intraday: pd.DataFrame, bands: pd.DataFrame, config: BacktestConfig
) -> Path:
    if config.strategy is None or config.strategy.name != "donchian":
        raise ValueError("Robust WFO v1 supports Donchian only")
    if config.wfo.n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if (
        config.wfo.mode != "calendar"
        or config.wfo.step_months != config.wfo.test_months
    ):
        raise ValueError(
            "Robust WFO requires calendar mode with step_months equal to test_months"
        )
    if config.wfo.optimizer != "optuna":
        raise ValueError("Robust WFO requires optimizer: optuna")
    if config.wfo.max_run_seconds is not None and config.wfo.max_run_seconds <= 0:
        raise ValueError("max_run_seconds must be positive")
    if config.wfo.resume and not config.wfo.checkpoint_dir:
        raise ValueError("Resume requires checkpoint_dir / --run-dir")
    variants = tuple(v for v in config.variants if not v.use_hmm_filter)
    if not variants or len({v.label for v in variants}) != len(variants):
        raise ValueError("Robust WFO requires uniquely labeled non-HMM variants")
    for variant in variants:
        if Path(variant.label).name != variant.label or variant.label in {".", ".."}:
            raise ValueError("Variant labels must be safe directory names")
    intraday = intraday.sort_values("DateTime").reset_index(drop=True).copy()
    intraday["Session"] = pd.to_datetime(intraday.Session).dt.date
    bands = bands.copy()
    bands["Session"] = pd.to_datetime(bands.Session).dt.date
    folds = robust_folds(intraday, config)
    fingerprint = _fingerprint(config, intraday, bands)
    run_dir = Path(
        config.wfo.checkpoint_dir
        or Path(config.reports.output_dir)
        / datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f_robust")
    )
    deadline = monotonic() + (config.wfo.max_run_seconds or 8 * 3600)
    stopped = Event()
    print(f"Robust run/checkpoints: {run_dir.resolve()}", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with CheckpointStore(run_dir, fingerprint, resume=config.wfo.resume) as coordinator:
        state = coordinator.load() or {
            "status": "running",
            "first_oos": str(folds[0]["test"][0]),
            "requested_start": str(config.backtest.start_date),
            "configuration": _clean(asdict(config)),
        }
        if coordinator.completed:
            return run_dir
        coordinator.save(state)

        def work(variant):
            directory = run_dir / variant.label
            resume = (directory / "manifest.json").exists()
            with CheckpointStore(
                directory, fingerprint + variant.label, resume=resume
            ) as store:
                return _ResearchRun(
                    store, config, variant, intraday, bands, folds, deadline, stopped
                ).run()

        executor = ThreadPoolExecutor(
            max_workers=min(config.wfo.robust.workers, len(variants))
        )
        try:
            futures = [executor.submit(work, variant) for variant in variants]
            outcomes = [future.result() for future in futures]
            completed = all(outcomes)
        except BaseException:
            stopped.set()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if not completed:
            state["status"] = "paused"
            coordinator.save(state)
            print(
                f"Paused safely; resume with --run-dir {run_dir} --resume", flush=True
            )
            return run_dir
        if config.wfo.robust.include_legacy:
            from excursion_bands.backtesting.wfo import run_wfo

            legacy_dir = run_dir / "legacy_comparison"
            legacy_config = replace(
                config,
                variants=variants,
                backtest=replace(
                    config.backtest,
                    start_date=(
                        pd.Timestamp(folds[0]["available_at"])
                        - pd.DateOffset(months=config.wfo.train_months)
                    ).date(),
                ),
                wfo=replace(
                    config.wfo,
                    protocol="legacy",
                    checkpoint_dir=str(legacy_dir),
                    resume=(legacy_dir / "manifest.json").exists(),
                    max_run_seconds=max(0.001, deadline - monotonic()),
                ),
            )
            run_wfo(intraday, bands, legacy_config)
            manifest = json.loads((legacy_dir / "manifest.json").read_text())
            # Legacy checkpoint store is authoritative, not the existence of CSV files.
            payload = manifest.get("payload", manifest)
            if not payload.get("complete", False):
                state["status"] = "paused_legacy"
                coordinator.save(state)
                return run_dir
        from contextlib import ExitStack

        with ExitStack() as stack:
            states = []
            for variant in variants:
                store = stack.enter_context(
                    CheckpointStore(
                        run_dir / variant.label,
                        fingerprint + variant.label,
                        resume=True,
                    )
                )
                states.append((store, store.load()))
            _write_reports(run_dir, states, config)
        state["status"] = "complete"
        coordinator.mark_complete(state)
    return run_dir
