# Excursion Bands

## Setup

Requirements:
- Python 3.12+
- `uv`

Install dependencies:

```bash
uv sync
```

## Data

Local market data is expected under `data/` 

Main configured raw input:

```text
data/raw/nq_1m.parquet
```

Processed/cache files are created under `data/processed/` when needed.

## Run The Data Pipeline

```bash
uv run python notebooks/data_pipeline.py
```

Full excursion bands feature pipeline:

```bash
uv run python notebooks/excursion_bands_pipeline.py
```

## Run Backtests

Donchian strategy:

```bash
uv run python notebooks/strategy_donchian_backtesting.py
```

ORB strategy:

```bash
uv run python notebooks/strategy_orb_backtesting.py
```

Outputs are written to:

```text
data/output/backtests/
```

## Main Configs

- Donchian backtest: `configs/strategies/donchian_1/backtest_default.yaml`
- ORB backtest: `configs/strategies/strategy_1/backtest_default.yaml`
- Data config: `configs/data/local_nq_5m.yaml`
- Session config: `configs/sessions/nq_default.yaml`
- Volatility config: `configs/features/volatility/specification.yaml`
- Band config: `configs/features/bands/nq_default.yaml`

## Notes

- Use `uv run ...` for scripts so `src/` imports resolve correctly.
- WFO, HMM filtering, Monte Carlo, and research plots are controlled from the strategy YAML configs.
- WFO supports a bounded, reproducible Optuna search (`optimizer: optuna`, `n_trials`, and `seed`)
  as well as the legacy full grid (`optimizer: grid`). OOS capital is carried continuously across
  non-overlapping test folds, and the stitched WFO report is the account-level performance view.

Run the regression checks with:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

WFO accounting and execution assumptions:

- Each variant has its own continuous OOS account. Positions close at fold end;
  cash carries forward. Current-equity sizing uses that balance. Initial-equity
  sizing continues to use the original configured capital.
- Overall return and drawdown come from the stitched OOS equity curve. Drawdown
  includes opening capital and marks open positions at bar closes; it does not
  measure unobserved intrabar equity extremes. Fold percentage returns are not summed.
- Signals enter at the next bar open, within the same session and before the
  scheduled cutoff. Stop gaps fill at the open; target gaps conservatively fill
  at the target. Opening fills precede ambiguous intrabar stop/target touches.
  Break-even triggers update stops for the following bar. Cutoff exits fill at
  the cutoff bar open, with the configured slippage and commissions.
- AE-to-FE remains the entry zone. FE is not an automatic exit.
- HMM is a second-stage filter after parameter selection. It now models the
  sequence of first strategy candidates per session, with forward state filtering
  and a chronological holdout. This changes the old every-bar HMM model; historical
  HMM results should be regenerated. Insufficient fitting samples use the configured fallback.
- Searches run sequentially for reproducibility (`max_workers` is retained for
  configuration compatibility). Repeated Optuna proposals reuse trial results;
  full-history frame retention is bounded. Monte Carlo summaries cover all
  simulations; WFO path CSVs retain up to `monte_carlo_max_paths_plotted` per method.
- Derived cache fingerprints include source files, input values, and relevant
  configuration so corrected prices and changed session/band settings rebuild caches.

## Robust optimizer and resumable backtests

Start the new raw/bands Donchian research workflow from the repository root:

```bash
uv run python -m excursion_bands.backtesting.cli --run-dir data/output/backtests/robust_nq
```

After Ctrl+C, a time-budget pause, or a restart, continue the **same run** with:

```bash
uv run python -m excursion_bands.backtesting.cli --run-dir data/output/backtests/robust_nq --resume
```

The default profile is `configs/strategies/donchian_1/backtest_robust.yaml`.
It allows eight hours per invocation, then pauses at a safe boundary; eight hours
is a budget, not a promise that the entire history will finish. `--max-hours 2`
changes the invocation budget; `--workers 1` lowers memory/concurrency. A new
experiment can use `--trials 8` for a smaller search, but the trial budget cannot
change when resuming. Use a different run directory for changed research settings.

The existing WFO also supports checkpoints without changing its selection method:

```bash
uv run python -m excursion_bands.backtesting.cli --config configs/strategies/donchian_1/backtest_default.yaml --run-dir data/output/backtests/legacy_nq
uv run python -m excursion_bands.backtesting.cli --config configs/strategies/donchian_1/backtest_default.yaml --run-dir data/output/backtests/legacy_nq --resume
```

### What the robust profile does

- Searches numeric parameter ranges instead of constructing a Cartesian grid.
  Each variant's Optuna sequence is seeded and sequential; up to two independent
  variants run concurrently. Position risk and AE–FE entry semantics stay fixed.
- Uses 24 months for candidate search (four six-month scoring blocks), then
  12 months for regime calibration and 12 months for router validation, followed
  by a complete three-month OOS test. Earlier history supplies causal indicator
  warmup. Insufficient starting history and incomplete final test windows are omitted.
- Scores net session returns, including no-trade sessions, as median block Sharpe
  minus half its interquartile range minus daily-equity drawdown magnitude.
  Trade-count requirements and nearby-parameter checks reject unsupported peaks.
- Builds low/normal/high volatility states from the preceding 20 completed-session
  log returns. Thresholds and library rankings are frozen before the outer test;
  sample requirements, shrinkage, and two-session confirmation limit switching.
- Replays the router with real capital and sizing on a later validation window.
  Adaptation needs positive paired bootstrap evidence and no worse drawdown;
  otherwise the general configuration remains active. No eligible candidate means cash.
- Reports general/regime policies, doubled-cost stress runs without retuning,
  and matched-parameter no-band ablations for the band model. The old selector
  runs separately in `legacy_comparison/` on aligned complete test windows.
  HMM is excluded from the robust profile; the existing HMM workflow is unchanged.

### Checkpoints and outputs

Robust runs checkpoint completed candidate evaluations, optimizer/sampler state,
validation sessions, and OOS sessions. Legacy runs checkpoint completed trials
and folds. Work interrupted *inside* one uncommitted operation may repeat; already
committed capital, trades, and folds are not duplicated.

`manifest.json` is the authoritative completion marker. Paused runs are not
published as completed backtests. Atomic, checksummed checkpoints retain the
latest two state generations, with immutable result artifacts. Do not delete
`artifacts/`, `checkpoints/`, `manifests/`, or the per-variant run directories.
Data/configuration/source changes are rejected on resume, except operational
options such as resume, runtime budget, output location, and worker count.
Checkpoint blobs contain pickled Optuna state: only resume your own trusted runs.

A live run lock prevents two processes from resuming the same run. Ctrl+C releases
locks. An uncatchable termination (power failure or `kill -9`) can leave a
`.checkpoint.lock`; the error identifies its owner. Verify that the owner has
exited before moving that specific stale lock aside, then resume. Never remove
a live lock. Robust runs have both a coordinator lock and per-variant locks.

Completed robust reports include `wfo_report.md`, `wfo_oos_metrics.csv`, stitched
equity/trade/yearly CSVs, `session_decisions.csv`, `wfo_train_trials.csv`, and
`parameter_libraries.json`. `optimizer_trials.csv` lists every Optuna proposal,
including duplicates; `wfo_test_folds.csv` provides fold-level policy results.
The main metrics table includes the aligned legacy comparator with the same
daily Sharpe convention. The library includes calibration evidence, rejected
gates, neighborhood scores, and candidate evaluations. Daily Sharpe/Sortino use
252 sessions/year; reported drawdown uses close-marked bar equity, not intrabar
extremes. Already-inspected history remains development evidence, not an untouched
holdout. A simpler model winning is a valid result; improved returns are not guaranteed.
