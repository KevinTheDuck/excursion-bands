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
