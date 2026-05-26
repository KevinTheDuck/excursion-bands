"""
Typed configuration objects for the backtesting package.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class VariantConfig:
    label: str
    use_band_filter: bool
    side_mode: str
    description: str | None = None


@dataclass(frozen=True)
class CoreDataConfig:
    data_config: str
    sessions_config: str
    volatility_config: str
    bands_config: str
    variants: str | None = None


@dataclass(frozen=True)
class InstrumentConfig:
    symbol: str
    point_value: float
    tick_size: float
    cfd_point_value: float


@dataclass(frozen=True)
class ExecutionConfig:
    commission_per_contract_side: float
    slippage_ticks_per_side: float
    fill_on: str
    same_bar_exit_priority: str


@dataclass(frozen=True)
class SizingConfig:
    mode: str
    fixed_units: float
    risk_pct: float
    max_units: float | None = None


@dataclass(frozen=True)
class RiskConfig:
    stop_loss_points: float | None
    take_profit_points: float | None


@dataclass(frozen=True)
class BenchmarkConfig:
    enabled: bool
    strategy: str


@dataclass(frozen=True)
class MonteCarloConfig:
    enabled: bool
    simulations: int
    sample_trades: int
    max_paths_plotted: int
    seed: int


@dataclass(frozen=True)
class ReportConfig:
    output_dir: str
    save_charts: bool
    monte_carlo: MonteCarloConfig


@dataclass(frozen=True)
class WfoConfig:
    enabled: bool
    max_parameter_combinations: int
    max_workers: int


@dataclass(frozen=True)
class BacktestSettingConfig:
    start_date: date
    initial_cash: float
    end_date: date | None = None


@dataclass(frozen=True)
class BacktestConfig:
    core_data: CoreDataConfig
    backtest: BacktestSettingConfig
    instrument: InstrumentConfig
    execution: ExecutionConfig
    sizing: SizingConfig
    risk: RiskConfig
    benchmark: BenchmarkConfig
    reports: ReportConfig
    wfo: WfoConfig
    variants: tuple[VariantConfig, ...] = ()
