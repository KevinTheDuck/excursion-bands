"""
Typed configuration objects for the backtesting package.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class VariantConfig:
    label: str
    use_band_filter: bool = False
    side_mode: str = "both"
    use_atr_buffer: bool = False
    use_vwap_filter: bool = False
    use_ml_filter: bool = False
    ml_execution_mode: str = "filter"
    ml_candidate_scope: str = "variant"
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
    mode: str = "calendar"
    train_months: int = 6
    test_months: int = 3
    step_months: int = 3
    train_sessions: int = 504
    test_sessions: int = 63
    step_sessions: int = 63
    objective: str = "sharpe"
    parameter_grid: dict[str, list] | None = None


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    random_state: int


@dataclass(frozen=True)
class MLConfig:
    enabled: bool
    probability_threshold: float
    min_train_samples: int
    warmup_start_date: date | None
    train_lookback_months: int | None
    refit_frequency_sessions: int
    fallback: str
    xgboost: XGBoostConfig


@dataclass(frozen=True)
class OpeningRangeConfig:
    start: str
    end: str


@dataclass(frozen=True)
class OrbAtrConfig:
    lookback_sessions: int
    buffer_mult: float


@dataclass(frozen=True)
class OrbStopConfig:
    mode: str


@dataclass(frozen=True)
class OrbAtrStopConfig:
    enabled: bool
    length: int
    multiplier: float


@dataclass(frozen=True)
class OrbTakeProfitConfig:
    rr: float


@dataclass(frozen=True)
class BreakEvenConfig:
    enabled: bool
    trigger_rr: float
    offset_points: float


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    opening_range: OpeningRangeConfig
    entry_bars_after_or: int
    force_exit_time: str
    atr: OrbAtrConfig
    stop: OrbStopConfig
    atr_stop: OrbAtrStopConfig
    take_profit: OrbTakeProfitConfig
    break_even: BreakEvenConfig


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
    ml: MLConfig | None = None
    strategy: StrategyConfig | None = None
    variants: tuple[VariantConfig, ...] = ()
