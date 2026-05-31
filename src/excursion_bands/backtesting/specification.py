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
    use_hmm_filter: bool = False
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
class WfoResearchConfig:
    enabled: bool = False
    monte_carlo_enabled: bool = False
    monte_carlo_methods: tuple[str, ...] = ("bootstrap", "reshuffle", "dropout")
    monte_carlo_simulations: int = 1000
    monte_carlo_sample_trades: int = 0
    monte_carlo_dropout_pct: float = 0.1
    monte_carlo_max_paths_plotted: int = 100
    monte_carlo_seed: int = 42
    parameter_stability_enabled: bool = True
    comparative_charts_enabled: bool = True


@dataclass(frozen=True)
class ReportConfig:
    output_dir: str
    save_charts: bool
    monte_carlo: MonteCarloConfig
    wfo_research: WfoResearchConfig


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
    reoptimization_mode: str = "always"
    degradation_objective: str = "total_return_pct"
    degradation_threshold: float = 0.0
    parameter_grid: dict[str, list] | None = None


@dataclass(frozen=True)
class HMMConfig:
    enabled: bool
    n_states: int
    max_iter: int
    random_state: int
    min_train_samples: int
    min_state_trades: int
    min_state_net_r: float
    min_state_win_rate: float
    top_states: int
    validation_fraction: float
    min_validation_trades: int
    min_validation_net_r_improvement: float
    min_validation_allow_rate: float
    warmup_start_date: date | None
    train_lookback_months: int | None
    fallback: str


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
class DonchianConfig:
    lookback_bars: int


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    force_exit_time: str
    atr: OrbAtrConfig
    stop: OrbStopConfig
    atr_stop: OrbAtrStopConfig
    take_profit: OrbTakeProfitConfig
    break_even: BreakEvenConfig
    opening_range: OpeningRangeConfig | None = None
    entry_bars_after_or: int = 0
    donchian: DonchianConfig | None = None


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
    hmm: HMMConfig | None = None
    strategy: StrategyConfig | None = None
    variants: tuple[VariantConfig, ...] = ()
