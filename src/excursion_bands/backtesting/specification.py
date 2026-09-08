"""
Typed configuration objects for the backtesting package.
"""

from dataclasses import dataclass, field
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


def default_search_space() -> dict[str, dict]:
    return {
        "strategy.donchian.lookback_bars": {
            "type": "int",
            "low": 10,
            "high": 160,
            "step": 5,
        },
        "features.bands.lookback_window": {
            "type": "int",
            "low": 10,
            "high": 120,
            "step": 5,
        },
        "strategy.atr_stop.multiplier": {
            "type": "float",
            "low": 0.75,
            "high": 4.0,
            "step": 0.25,
        },
        "strategy.take_profit.rr": {
            "type": "float",
            "low": 1.0,
            "high": 6.0,
            "step": 0.25,
        },
        "strategy.break_even.enabled": {
            "type": "categorical",
            "choices": [False, True],
        },
    }


@dataclass(frozen=True)
class RobustWfoConfig:
    search_months: int = 24
    search_blocks: int = 4
    calibration_months: int = 12
    gate_months: int = 12
    shortlist_size: int = 5
    min_total_trades: int = 60
    min_block_trades: int = 10
    min_regime_sessions: int = 60
    min_regime_trades: int = 20
    shrinkage_sessions: int = 126
    confirmation_sessions: int = 2
    gate_min_trades: int = 30
    bootstrap_simulations: int = 1000
    bootstrap_block_sessions: int = 20
    workers: int = 2
    stress_cost_multiplier: float = 2.0
    include_legacy: bool = True
    search_space: dict[str, dict] = field(default_factory=default_search_space)

    def __post_init__(self):
        for name in (
            "search_months",
            "search_blocks",
            "calibration_months",
            "gate_months",
            "shortlist_size",
            "shrinkage_sessions",
            "confirmation_sessions",
            "bootstrap_simulations",
            "bootstrap_block_sessions",
            "workers",
            "min_regime_sessions",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"wfo.robust.{name} must be positive")
        if self.search_months % self.search_blocks:
            raise ValueError("search_months must be divisible by search_blocks")
        for name in (
            "min_total_trades",
            "min_block_trades",
            "min_regime_sessions",
            "min_regime_trades",
            "gate_min_trades",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"wfo.robust.{name} cannot be negative")
        if not 1 <= self.stress_cost_multiplier < float("inf"):
            raise ValueError("stress_cost_multiplier must be at least one")
        if type(self.include_legacy) is not bool:
            raise ValueError("include_legacy must be a boolean")
        allowed = default_search_space()
        if not self.search_space or set(self.search_space) - set(allowed):
            raise ValueError(
                "robust search_space must contain supported strategy/band parameters"
            )
        for key, spec in self.search_space.items():
            kind = spec.get("type")
            if kind != allowed[key]["type"]:
                raise ValueError(f"Incorrect search type for {key}")
            if kind == "categorical":
                choices = spec.get("choices", [])
                if not choices or any(type(value) is not bool for value in choices):
                    raise ValueError(f"{key} requires boolean choices")
            else:
                low, high, step = (spec.get(name) for name in ("low", "high", "step"))
                if any(
                    not isinstance(value, (int, float)) for value in (low, high, step)
                ):
                    raise ValueError(f"{key} requires numeric low/high/step")
                if not (
                    0 < low <= high and 0 < step < float("inf") and high < float("inf")
                ):
                    raise ValueError(f"Invalid bounds for {key}")
                if kind == "int" and any(
                    type(value) is not int for value in (low, high, step)
                ):
                    raise ValueError(f"{key} requires integer bounds and step")


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
    optimizer: str = "grid"
    n_trials: int = 64
    seed: int = 42
    protocol: str = "legacy"
    checkpoint_dir: str | None = None
    resume: bool = False
    max_run_seconds: float | None = None
    robust: RobustWfoConfig = field(default_factory=RobustWfoConfig)


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
