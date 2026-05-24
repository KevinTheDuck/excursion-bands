
"""
[backtesting/specification.py]
Specification module to define dataclass used on backtesting
"""

from dataclasses import dataclass
from datetime import date

@dataclass
class VariantConfig:
    label: str
    use_band_filter: bool
    side_mode: bool
    description: str | None = None

@dataclass
class CoreDataConfig:
    data_config: str
    sessions_config: str
    volatility_config: str
    bands_config: str
    variants: str

@dataclass
class BacktestSettingConfig:
    start_date: date
    initial_cash: float

@dataclass
class BacktestConfig:
    core_data: CoreDataConfig
    backtest: BacktestSettingConfig
    variants: tuple[VariantConfig, ...]
