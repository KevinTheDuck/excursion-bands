
"""
[backtesting/specification.py]
Specification module to define dataclass used on backtesting
"""

from dataclasses import dataclass

@dataclass
class CoreDataConfig:
    data_config: str
    sessions_config: str
    volatility_config: str
    bands_config: str


@dataclass
class BacktestConfig:
    core_data: CoreDataConfig
