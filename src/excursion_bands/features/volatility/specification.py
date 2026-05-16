from dataclasses import dataclass
from typing import Literal

import polars as pl

"""
[volatility/specificaton.py]
A specification for volatility estimators, values loaded from config files through loader
"""

@dataclass(frozen=True)
class VolatilitySpec:
    open: str
    high_cols: tuple[str, ...]
    low_cols: tuple[str, ...]
    close: str
    prev_close: str
    output_col: str
