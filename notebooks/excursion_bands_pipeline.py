from typing import Literal
import polars as pl

from excursion_bands.features.volatility.loader import load_volatility_spec, VolatilitySpec
from excursion_bands.pipeline import (
    load_raw_data, load_processed_data, load_aggregated_data,
    process_raw_data
)
from excursion_bands.data import(load_yaml)
from excursion_bands.paths import CONFIGS
from excursion_bands.utils import logger

data_cfg = load_yaml(CONFIGS / "data/local_nq.yaml")
sessions_cfg = load_yaml(CONFIGS / "sessions/nq_default.yaml")

df_1m = load_raw_data(data_cfg)
df_30m = load_processed_data(data_cfg, df_1m)

df_1m = process_raw_data(data_cfg, sessions_cfg, df_1m)
df_30m = process_raw_data(data_cfg, sessions_cfg, df_30m)
aggregated_data = load_aggregated_data(data_cfg, df_30m)

print(df_1m.head(3))
print(df_30m.head(3))
print(aggregated_data.head(3))

volatility_specs_cfg = load_yaml(CONFIGS / "features/volatility/specification.yaml")

def _log_returns(spec: VolatilitySpec) -> tuple[pl.Expr, pl.Expr]:
    log_overnight = (pl.col(spec.open) / pl.col(spec.prev_close).shift(1)).log()

    log_oc = (pl.col(spec.close) / pl.col(spec.open)).log()

    return log_overnight.alias("_log_overnight"), log_oc.alias("_log_oc")

def _rogers_satchell(spec: VolatilitySpec) -> pl.Expr:
    h = pl.max_horizontal(*spec.high_cols)
    l = pl.min_horizontal(*spec.low_cols)
    o = pl.col(spec.open)
    c = pl.col(spec.close)

    rs = (h / c).log() * (h / o).log() + (l / c).log() * (l / o).log()
    return rs.alias("_rs")

def _historical_yz_expr(window: int) -> pl.Expr:
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    return (
        pl.col("_log_overnight").rolling_var(window)
        + k * pl.col("_log_oc").rolling_var(window)
        + (1 - k) * pl.col("_rs").rolling_mean(window)
    ).sqrt()

def _today_yz_expr() -> pl.Expr:
    return (
        pl.col("_log_overnight") ** 2
        + pl.col("_log_oc") ** 2
        + pl.col("_rs")
    ).sqrt()

def yang_zhang(config_file: dict, df: pl.DataFrame, mode: Literal["historical", "today"]):
    _tag_str = "[yang_zhang]"
    spec = load_volatility_spec(config_file, mode)
    window = config_file["lookback_window"]

    print(logger(_tag_str, f"Calculating Yang Zhang volatility for {window}"))

    log_overnight, log_oc = _log_returns(spec)
    rs = _rogers_satchell(spec)

    df = df.with_columns([
        log_overnight,
        log_oc,
        rs
    ])

    if mode == "historical":
        yz_expr = _historical_yz_expr(window)
    else:
        yz_expr = _today_yz_expr()

    return df.with_columns(
        yz_expr.alias(spec.output_col)
    ).drop([
        "_log_overnight",
        "_log_oc",
        "_rs"
    ])

aggregated_data = yang_zhang(volatility_specs_cfg, aggregated_data, "historical")
print(aggregated_data.select(["Session", "Sigma_historical"]).tail(3))
