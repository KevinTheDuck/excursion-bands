from typing import Literal
import polars as pl

from excursion_bands.features.volatility import load_volatility_spec, VolatilitySpec
from excursion_bands.utils import logger

def _log_returns(spec: VolatilitySpec) -> tuple[pl.Expr, pl.Expr]:
    """
    Compute log return features used in the Yang-Zhang volatility estimator.

    Parameters
    ----------
    spec : VolatilitySpec
        Specification object containing column mappings:
        - open: open price column
        - close: close price column
        - prev_close: previous close column

    Returns
    -------
    tuple[pl.Expr, pl.Expr]
        - _log_overnight: log return from previous close to current open
        - _log_oc: log return from open to close
    """
    log_overnight = (pl.col(spec.open) / pl.col(spec.prev_close).shift(1)).log()

    log_oc = (pl.col(spec.close) / pl.col(spec.open)).log()

    return log_overnight.alias("_log_overnight"), log_oc.alias("_log_oc")

def _rogers_satchell(spec: VolatilitySpec) -> pl.Expr:
    """
    Compute Rogers-Satchell volatility component.

    Uses high, low, open, and close prices to estimate
    volatility independent of drift.

    Parameters
    ----------
    spec : VolatilitySpec
        Column mapping specification including:
        - high_cols: list of high price columns
        - low_cols: list of low price columns
        - open: open price column
        - close: close price column

    Returns
    -------
    pl.Expr
        Rogers-Satchell volatility contribution per row.
    """
    h = pl.max_horizontal(*spec.high_cols)
    l = pl.min_horizontal(*spec.low_cols)
    o = pl.col(spec.open)
    c = pl.col(spec.close)

    rs = (h / c).log() * (h / o).log() + (l / c).log() * (l / o).log()
    return rs.alias("_rs")

def _historical_yz_expr(window: int) -> pl.Expr:
    """
    Construct historical Yang-Zhang volatility expression.

    Combines rolling variance of:
    - overnight log returns
    - open-close log returns

    and rolling mean of Rogers-Satchell component.

    Parameters
    ----------
    window : int
        Rolling lookback window size.

    Returns
    -------
    pl.Expr
        Square-rooted Yang-Zhang volatility estimate.
    """
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    return (
        pl.col("_log_overnight").rolling_var(window)
        + k * pl.col("_log_oc").rolling_var(window)
        + (1 - k) * pl.col("_rs").rolling_mean(window)
    ).sqrt()

def _today_yz_expr() -> pl.Expr:
    """
    Construct single-period Yang-Zhang volatility estimate.

    Computes instantaneous volatility using:
    - overnight log return squared
    - open-close log return squared
    - Rogers-Satchell component

    Returns
    -------
    pl.Expr
        Instantaneous Yang-Zhang volatility estimate.
    """
    return (
        pl.col("_log_overnight") ** 2
        + pl.col("_log_oc") ** 2
        + pl.col("_rs")
    ).sqrt()

def yang_zhang(config_file: dict, df: pl.DataFrame, mode: Literal["historical", "today"]):
    """
    Compute Yang-Zhang volatility feature.

    Parameters
    ----------
    config_file : dict
        Configuration containing:
        - volatility specs (column mappings)
        - lookback window size

    df : pl.DataFrame
        Input dataframe containing OHLC price data.

    mode : {"historical", "today"}
        - historical: rolling volatility estimate
        - today: single-period volatility estimate

    Returns
    -------
    pl.DataFrame
        Input dataframe with Yang-Zhang volatility column added
        and intermediate features removed.

    Notes
    -----
    Intermediate features used:
    - _log_overnight
    - _log_oc
    - _rs

    These are temporarily added and removed within the pipeline.
    """
    _tag_str = "[features/volatility/yang_zhang/yang_zhang]"
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
