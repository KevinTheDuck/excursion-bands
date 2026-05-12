from datetime import time

import polars as pl

from excursion_bands.utils import logger

"""
[standard.py]
Module to standardize data to desired state before any preprocessing
"""


def convert_to_timezone(
    df: pl.DataFrame,
    dt_col: str,
    current_tz: str,
    target_tz: str,
) -> pl.DataFrame:
    _tag_str = "[data/standard/convert_to_timezone]"
    print(logger(_tag_str, "Converting to target timezone"))

    return df.with_columns(
        pl.col(dt_col).dt.replace_time_zone(current_tz).dt.convert_time_zone(target_tz)
    )


def remove_incomplete_days(df: pl.DataFrame) -> pl.DataFrame:
    _tag_str = "[data/standard/remove_incomplete_days]"
    print(logger(_tag_str, "Removing incomplete days (n_sessions != 4)"))

    valid_days = (
        df.group_by("Session")
        .agg(pl.col("Intraday_Session").n_unique().alias("n_sessions"))
        .filter(pl.col("n_sessions") == 4)
        .select("Session")
    )

    return df.join(valid_days, on="Session", how="inner")


def session_tagging(
    df: pl.DataFrame,
    dt_col: str,
    eod: int,
) -> pl.DataFrame:
    _tag_str = "[data/standard/session_tagging]"
    print(logger(_tag_str, "Tagging daily sessions"))

    return df.with_columns(
        pl.when(pl.col(dt_col).dt.hour() >= eod)
        .then((pl.col(dt_col) + pl.duration(days=1)).dt.date())
        .otherwise(pl.col(dt_col).dt.date())
        .alias("Session")
    )


def intraday_session_tagging(
    df: pl.DataFrame,
    dt_col: str,
    sessions: dict,
) -> pl.DataFrame:
    _tag_str = "[data/standard/intraday_session_tagging]"
    print(logger(_tag_str, "Tagging intraday sessions"))

    time_col = pl.col(dt_col).dt.time()

    expr = None

    for session_name, interval in sessions.items():
        start = time.fromisoformat(interval["start"])
        end = time.fromisoformat(interval["end"])
        label = pl.lit(session_name)

        if start < end:
            condition = time_col.is_between(start, end, closed="left")
        else:
            condition = (time_col >= start) | (time_col < end)

        expr = (
            pl.when(condition).then(label)
            if expr is None
            else expr.when(condition).then(label)
        )

    df = df.with_columns(expr.otherwise(pl.lit("Closed")).alias("Intraday_Session"))

    return df
