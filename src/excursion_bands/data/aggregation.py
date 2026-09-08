import polars as pl

from excursion_bands.utils import logger

"""
[aggregation.py]
Module for aggregating sessions
"""


def aggregate_sessions(df: pl.DataFrame) -> pl.DataFrame:
    _tag_str = "[data/aggregation/aggregate_session]"
    print(logger(_tag_str, "Aggregating sessions"))

    # Each row represents single trading day
    return (
        df.sort("DateTime").group_by(["Session", "Intraday_Session"])
        .agg(
            [
                pl.col("Open").first().alias("O"),
                pl.col("High").max().alias("H"),
                pl.col("Low").min().alias("L"),
                pl.col("Close").last().alias("C"),
            ]
        )
        .pivot(
            index="Session",
            on="Intraday_Session",
            values=["O", "H", "L", "C"],
        )
        .sort("Session")
    )
