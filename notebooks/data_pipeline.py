# %% Modules
import polars as pl

from excursion_bands.data import (
    convert_to_timezone,
    intraday_session_tagging,
    load_parquet,
    load_yaml,
    remove_incomplete_days,
    session_tagging,
    write_parquet,
)
from excursion_bands.paths import CONFIGS, resolve_path
from excursion_bands.utils import logger

# %% Loading config
cfg = load_yaml(CONFIGS / "data/local_nq.yaml")


# %% Loading from local
def load_raw_data(config_file: dict) -> pl.DataFrame:
    _tag_str = "[load_raw_data]"
    file_path = config_file["raw"]["main"]
    print(logger(_tag_str, f"Loading raw data from {file_path}"))

    data_path, _ = resolve_path(file_path)
    df = load_parquet(data_path)
    return df


raw_1m = load_raw_data(cfg)
raw_1m.head(3)


# %% 1m -> 30m aggregate
def aggregate_1m_data(df: pl.DataFrame) -> pl.DataFrame:
    _tag_str = "[aggregate_1m_data]"
    print(logger(_tag_str, "Aggregating 1m data -> 30m data.."))
    return (
        df.sort("DateTime")
        .group_by_dynamic("DateTime", every="30m")
        .agg(
            [
                pl.col("Open").first(),
                pl.col("High").max(),
                pl.col("Low").min(),
                pl.col("Close").last(),
                pl.col("Volume").sum(),
            ]
        )
    )


# %% Loading 30m data from raw_1m
def load_30m_data(
    config_file: dict, raw_data: pl.DataFrame | None = None
) -> pl.DataFrame:
    _tag_str = "[load_30m_data]"
    file_path = cfg["processed"]["main"]
    print(logger(_tag_str, f"Loading 30m data from {file_path}"))

    data_path, exists = resolve_path(file_path)

    if not exists:
        print(logger(_tag_str, "30m data doesn't exists creating a new one..."))
        df = aggregate_1m_data(raw_data)
        write_parquet(df, data_path)
        return df

    df = load_parquet(data_path)
    return df


raw_30m = load_30m_data(cfg, raw_1m)
raw_30m.head(3)


# %% Processing raw data
def process_raw_data(config_file: dict, raw_data: pl.DataFrame) -> pl.DataFrame:
    _tag_str = "[process_raw_data]"
    print(logger(_tag_str, "Processing raw data..."))

    datetime_col = config_file["timezone"]["datetime_col"]
    session_cfg = config_file["session"]

    df = convert_to_timezone(
        raw_data,
        datetime_col,
        config_file["timezone"]["broker"],
        config_file["timezone"]["target"],
    )

    df = session_tagging(df, datetime_col, config_file["timezone"]["eod_close"])
    df = intraday_session_tagging(df, datetime_col, session_cfg)
    df = remove_incomplete_days(df)

    return df.select(
        pl.col(
            [
                "DateTime",
                "Session",
                "Intraday_Session",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        )
    )


df_30m = process_raw_data(cfg, raw_30m)
df_1m = process_raw_data(cfg, raw_1m)

# print(df_30m.head(3))
# print(df_1m.tail(3))
