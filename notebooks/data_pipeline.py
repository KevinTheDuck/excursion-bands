import polars as pl

from excursion_bands.data import (
    aggregate_sessions,
    convert_to_timezone,
    filter_valid_sessions,
    intraday_session_tagging,
    load_parquet,
    load_yaml,
    remove_incomplete_days,
    session_tagging,
    write_parquet,
)
from excursion_bands.paths import CONFIGS, resolve_path
from excursion_bands.utils import logger

cfg = load_yaml(CONFIGS / "data/local_nq.yaml")
sessions = load_yaml(CONFIGS / "sessions/nq_default.yaml")

def load_raw_data(config_file: dict) -> pl.DataFrame:
    _tag_str = "[load_raw_data]"
    file_path = config_file["raw"]["main"]
    print(logger(_tag_str, f"Loading raw data from {file_path}"))

    data_path, _ = resolve_path(file_path)
    df = load_parquet(data_path)
    return df


raw_1m = load_raw_data(cfg)


def aggregate_1m_data(df: pl.DataFrame, timeframe: str = "30m") -> pl.DataFrame:
    _tag_str = "[aggregate_1m_data]"
    print(logger(_tag_str, f"Aggregating 1m data -> {timeframe} data.."))
    return (
        df.sort("DateTime")
        .group_by_dynamic("DateTime", every=timeframe)
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


def load_processed_data(
    config_file: dict, raw_data: pl.DataFrame | None
) -> pl.DataFrame:
    _tag_str = "[load_processed_data]"
    file_path = config_file["processed"]["main"]
    desired_timeframe = config_file["processed"]["timeframe"]
    print(
        logger(_tag_str, f"Loading processed {desired_timeframe} data from {file_path}")
    )

    data_path, exists = resolve_path(file_path)

    if not exists:
        print(
            logger(
                _tag_str,
                f"{desired_timeframe} data doesn't exists creating a new one...",
            )
        )

        if raw_data is None:
            raise ValueError(logger(_tag_str, "raw_data is needed when data doesn't exists"))

        df = aggregate_1m_data(raw_data, desired_timeframe)
        write_parquet(df, data_path)
        return df

    df = load_parquet(data_path)
    return df


raw_30m = load_processed_data(cfg, raw_1m)
print(raw_30m.shape)


def process_raw_data(data_config_file: dict, session_config_file: dict, raw_data: pl.DataFrame) -> pl.DataFrame:
    _tag_str = "[process_raw_data]"
    print(logger(_tag_str, "Processing raw data..."))

    datetime_col = data_config_file["timezone"]["datetime_col"]
    session_cfg = session_config_file["session"]

    df = convert_to_timezone(
        raw_data,
        datetime_col,
        data_config_file["timezone"]["broker"],
        data_config_file["timezone"]["target"],
    )

    df = session_tagging(df, datetime_col, data_config_file["timezone"]["eod_close"])
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


df_30m = process_raw_data(cfg, sessions, raw_30m)
df_1m = process_raw_data(cfg, sessions, raw_1m)

print(df_30m.head(3))
print(df_1m.tail(3))


def load_aggregated_data(config_file: dict, data: pl.DataFrame) -> pl.DataFrame:
    _tag_str = "[load_aggregated_data]"
    print(logger(_tag_str, "Loading aggregated data"))

    file_path = config_file["processed"]["aggregated"]
    data_path, exists = resolve_path(file_path)

    if not exists:
        print(
            logger(_tag_str, "Aggregated Data doesn't exists yet, creating a new one")
        )
        df = aggregate_sessions(data)
        df = filter_valid_sessions(df)
        df = df.with_columns(pl.col("O_pre_target_1").alias("O_ref"))
        df = df.sort("Session", descending=False)

        write_parquet(df, data_path)
        return df

    df = load_parquet(data_path)
    return df


aggregated_data = load_aggregated_data(cfg, df_30m)
print(aggregated_data.head(3))
print(aggregated_data.shape)
