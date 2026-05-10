# %% Modules
import polars as pl

from excursion_bands.data import load_parquet, load_yaml, write_parquet
from excursion_bands.paths import CONFIGS, resolve_path
from excursion_bands.utils import logger

# %% Loading config
cfg = load_yaml(CONFIGS / "data/local_nq.yaml")


# %% Loading from local
def load_raw_data(config_file: dict) -> pl.DataFrame:
    _tag_str = "[load_raw_data]"
    file_path = cfg["raw"]["main"]
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
        if raw_data is None:
            raise ValueError(
                logger(_tag_str, "Error: raw_data is None and 30m doesn't exists yet")
            )
        df = aggregate_1m_data(raw_data)
        write_parquet(df, data_path)
        return df

    df = load_parquet(data_path)
    return df


raw_30m = load_30m_data(cfg, raw_1m)
raw_30m.head(3)
