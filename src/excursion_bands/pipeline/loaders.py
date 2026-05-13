import polars as pl

from excursion_bands.data import load_parquet, write_parquet, aggregate_sessions, filter_valid_sessions
from excursion_bands.paths import resolve_path
from excursion_bands.utils import logger
from excursion_bands.pipeline import aggregate_1m_data

def load_raw_data(config_file: dict) -> pl.DataFrame:
    """
    Load the primary raw dataset specified in the config file.

    Parameters
    ----------

    config_file : dict
        Configuration dictionary containing raw data paths.

        Expected Structure:
            {
                "raw": {
                    "main": "<path_to_parquet>"
                }
            }

    Returns
    -------
    pl.DataFrame
        Loaded raw dataset as a Polars DataFrame

    Notes
    -----
    File path listed in the config file must direct to the .parquet file
    """
    _tag_str = "[pipeline/loaders/load_raw_data]"
    file_path = config_file["raw"]["main"]
    print(logger(_tag_str, f"Loading raw data from {file_path}"))

    data_path, _ = resolve_path(file_path)
    df = load_parquet(data_path)
    return df

def load_processed_data(
    config_file: dict, raw_data: pl.DataFrame | None
) -> pl.DataFrame:
    """
    Load processed timeframe data from storage or create it from raw data.

    Parameters
    ----------
    config_file : dict
        Configuration dictionary containing processed data settings.

        Expected structure:
        {
            "processed": {
                "main": "<path_to_processed_parquet>",
                "timeframe": "<target_timeframe>"
            }
        }

    raw_data : pl.DataFrame | None
        Raw 1-minute OHLCV dataset used to generate processed data
        when the processed file does not already exist.

        Required if the processed dataset is missing.

    Returns
    -------
    pl.DataFrame
        Processed OHLCV dataframe at the configured timeframe.

    Notes
    -----
    If the processed dataset already exists:
        - Loads directly from parquet storage.

    If the processed dataset does not exist:
        - Aggregates `raw_data` into the desired timeframe
        - Saves the processed dataset to parquet
        - Returns the newly created dataframe

    File path listed in the config file must direct to the .parquet file

    Raises
    ------
    ValueError
        If processed data does not exist and `raw_data` is None.
    """
    _tag_str = "[pipeline/loaders/load_processed_data]"
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
            raise ValueError(logger(_tag_str, "raw_data is needed when data doesn't exist"))

        df = aggregate_1m_data(raw_data, desired_timeframe)
        write_parquet(df, data_path)
        return df

    df = load_parquet(data_path)
    return df

def load_aggregated_data(config_file: dict, data: pl.DataFrame | None) -> pl.DataFrame:
    """
    Load aggregated session-level data from storage or create it from
    processed intraday data.

    Parameters
    ----------
    config_file : dict
        Configuration dictionary containing the output path for
        aggregated session data.

        Expected structure:
        {
            "processed": {
                "aggregated": "<path_to_aggregated_parquet>"
            }
        }

    data : pl.DataFrame
        Processed intraday dataset used to generate aggregated data
        when the aggregated file does not already exist.

        Expected to already contain:
        - Session
        - Intraday_Session
        - OHLCV columns
        - any required fields used by `aggregate_sessions()`

    Returns
    -------
    pl.DataFrame
        Aggregated session-level dataframe.

    Notes
    -----
    If aggregated data already exists:
        - Loads directly from parquet storage.

    If aggregated data does not exist:
        - Aggregates intraday session data
        - Filters invalid sessions
        - Creates `O_ref` from `O_pre_target_1`
        - Sorts data chronologically by `Session`
        - Saves the newly created dataset to parquet
    """
    _tag_str = "[pipeline/loaders/load_aggregated_data]"
    print(logger(_tag_str, "Loading aggregated data"))

    file_path = config_file["processed"]["aggregated"]
    data_path, exists = resolve_path(file_path)

    if not exists:
        print(
            logger(_tag_str, "Aggregated data doesn't exist yet, creating a new one")
        )

        if data is None:
            raise ValueError(logger(_tag_str, "data is needed when data doesn't exist"))

        df = aggregate_sessions(data)
        df = filter_valid_sessions(df)
        df = df.with_columns(pl.col("O_pre_target_1").alias("O_ref"))
        df = df.sort("Session", descending=False)

        write_parquet(df, data_path)
        return df

    df = load_parquet(data_path)
    return df
