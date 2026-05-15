import polars as pl

from excursion_bands.utils import logger
from excursion_bands.data import convert_to_timezone, session_tagging, intraday_session_tagging, remove_incomplete_days

def aggregate_1m_data(df: pl.DataFrame, timeframe: str = "30m") -> pl.DataFrame:
    """
    Aggregate raw 1-minute OHLCV data into a higher timeframe.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing raw 1-minute market data.

        Required columns:
        - DateTime
        - Open
        - High
        - Low
        - Close
        - Volume

        `DateTime` must be of type Datetime and represent
        chronologically ordered market timestamps.

    timeframe : str, default="30m"
        Target aggregation interval used in Polars dynamic grouping.

        Examples:
        - "5m"   -> 5-minute candles
        - "15m"  -> 15-minute candles
        - "30m"  -> 30-minute candles
        - "1h"   -> hourly candles

    Returns
    -------
    pl.DataFrame
        Aggregated OHLCV dataframe where each row represents
        a higher timeframe candle:
        - Open  -> first value
        - High  -> maximum value
        - Low   -> minimum value
        - Close -> last value
        - Volume -> summed value

    Notes
    -----
    Data is sorted by `DateTime` before aggregation to ensure
    correct OHLC calculations.
    """
    _tag_str = "[pipeline/processing/aggregate_1m_data]"
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

def process_raw_data(
    data_config_file: dict,
    session_config_file: dict,
    raw_data: pl.DataFrame
) -> pl.DataFrame:
    """
    Process raw market data into a cleaned, session-aware dataset.

    Parameters
    ----------
    data_config_file : dict
        Configuration dictionary containing data and timezone settings.

        Required sections:
        - timezone
            - datetime_col
            - broker
            - target
            - eod_close

    session_config_file : dict
        Configuration dictionary containing intraday session definitions.

        Required structure:
        {
            "session": {
                "<session_name>": {
                    "start": "HH:MM",
                    "end": "HH:MM"
                }
            }
        }

    raw_data : pl.DataFrame
        Raw OHLCV market dataset.

        Expected columns:
        - DateTime
        - Open
        - High
        - Low
        - Close
        - Volume

    Returns
    -------
    pl.DataFrame
        Processed dataframe containing:
        - DateTime
        - Session
        - Intraday_Session
        - Open
        - High
        - Low
        - Close
        - Volume

    Notes
    -----
    Processing pipeline performs:

    1. Convert broker timestamps to target timezone
    2. Assign daily session labels
    3. Assign intraday session labels
    4. Remove incomplete trading days
    5. Return only required output columns
    """
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

    df = session_tagging(
        df,
        datetime_col,
        data_config_file["timezone"]["eod_close"]
    )

    df = intraday_session_tagging(
        df,
        datetime_col,
        session_cfg
    )

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
