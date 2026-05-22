import statistics

import polars as pl


def get_available_sessions(df: pl.DataFrame) -> list[str]:
    """
    Return sorted session labels available in the dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing a `Session` column.

    Returns
    -------
    list[str]
        Sorted session labels converted to strings.
    """
    if "Session" not in df.columns:
        raise ValueError("Expected dataframe with 'Session' column")

    return sorted(str(session) for session in df["Session"].unique().to_list())


def validate_single_session(df: pl.DataFrame) -> None:
    """
    Ensure a dataframe contains rows for exactly one trading session.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing a `Session` column.
    """
    if "Session" not in df.columns:
        raise ValueError("Expected dataframe with 'Session' column")

    n_sessions = df["Session"].n_unique()
    if n_sessions != 1:
        raise ValueError(f"Expected exactly one session, found {n_sessions}")


def select_session(df: pl.DataFrame, session: str | None = None) -> pl.DataFrame:
    """
    Select a single session from a dataframe and sort it chronologically.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `Session` and `DateTime` columns.

    session : str | None, default=None
        Session label to filter. If omitted, the latest available session
        is used.

    Returns
    -------
    pl.DataFrame
        Single-session dataframe sorted by `DateTime` when available.
    """
    if "Session" not in df.columns:
        raise ValueError("Expected dataframe with 'Session' column")

    session_key = session
    if session_key is None:
        available_sessions = get_available_sessions(df)
        if not available_sessions:
            raise ValueError("No sessions available in dataframe")
        session_key = available_sessions[-1]

    filtered = df.filter(pl.col("Session").cast(pl.String) == session_key)

    if filtered.is_empty():
        raise ValueError(f"Session '{session_key}' not found in dataframe")

    if "DateTime" in filtered.columns:
        filtered = filtered.sort("DateTime")

    validate_single_session(filtered)
    return filtered


def infer_timeframe_label(df: pl.DataFrame) -> str | None:
    """
    Infer a human-readable timeframe label from sorted `DateTime` values.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `DateTime` column.

    Returns
    -------
    str | None
        Timeframe label like "5m" or "2h" when available.
    """
    if "DateTime" not in df.columns or df.height < 2:
        return None

    timestamps = df["DateTime"].to_list()
    diffs = [
        (b - a).total_seconds()
        for a, b in zip(timestamps, timestamps[1:], strict=False)
        if a is not None and b is not None
    ]
    diffs = [diff for diff in diffs if diff > 0]
    if not diffs:
        return None

    median_seconds = statistics.median(diffs)

    if median_seconds >= 3600 and median_seconds % 3600 == 0:
        hours = int(median_seconds / 3600)
        return f"{hours}h"

    if median_seconds >= 60 and median_seconds % 60 == 0:
        minutes = int(median_seconds / 60)
        return f"{minutes}m"

    if median_seconds >= 60:
        minutes = median_seconds / 60
        return f"{minutes:.1f}m"

    return f"{median_seconds:.0f}s"
