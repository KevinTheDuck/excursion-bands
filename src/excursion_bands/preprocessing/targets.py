import polars as pl

from excursion_bands.utils import logger


SLOT_COLUMN_MAPPING = {
    1: {
        "O_pre_target_1": "O_pre_target_1",
        "H_pre_target_1": "H_pre_target_1",
        "L_pre_target_1": "L_pre_target_1",
        "C_pre_target_1": "C_pre_target_1",
        "O_pre_target_2": "O_pre_target_2",
        "H_pre_target_2": "H_pre_target_2",
        "L_pre_target_2": "L_pre_target_2",
        "C_pre_target_2": "C_pre_target_2",
        "O_target": "O_target_1",
        "H_target": "H_target_1",
        "L_target": "L_target_1",
        "C_target": "C_target_1",
    },
}


def _shared_columns(df: pl.DataFrame) -> list[str]:
    slot_specific_columns = {
        "O_pre_target_1",
        "H_pre_target_1",
        "L_pre_target_1",
        "C_pre_target_1",
        "O_pre_target_2",
        "H_pre_target_2",
        "L_pre_target_2",
        "C_pre_target_2",
        "O_target_1",
        "H_target_1",
        "L_target_1",
        "C_target_1",
        "O_target_2",
        "H_target_2",
        "L_target_2",
        "C_target_2",
        "O_ref",
    }

    return [
        column
        for column in df.columns
        if column not in slot_specific_columns and not column.startswith("_")
    ]


def _build_slot_frame(
    df: pl.DataFrame,
    slot: int,
    shared_columns: list[str],
) -> pl.DataFrame:
    if slot == 2:
        return _build_slot_2_frame(df, shared_columns)

    mapping = SLOT_COLUMN_MAPPING[slot]

    expressions = [pl.col(column) for column in shared_columns]
    expressions.extend(
        [
            pl.lit(slot).alias("Target_Slot"),
            pl.col("C_target_2").alias("Daily_Close"),
            pl.col(mapping["O_pre_target_1"]).alias("O_ref"),
        ]
    )

    for output_column, source_column in mapping.items():
        expressions.append(pl.col(source_column).alias(output_column))

    return df.select(expressions)


def _build_slot_2_frame(df: pl.DataFrame, shared_columns: list[str]) -> pl.DataFrame:
    expressions = [pl.col(column) for column in shared_columns]
    expressions.extend(
        [
            pl.lit(2).alias("Target_Slot"),
            pl.col("C_target_2").alias("Daily_Close"),
            # Slot 2 merges old pre_target_1 + pre_target_2 into the new pre_target_1 block.
            pl.col("O_pre_target_1").alias("O_ref"),
            pl.col("O_pre_target_1").alias("O_pre_target_1"),
            pl.max_horizontal(pl.col("H_pre_target_1"), pl.col("H_pre_target_2")).alias(
                "H_pre_target_1"
            ),
            pl.min_horizontal(pl.col("L_pre_target_1"), pl.col("L_pre_target_2")).alias(
                "L_pre_target_1"
            ),
            pl.col("C_pre_target_2").alias("C_pre_target_1"),
            pl.col("O_target_1").alias("O_pre_target_2"),
            pl.col("H_target_1").alias("H_pre_target_2"),
            pl.col("L_target_1").alias("L_pre_target_2"),
            pl.col("C_target_1").alias("C_pre_target_2"),
            pl.col("O_target_2").alias("O_target"),
            pl.col("H_target_2").alias("H_target"),
            pl.col("L_target_2").alias("L_target"),
            pl.col("C_target_2").alias("C_target"),
        ]
    )

    return df.select(expressions)


def split_excursion_band_rows(df: pl.DataFrame) -> pl.DataFrame:
    """
    Split full-session excursion band rows into target-slot training rows.

    Parameters
    ----------
    df : pl.DataFrame
        Full-session dataframe produced after volatility and excursion band
        calculation.

        Expected to contain wide session OHLC columns for:
        - pre_target_1
        - pre_target_2
        - target_1
        - target_2

    Returns
    -------
    pl.DataFrame
        Clean normalized dataframe with two rows per `Session`:
        - `Target_Slot` in {1, 2}
        - shared full-day context copied to both rows
        - slot-local OHLC columns renamed to generic
          `pre_target_1`, `pre_target_2`, and `target` fields
        - `Daily_Close` preserved as the original `C_target_2`
        - `O_ref` realigned to the new `O_pre_target_1`

    Notes
    -----
    This transform is intentionally post-feature preprocessing. It keeps
    full-day feature engineering untouched and reshapes the data only for
    downstream training use.
    """
    _tag_str = "[preprocessing/targets/split_excursion_band_rows]"
    print(logger(_tag_str, "Splitting excursion band rows into target slots"))

    required_columns = {
        "Session",
        "C_target_2",
        "O_pre_target_1",
        "H_pre_target_1",
        "L_pre_target_1",
        "C_pre_target_1",
        "O_pre_target_2",
        "H_pre_target_2",
        "L_pre_target_2",
        "C_pre_target_2",
        "O_target_1",
        "H_target_1",
        "L_target_1",
        "C_target_1",
        "O_target_2",
        "H_target_2",
        "L_target_2",
        "C_target_2",
    }

    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for target splitting: {missing}")

    shared_columns = _shared_columns(df)

    slot_1 = _build_slot_frame(df, 1, shared_columns)
    slot_2 = _build_slot_frame(df, 2, shared_columns)

    return pl.concat([slot_1, slot_2], how="vertical").sort(
        ["Session", "Target_Slot"],
        descending=[False, False],
    )
