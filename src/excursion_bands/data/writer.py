from pathlib import Path

import polars as pl

from excursion_bands.utils import logger

"""
[data/writer.py]
A general purpose data writer, used to safe or write files
"""


def write_parquet(df: pl.DataFrame, file_path: Path, overwrite: bool = False):
    _tag_str = "[data/writer/write_parquet]"
    print(logger(_tag_str, f"Writing dataframe into parquet to {file_path}"))

    if file_path.is_file() and not overwrite:
        raise FileExistsError(
            logger(
                _tag_str,
                f"Error: File already exists in {file_path} consider overwrite=True or delete file!",
            )
        )

    df.write_parquet(file_path)
