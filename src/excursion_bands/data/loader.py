from pathlib import Path
from typing import Any

import polars as pl
import yaml

from excursion_bands.utils import logger

"""
[Data/loader.py]
A general purpose data loaders, file_path are usually expected to be a direct path to a file not a directory
"""


def load_parquet(file_path: Path) -> pl.DataFrame:
    _tag_str = "[data/loader/load_paquet]"
    print(logger(_tag_str, f"Loading parquet file from {file_path}"))

    # path must direct to a file
    if not (file_path.is_file()):
        raise FileNotFoundError(logger(_tag_str, f"Error: File not found! {file_path}"))

    return pl.read_parquet(file_path)


def load_yaml(file_path: Path) -> dict[str, Any]:
    _tag_str = "[data/loader/load_yaml]"
    print(logger(_tag_str, f"Loading YAML from {file_path}"))

    # path must direct to a file
    if not (file_path.is_file()):
        raise FileNotFoundError(logger(_tag_str, f"Error: File not found! {file_path}"))

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(logger(_tag_str, f"Error: YAML file is empty: {file_path}"))

    return config
