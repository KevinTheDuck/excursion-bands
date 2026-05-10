# %% Modules
import polars as pl

from excursion_bands.data import load_parquet, load_yaml
from excursion_bands.paths import CONFIGS, resolve_path

# %% Loading config
cfg = load_yaml(CONFIGS / "data/local_nq.yaml")


# %% Loading from local
def load_raw_data(config_file: dict) -> pl.DataFrame:
    _tag_str = "[load_raw_data]"
    file_path = cfg["raw"]["main"]

    raw_data, _ = resolve_path(file_path)
    df = load_parquet(raw_data)
    return df


raw_1m = load_raw_data(cfg)
raw_1m.head(3)
