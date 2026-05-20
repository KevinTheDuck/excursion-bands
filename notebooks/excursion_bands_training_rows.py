from excursion_bands.data import load_yaml
from excursion_bands.paths import CONFIGS
from excursion_bands.pipeline import (
    load_aggregated_data,
    load_excursion_bands_data,
    load_processed_data,
    load_raw_data,
    process_raw_data,
)
from excursion_bands.preprocessing import split_excursion_band_rows


data_cfg = load_yaml(CONFIGS / "data/local_nq.yaml")
sessions_cfg = load_yaml(CONFIGS / "sessions/nq_default.yaml")
volatility_cfg = load_yaml(CONFIGS / "features/volatility/specification.yaml")
bands_cfg = load_yaml(CONFIGS / "features/bands/nq_default.yaml")

df_1m = load_raw_data(data_cfg)
df_30m = load_processed_data(data_cfg, df_1m)
df_30m = process_raw_data(data_cfg, sessions_cfg, df_30m)

aggregated_data = load_aggregated_data(data_cfg, df_30m)
excursion_bands_data = load_excursion_bands_data(
    data_cfg,
    volatility_cfg,
    bands_cfg,
    aggregated_data,
)

training_rows = split_excursion_band_rows(excursion_bands_data)

print(training_rows.columns)
print(training_rows.tail(4))
