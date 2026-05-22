import matplotlib.pyplot as plt

from excursion_bands.data import load_yaml
from excursion_bands.paths import CONFIGS
from excursion_bands.pipeline import (
    load_aggregated_data,
    load_excursion_bands_data,
    load_processed_data,
    load_raw_data,
    process_raw_data,
)
from excursion_bands.visualization import plot_excursion_band_session


data_cfg = load_yaml(CONFIGS / "data/local_nq.yaml")
sessions_cfg = load_yaml(CONFIGS / "sessions/nq_default.yaml")
volatility_cfg = load_yaml(CONFIGS / "features/volatility/specification.yaml")
bands_cfg = load_yaml(CONFIGS / "features/bands/nq_default.yaml")

df_1m = load_raw_data(data_cfg)
df_main = load_processed_data(data_cfg, df_1m)
df_main= process_raw_data(data_cfg, sessions_cfg, df_main)

aggregated_data = load_aggregated_data(data_cfg, df_main)
excursion_bands_data = load_excursion_bands_data(
    data_cfg,
    volatility_cfg,
    bands_cfg,
    aggregated_data,
)

fig, ax = plot_excursion_band_session(
    df=df_main,
    df_bands=excursion_bands_data,
    session=("2026-02-04"),
    show_centers=False,
    show_ae=True,
    show_fe=True,
)

plt.show()
