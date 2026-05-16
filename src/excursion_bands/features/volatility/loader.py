from excursion_bands.features.volatility import VolatilitySpec
from excursion_bands.utils.verbose import logger

"""
[volatility/loader.py]
Loading volatility specification and return them as VolatilitySpec object
"""

def load_volatility_spec(config_file: dict, mode: str) -> VolatilitySpec:
    _tag_str = "[features/volatility/loader/load_volatility_spec]"
    print(logger(_tag_str, f"Loading volatility specification mode: {mode}"))
    spec_dict = config_file["specs"][mode]

    return VolatilitySpec(
        open=spec_dict["open"],
        high_cols=tuple(spec_dict["high_cols"]),
        low_cols=tuple(spec_dict["low_cols"]),
        close=spec_dict["close"],
        prev_close=spec_dict["prev_close"],
        output_col=spec_dict["output_col"],
    )
