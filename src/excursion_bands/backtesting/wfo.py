"""
Walk-forward optimization scaffolding.
"""

from collections.abc import Sequence

from excursion_bands.backtesting.specification import WfoConfig


def validate_parameter_sweep(parameters: Sequence[object], config: WfoConfig) -> None:
    if len(parameters) > config.max_parameter_combinations:
        raise ValueError(
            f"Parameter sweep has {len(parameters)} combinations, exceeding "
            f"configured max_parameter_combinations={config.max_parameter_combinations}."
        )
    if config.max_workers < 1:
        raise ValueError("wfo.max_workers must be >= 1")
