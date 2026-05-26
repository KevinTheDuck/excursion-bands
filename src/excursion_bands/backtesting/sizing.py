"""
Position sizing rules.
"""

from excursion_bands.backtesting.specification import InstrumentConfig, SizingConfig


def calculate_units(
    sizing: SizingConfig,
    instrument: InstrumentConfig,
    initial_cash: float,
    current_equity: float,
    stop_distance_points: float | None,
) -> float:
    if sizing.mode == "fixed_units":
        units = sizing.fixed_units
    elif sizing.mode in {"risk_initial_equity", "risk_current_equity"}:
        if stop_distance_points is None or stop_distance_points <= 0:
            raise ValueError(
                "Risk-based sizing requires a positive stop loss distance. "
                "Set risk.stop_loss_points or use fixed_units sizing."
            )
        risk_base = initial_cash if sizing.mode == "risk_initial_equity" else current_equity
        risk_amount = risk_base * sizing.risk_pct
        units = risk_amount / (stop_distance_points * instrument.cfd_point_value)
    else:
        raise ValueError(f"Unsupported sizing mode: {sizing.mode}")

    if sizing.max_units is not None:
        units = min(units, sizing.max_units)

    return float(max(units, 0.0))
