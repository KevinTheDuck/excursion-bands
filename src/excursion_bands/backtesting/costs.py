"""
Execution cost helpers for CFD-style futures backtests.
"""

from excursion_bands.backtesting.specification import ExecutionConfig, InstrumentConfig


def slippage_points(execution: ExecutionConfig, instrument: InstrumentConfig) -> float:
    return execution.slippage_ticks_per_side * instrument.tick_size


def commission_per_unit(execution: ExecutionConfig, instrument: InstrumentConfig) -> float:
    contract_to_cfd_ratio = instrument.point_value / instrument.cfd_point_value
    return execution.commission_per_contract_side / contract_to_cfd_ratio


def commission_for_units(
    units: float, execution: ExecutionConfig, instrument: InstrumentConfig
) -> float:
    return abs(units) * commission_per_unit(execution, instrument)


def apply_entry_slippage(side: str, price: float, slip_points: float) -> float:
    if side == "long":
        return price + slip_points
    if side == "short":
        return price - slip_points
    raise ValueError(f"Unsupported side: {side}")


def apply_exit_slippage(side: str, price: float, slip_points: float) -> float:
    if side == "long":
        return price - slip_points
    if side == "short":
        return price + slip_points
    raise ValueError(f"Unsupported side: {side}")
