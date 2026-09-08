"""
Small single-position bar backtesting engine.
"""

from collections.abc import Callable, Collection
from dataclasses import replace
from datetime import date, datetime
from math import isfinite

import pandas as pd

from excursion_bands.backtesting.costs import (
    apply_entry_slippage,
    apply_exit_slippage,
    commission_for_units,
    slippage_points,
)
from excursion_bands.backtesting.metrics import calculate_metrics
from excursion_bands.backtesting.models import BacktestResult, Position, Signal, Trade
from excursion_bands.backtesting.sizing import calculate_units
from excursion_bands.backtesting.specification import BacktestConfig

SignalValue = Signal | str | None
SignalFunc = Callable[[int, pd.Series, pd.DataFrame, Position | None], SignalValue]


def _canonical_session(value: object) -> object:
    """Compare date and timestamp session labels without changing custom labels."""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date()
    return value


class _BarView:
    """Cheap row adapter used by the hot loop.

    Built-in strategies use attribute access while existing user callbacks
    commonly use ``bar["Close"]`` or ``bar.get(...)``.  Keeping both forms
    avoids the cost of constructing a pandas Series for every bar without
    changing the callback surface.
    """

    __slots__ = ("_index", "_row")

    def __init__(self, row: tuple, index: int) -> None:
        self._row = row
        self._index = index

    def __getattr__(self, name: str):
        return getattr(self._row, name)

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self._row, key)
        return self._row[key]

    def get(self, key: str, default=None):
        return getattr(self._row, key, default)

    @property
    def index(self) -> tuple[str, ...]:
        """Column names, matching the common ``Series.index`` use case."""
        return self._row._fields

    @property
    def name(self) -> int:
        """Original integer row index exposed by ``Series.name``."""
        return self._index

    def __contains__(self, key: object) -> bool:
        return key in self._row._fields


def _position_value(position: Position, close_price: float, cfd_point_value: float) -> float:
    direction = 1 if position.side == "long" else -1
    return (close_price - position.entry_price) * direction * position.units * cfd_point_value


def _build_stop_take_profit(
    side: str, entry_price: float, stop_points: float | None, take_profit_points: float | None
) -> tuple[float | None, float | None]:
    if side == "long":
        stop = None if stop_points is None else entry_price - stop_points
        take_profit = None if take_profit_points is None else entry_price + take_profit_points
    else:
        stop = None if stop_points is None else entry_price + stop_points
        take_profit = None if take_profit_points is None else entry_price - take_profit_points
    return stop, take_profit


def _exit_from_bar(position: Position, bar: pd.Series, priority: str) -> tuple[float, str] | None:
    if position.force_exit_time is not None and bar.DateTime >= position.force_exit_time:
        # The scheduled exit is a market order at the opening of the cutoff
        # bar.  This keeps the backtest from using information inside that
        # bar to fill an already scheduled liquidation.
        return float(bar.Open), "force_exit"

    high = float(bar.High)
    low = float(bar.Low)
    open_price = float(bar.Open)
    # Opening gaps have a known order: they precede the intrabar range.
    if position.stop_loss is not None and (
        open_price <= position.stop_loss if position.side == "long"
        else open_price >= position.stop_loss
    ):
        return open_price, "stop_loss"
    if position.take_profit is not None and (
        open_price >= position.take_profit if position.side == "long"
        else open_price <= position.take_profit
    ):
        return float(position.take_profit), "take_profit"

    if position.side == "long":
        stop_hit = position.stop_loss is not None and (
            float(bar.Open) <= position.stop_loss or low <= position.stop_loss
        )
        tp_hit = position.take_profit is not None and (
            float(bar.Open) >= position.take_profit or high >= position.take_profit
        )
        stop_fill = None
        if position.stop_loss is not None:
            stop_fill = (
                float(bar.Open)
                if float(bar.Open) <= position.stop_loss
                else float(position.stop_loss)
            )
    else:
        stop_hit = position.stop_loss is not None and (
            float(bar.Open) >= position.stop_loss or high >= position.stop_loss
        )
        tp_hit = position.take_profit is not None and (
            float(bar.Open) <= position.take_profit or low <= position.take_profit
        )
        stop_fill = None
        if position.stop_loss is not None:
            stop_fill = (
                float(bar.Open)
                if float(bar.Open) >= position.stop_loss
                else float(position.stop_loss)
            )

    if stop_hit and tp_hit:
        if priority != "stop":
            return float(position.take_profit), "take_profit"
        return float(stop_fill), "stop_loss"
    if stop_hit:
        return float(stop_fill), "stop_loss"
    if tp_hit:
        return float(position.take_profit), "take_profit"

    # A break-even trigger observed in this bar changes the stop for the next
    # bar.  Evaluating the original stop first prevents an intrabar trigger
    # from retroactively cancelling adverse movement earlier in the bar.
    if not position.break_even_moved and position.break_even_trigger is not None:
        triggered = (
            high >= position.break_even_trigger
            if position.side == "long"
            else low <= position.break_even_trigger
        )
        if triggered and position.break_even_stop is not None:
            position.stop_loss = position.break_even_stop
            position.break_even_moved = True
    return None


def _normalize_signal(signal: SignalValue) -> Signal | None:
    if signal is None:
        return None
    if isinstance(signal, Signal):
        return signal
    if signal in {"long", "short"}:
        return Signal(side=signal)
    raise ValueError(f"Unsupported signal value: {signal}")


def run_backtest(
    name: str,
    bars: pd.DataFrame,
    config: BacktestConfig,
    signal_func: SignalFunc,
    *,
    starting_cash: float | None = None,
    session_filter: Collection[object] | None = None,
    fast_rows: bool = False,
) -> BacktestResult:
    if config.execution.fill_on not in {"next_open", "next_bar_open"}:
        raise ValueError("execution.fill_on must be 'next_open'")
    required = {"DateTime", "Open", "High", "Low", "Close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"Backtest bars missing required columns: {sorted(missing)}")

    bars = bars.sort_values("DateTime").reset_index(drop=True).copy()
    bars["DateTime"] = pd.to_datetime(bars["DateTime"])
    bars = bars[bars["DateTime"].dt.date >= config.backtest.start_date]
    if config.backtest.end_date is not None:
        bars = bars[bars["DateTime"].dt.date <= config.backtest.end_date]
    if session_filter is not None:
        if "Session" not in bars.columns:
            raise ValueError("session_filter requires a Session column")
        allowed_sessions = {_canonical_session(value) for value in session_filter}
        bars = bars[
            bars["Session"].map(_canonical_session).isin(allowed_sessions)
        ]
    bars = bars.reset_index(drop=True)

    if len(bars) < 2:
        raise ValueError("Backtest requires at least two bars after date filtering")

    initial_cash = (
        float(config.backtest.initial_cash)
        if starting_cash is None
        else float(starting_cash)
    )
    if not isfinite(initial_cash) or initial_cash <= 0:
        raise ValueError("starting_cash must be a finite positive value")
    cash = initial_cash
    position: Position | None = None
    pending_entry: Signal | None = None
    trades: list[Trade] = []
    equity_rows = []
    slip = slippage_points(config.execution, config.instrument)

    rows = (
        ((i, _BarView(row, i)) for i, row in enumerate(bars.itertuples(index=False)))
        if fast_rows else bars.iterrows()
    )
    for i, bar in rows:
        if position is None and pending_entry is not None:
            eligible = True
            if pending_entry.force_exit_time is not None and bar.DateTime >= pending_entry.force_exit_time:
                eligible = False
            if (
                eligible
                and pending_entry.session is not None
                and hasattr(bar, "Session")
                and _canonical_session(bar.Session)
                != _canonical_session(pending_entry.session)
            ):
                eligible = False
            if eligible:
                entry_price = apply_entry_slippage(pending_entry.side, float(bar.Open), slip)
                stop_distance = None
                if pending_entry.stop_loss is not None:
                    stop_distance = abs(entry_price - pending_entry.stop_loss)
                elif config.risk.stop_loss_points is not None:
                    stop_distance = config.risk.stop_loss_points
                units = calculate_units(
                    config.sizing,
                    config.instrument,
                    config.backtest.initial_cash,
                    cash,
                    stop_distance,
                )
                if units > 0:
                    entry_commission = commission_for_units(
                        units, config.execution, config.instrument
                    )
                    cash -= entry_commission
                    if pending_entry.stop_loss is not None or pending_entry.take_profit is not None:
                        stop = pending_entry.stop_loss
                        take_profit = pending_entry.take_profit
                    else:
                        stop, take_profit = _build_stop_take_profit(
                            pending_entry.side,
                            entry_price,
                            config.risk.stop_loss_points,
                            config.risk.take_profit_points,
                        )
                    position = Position(
                        side=pending_entry.side,
                        units=units,
                        entry_time=bar.DateTime,
                        entry_price=entry_price,
                        stop_loss=stop,
                        take_profit=take_profit,
                        force_exit_time=pending_entry.force_exit_time,
                        entry_commission=entry_commission,
                        break_even_trigger=pending_entry.break_even_trigger,
                        break_even_stop=pending_entry.break_even_stop,
                        source=pending_entry.source,
                        probability=pending_entry.probability,
                        initial_stop_loss=stop,
                    )
            pending_entry = None

        if position is not None:
            exit_candidate = _exit_from_bar(
                position, bar, config.execution.same_bar_exit_priority
            )
            if exit_candidate is not None:
                raw_exit_price, exit_reason = exit_candidate
                fill_price = apply_exit_slippage(position.side, raw_exit_price, slip)
                exit_commission = commission_for_units(
                    position.units, config.execution, config.instrument
                )
                direction = 1 if position.side == "long" else -1
                gross_pnl = (
                    (fill_price - position.entry_price)
                    * direction
                    * position.units
                    * config.instrument.cfd_point_value
                )
                total_commission = position.entry_commission + exit_commission
                cash_pnl = gross_pnl - exit_commission
                net_pnl = gross_pnl - total_commission
                cash += cash_pnl
                capital_at_risk = (
                    position.entry_price
                    * position.units
                    * config.instrument.cfd_point_value
                )
                trades.append(
                    Trade(
                        entry_time=position.entry_time,
                        exit_time=bar.DateTime,
                        side=position.side,
                        units=position.units,
                        entry_price=position.entry_price,
                        exit_price=fill_price,
                        gross_pnl=gross_pnl,
                        commission=total_commission,
                        net_pnl=net_pnl,
                        return_pct=net_pnl / capital_at_risk
                        if capital_at_risk > 0
                        else 0.0,
                        exit_reason=exit_reason,
                        source=position.source,
                        probability=position.probability,
                    )
                )
                position = None

        signal = _normalize_signal(signal_func(i, bar, bars, position))
        if signal is not None and signal.session is None and hasattr(bar, "Session"):
            signal = replace(signal, session=bar.Session)
        if position is None and pending_entry is None and signal is not None and i + 1 < len(bars):
            pending_entry = signal

        unrealized = 0.0
        if position is not None:
            unrealized = _position_value(
                position, float(bar.Close), config.instrument.cfd_point_value
            )
        equity_rows.append(
            {
                "DateTime": bar.DateTime,
                "Cash": cash,
                "Equity": cash + unrealized,
                "InPosition": 1 if position is not None else 0,
            }
        )

    if position is not None:
        final_bar = bars.iloc[-1]
        fill_price = apply_exit_slippage(position.side, float(final_bar.Close), slip)
        exit_commission = commission_for_units(position.units, config.execution, config.instrument)
        direction = 1 if position.side == "long" else -1
        gross_pnl = (
            (fill_price - position.entry_price)
            * direction
            * position.units
            * config.instrument.cfd_point_value
        )
        cash += gross_pnl - exit_commission
        total_commission = position.entry_commission + exit_commission
        net_pnl = gross_pnl - total_commission
        capital_at_risk = (
            position.entry_price
            * position.units
            * config.instrument.cfd_point_value
        )
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=final_bar.DateTime,
                side=position.side,
                units=position.units,
                entry_price=position.entry_price,
                exit_price=fill_price,
                gross_pnl=gross_pnl,
                commission=total_commission,
                net_pnl=net_pnl,
                return_pct=net_pnl / capital_at_risk if capital_at_risk > 0 else 0.0,
                exit_reason="final_bar",
                source=position.source,
                probability=position.probability,
            )
        )
        equity_rows[-1]["Cash"] = cash
        equity_rows[-1]["Equity"] = cash
        equity_rows[-1]["InPosition"] = 0

    equity_curve = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame([trade.__dict__ for trade in trades])
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "side",
                "units",
                "entry_price",
                "exit_price",
                "gross_pnl",
                "commission",
                "net_pnl",
                "return_pct",
                "exit_reason",
                "source",
                "probability",
            ]
        )
    trades_df = trades_df.rename(
        columns={
            "entry_time": "EntryTime",
            "exit_time": "ExitTime",
            "side": "Side",
            "units": "Units",
            "entry_price": "EntryPrice",
            "exit_price": "ExitPrice",
            "gross_pnl": "GrossPnL",
            "commission": "Commission",
            "net_pnl": "NetPnL",
            "return_pct": "ReturnPct",
            "exit_reason": "ExitReason",
            "source": "Source",
            "probability": "Probability",
        }
    )
    metrics = calculate_metrics(equity_curve, trades_df, initial_cash)
    return BacktestResult(
        name=name,
        metrics=metrics,
        equity_curve=equity_curve,
        trades=trades_df,
        config_summary={
            "symbol": config.instrument.symbol,
            "starting_cash": initial_cash,
            "slippage_points_per_side": slip,
            "commission_per_cfd_unit_side": commission_for_units(
                1, config.execution, config.instrument
            ),
            "sizing_mode": config.sizing.mode,
            "fixed_units": config.sizing.fixed_units,
        },
    )
