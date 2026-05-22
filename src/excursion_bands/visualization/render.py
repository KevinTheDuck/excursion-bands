from collections.abc import Sequence
import statistics

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import polars as pl


SESSION_COLORS = {
    "pre_target_1": "#7DA6FF",
    "pre_target_2": "#6FD3C1",
    "target_1": "#8BCF7B",
    "target_2": "#F3A65A",
}


def _require_columns(df: pl.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _infer_candle_width(
    x: Sequence[float],
    width_ratio: float = 0.8,
    fallback: float = 0.018,
) -> float:
    if len(x) < 2:
        return fallback

    diffs = [
        delta
        for delta in (b - a for a, b in zip(x, x[1:], strict=False))
        if delta > 0
    ]
    if not diffs:
        return fallback

    return statistics.median(diffs) * width_ratio


def plot_session_candles(
    df: pl.DataFrame,
    ax=None,
    candle_width: float | None = None,
):
    """
    Plot one-session OHLC candles colored by intraday session bucket.

    Parameters
    ----------
    candle_width : float | None
        Width of each candle body in matplotlib date units. If omitted,
        it is inferred from the median bar spacing.
    """
    _require_columns(
        df,
        ["DateTime", "Open", "High", "Low", "Close", "Intraday_Session"],
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(16, 8))

    pdf = df.to_pandas()
    pdf = pdf.sort_values("DateTime")
    x = mdates.date2num(pdf["DateTime"].tolist())
    if candle_width is None:
        candle_width = _infer_candle_width(x)

    used_labels: set[str] = set()
    for idx, row in enumerate(pdf.itertuples(index=False)):
        session_name = row.Intraday_Session
        color = SESSION_COLORS.get(session_name, "#7f7f7f")
        edge_color = "#2B2F36"

        open_price = float(row.Open)
        close_price = float(row.Close)
        high_price = float(row.High)
        low_price = float(row.Low)

        lower_body = min(open_price, close_price)
        body_height = abs(close_price - open_price)
        body_height = body_height if body_height > 0 else 0.01

        ax.vlines(x[idx], low_price, high_price, color=edge_color, linewidth=1.0, zorder=2)

        label = session_name if session_name not in used_labels else None
        used_labels.add(session_name)
        rect = Rectangle(
            (x[idx] - candle_width / 2, lower_body),
            candle_width,
            body_height,
            facecolor=color,
            edgecolor=edge_color,
            linewidth=0.8,
            alpha=0.85,
            label=label,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_facecolor("#FCFCFD")
    ax.grid(True, linestyle="--", alpha=0.18, color="#AAB2BF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#D3D8E0")
    ax.spines["bottom"].set_color("#D3D8E0")
    return ax


def plot_overlays(ax, overlays: list[dict]) -> None:
    """
    Plot overlay specifications on an existing axis.
    """
    for overlay in overlays:
        kind = overlay["kind"]
        if kind == "hspan":
            ax.axhspan(
                overlay["y_min"],
                overlay["y_max"],
                color=overlay.get("color", "black"),
                alpha=overlay.get("alpha", 0.12),
                label=overlay.get("label"),
                zorder=0,
            )
        elif kind == "hline":
            ax.axhline(
                overlay["y"],
                color=overlay.get("color", "black"),
                linestyle=overlay.get("linestyle", "-"),
                linewidth=overlay.get("linewidth", 1.0),
                label=overlay.get("label"),
                alpha=overlay.get("alpha", 0.9),
                zorder=1,
            )
        elif kind == "line":
            ax.plot(
                overlay["x"],
                overlay["y"],
                color=overlay.get("color", "black"),
                linestyle=overlay.get("linestyle", "-"),
                linewidth=overlay.get("linewidth", 1.2),
                label=overlay.get("label"),
                alpha=overlay.get("alpha", 0.9),
                zorder=4,
            )
        else:
            raise ValueError(f"Unsupported overlay kind '{kind}'")


def render_session_chart(
    price_df: pl.DataFrame,
    overlays: list[dict] | None = None,
    title: str | None = None,
    ax=None,
):
    """
    Render a single-session chart with candles and optional overlays.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 8))
    else:
        fig = ax.figure

    ax = plot_session_candles(price_df, ax=ax)

    if overlays:
        plot_overlays(ax, overlays)

    if title is not None:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels, strict=False):
            if label and label not in unique:
                unique[label] = handle
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=9)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax
