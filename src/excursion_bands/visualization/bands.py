import polars as pl

from excursion_bands.visualization.overlays import (
    build_band_overlay_row,
    build_horizontal_band_overlays,
)
from excursion_bands.visualization.render import render_session_chart
from excursion_bands.visualization.session import infer_timeframe_label, select_session


def plot_excursion_band_session(
    df: pl.DataFrame,
    df_bands: pl.DataFrame,
    session: str | None = None,
    timeframe_label: str | None = None,
    show_centers: bool = True,
    show_ae: bool = True,
    show_fe: bool = True,
    overlays: list[dict] | None = None,
):
    """
    Plot one trading session of candles with excursion band overlays.

    Parameters
    ----------
    df : pl.DataFrame
        Processed intraday dataframe (any timeframe) containing one or more sessions.

    df_bands : pl.DataFrame
        Session-level dataframe containing excursion band columns.

    session : str | None, default=None
        Session to visualize. If omitted, the latest available session is used.

    timeframe_label : str | None, default=None
        Optional label (e.g., "5m", "1h") to display in the chart title.
        If omitted, it is inferred from the session data when possible.

    show_centers : bool, default=True
        Whether to plot band center levels.

    show_ae : bool, default=True
        Whether to plot AE levels.

    show_fe : bool, default=True
        Whether to plot FE levels.

    overlays : list[dict] | None, default=None
        Additional generic overlay specifications to render.

    Returns
    -------
    tuple
        Matplotlib `(fig, ax)` for the rendered session chart.
    """
    session_df = select_session(df, session=session)
    active_session = str(session_df["Session"][0])

    band_row = build_band_overlay_row(df_bands, session=active_session)
    band_overlays = build_horizontal_band_overlays(
        band_row,
        show_centers=show_centers,
        show_ae=show_ae,
        show_fe=show_fe,
    )

    extra_overlays = overlays or []
    inferred_label = timeframe_label or infer_timeframe_label(session_df)
    if inferred_label:
        title = f"Excursion Bands | Session {active_session} | {inferred_label}"
    else:
        title = f"Excursion Bands | Session {active_session}"
    return render_session_chart(
        price_df=session_df,
        overlays=band_overlays + extra_overlays,
        title=title,
    )
