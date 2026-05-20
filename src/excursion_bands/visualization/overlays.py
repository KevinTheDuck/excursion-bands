from collections.abc import Sequence

import polars as pl


BAND_STYLE = {
    "Band_AE_Neg_Lower": {"color": "#5B8FF9", "linestyle": "--", "linewidth": 1.0},
    "Band_AE_Neg_Upper": {"color": "#5B8FF9", "linestyle": "--", "linewidth": 1.0},
    "Band_AE_Pos_Lower": {"color": "#45B7D1", "linestyle": "--", "linewidth": 1.0},
    "Band_AE_Pos_Upper": {"color": "#45B7D1", "linestyle": "--", "linewidth": 1.0},
    "Band_FE_Neg_Lower": {"color": "#D66B8D", "linestyle": ":", "linewidth": 1.0},
    "Band_FE_Neg_Upper": {"color": "#D66B8D", "linestyle": ":", "linewidth": 1.0},
    "Band_FE_Pos_Lower": {"color": "#F6A04D", "linestyle": ":", "linewidth": 1.0},
    "Band_FE_Pos_Upper": {"color": "#F6A04D", "linestyle": ":", "linewidth": 1.0},
    "Band_AE_Neg_Center": {"color": "#2F5FB3", "linestyle": "-", "linewidth": 1.3},
    "Band_AE_Pos_Center": {"color": "#2C8FA6", "linestyle": "-", "linewidth": 1.3},
    "Band_FE_Neg_Center": {"color": "#B24C6F", "linestyle": "-", "linewidth": 1.3},
    "Band_FE_Pos_Center": {"color": "#C97A2F", "linestyle": "-", "linewidth": 1.3},
}

ZONE_STYLE = {
    "Band_AE_Neg": {"color": "#5B8FF9", "alpha": 0.12},
    "Band_AE_Pos": {"color": "#45B7D1", "alpha": 0.12},
    "Band_FE_Neg": {"color": "#D66B8D", "alpha": 0.10},
    "Band_FE_Pos": {"color": "#F6A04D", "alpha": 0.10},
}


def build_horizontal_line_overlay(
    y: float,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.0,
) -> dict:
    """
    Build a generic horizontal line overlay specification.
    """
    return {
        "kind": "hline",
        "y": y,
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
    }


def build_line_overlay(
    x: Sequence,
    y: Sequence[float],
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.2,
) -> dict:
    """
    Build a generic line overlay specification.
    """
    return {
        "kind": "line",
        "x": list(x),
        "y": list(y),
        "label": label,
        "color": color,
        "linestyle": linestyle,
        "linewidth": linewidth,
    }


def build_horizontal_span_overlay(
    y_min: float,
    y_max: float,
    label: str,
    color: str,
    alpha: float = 0.12,
) -> dict:
    """
    Build a generic horizontal span overlay specification.
    """
    return {
        "kind": "hspan",
        "y_min": y_min,
        "y_max": y_max,
        "label": label,
        "color": color,
        "alpha": alpha,
    }


def build_band_overlay_row(df_bands: pl.DataFrame, session: str | None = None) -> dict:
    """
    Extract one session row of band values from aggregated feature data.
    """
    if "Session" not in df_bands.columns:
        raise ValueError("Expected dataframe with 'Session' column")

    target = session
    if target is None:
        target = sorted(str(value) for value in df_bands["Session"].unique().to_list())[-1]

    row_df = df_bands.filter(pl.col("Session").cast(pl.String) == target)
    if row_df.height != 1:
        raise ValueError(f"Expected exactly one band row for session '{target}', found {row_df.height}")

    return row_df.to_dicts()[0]


def build_horizontal_band_overlays(
    band_row: dict,
    show_centers: bool = True,
    show_ae: bool = True,
    show_fe: bool = True,
) -> list[dict]:
    """
    Convert one band row into horizontal overlay specifications.
    """
    overlays: list[dict] = []

    zone_columns: list[tuple[str, str, str]] = []
    if show_ae:
        zone_columns.extend(
            [
                ("Band_AE_Neg", "Band_AE_Neg_Lower", "Band_AE_Neg_Upper"),
                ("Band_AE_Pos", "Band_AE_Pos_Lower", "Band_AE_Pos_Upper"),
            ]
        )

    if show_fe:
        zone_columns.extend(
            [
                ("Band_FE_Neg", "Band_FE_Neg_Lower", "Band_FE_Neg_Upper"),
                ("Band_FE_Pos", "Band_FE_Pos_Lower", "Band_FE_Pos_Upper"),
            ]
        )

    for zone_name, lower_col, upper_col in zone_columns:
        if lower_col not in band_row or upper_col not in band_row:
            raise ValueError(f"Missing required band columns '{lower_col}'/'{upper_col}'")

        style = ZONE_STYLE[zone_name]
        overlays.append(
            build_horizontal_span_overlay(
                y_min=float(band_row[lower_col]),
                y_max=float(band_row[upper_col]),
                label=zone_name,
                color=style["color"],
                alpha=style["alpha"],
            )
        )

    columns: list[str] = []
    if show_ae:
        columns.extend(
            [
                "Band_AE_Neg_Lower",
                "Band_AE_Neg_Upper",
                "Band_AE_Pos_Lower",
                "Band_AE_Pos_Upper",
            ]
        )
        if show_centers:
            columns.extend(["Band_AE_Neg_Center", "Band_AE_Pos_Center"])

    if show_fe:
        columns.extend(
            [
                "Band_FE_Neg_Lower",
                "Band_FE_Neg_Upper",
                "Band_FE_Pos_Lower",
                "Band_FE_Pos_Upper",
            ]
        )
        if show_centers:
            columns.extend(["Band_FE_Neg_Center", "Band_FE_Pos_Center"])

    for column in columns:
        if column not in band_row:
            raise ValueError(f"Missing required band column '{column}'")

        style = BAND_STYLE[column]
        overlays.append(
            build_horizontal_line_overlay(
                y=float(band_row[column]),
                label=column,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
            )
        )

    return overlays
