import polars as pl
from excursion_bands.utils import logger

def _calculate_z_body(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute normalized session body magnitude used for direction detection.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing:
        - C_target_2: closing price of the final target session
        - O_ref: reference opening price
        - Sigma_historical: historical volatility estimate

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_z_body` added, defined as the absolute
        log body size normalized by historical volatility.
    """
    return df.with_columns(
        (
            (pl.col("C_target_2") / pl.col("O_ref")).log().abs()
            / pl.col("Sigma_historical")
        ).alias("_z_body")
    )


def _calculate_z_sigma(
    df: pl.DataFrame, n: int
) -> pl.DataFrame:
    """
    Compute relative volatility regime score.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `Sigma_historical`.

    n : int
        Rolling lookback window used to compare current volatility
        against its recent mean.

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_z_sigma` added, defined as current
        historical volatility divided by the lagged rolling mean.
    """
    return df.with_columns(
        (
            pl.col("Sigma_historical")
            / pl.col("Sigma_historical").rolling_mean(n).shift(1)
        ).alias("_z_sigma")
    )


def _calculate_threshold(
    df: pl.DataFrame,
    tau_0: float,
    tau_min: float,
    tau_max: float,
) -> pl.DataFrame:
    """
    Compute adaptive body threshold for directional classification.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `_z_sigma`.

    tau_0 : float
        Base threshold level.

    tau_min : float
        Lower bound applied to the adaptive threshold.

    tau_max : float
        Upper bound applied to the adaptive threshold.

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_tau` added.

    Notes
    -----
    Thresholds tighten or loosen with the volatility regime and are
    clipped to remain within configured bounds.
    """
    return df.with_columns(
        (tau_0 * (pl.col("_z_sigma") ** -0.5)).clip(tau_min, tau_max).alias("_tau")
    )


def _get_day_boundaries(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute full-session low and high boundaries across all buckets.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing per-session low and high columns for:
        - pre_target_1
        - pre_target_2
        - target_1
        - target_2

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_l_day` and `_h_day` added.
    """
    return df.with_columns(
        [
            pl.min_horizontal(
                pl.col("L_pre_target_1"),
                pl.col("L_pre_target_2"),
                pl.col("L_target_1"),
                pl.col("L_target_2"),
            ).alias("_l_day"),
            pl.max_horizontal(
                pl.col("H_pre_target_1"),
                pl.col("H_pre_target_2"),
                pl.col("H_target_1"),
                pl.col("H_target_2"),
            ).alias("_h_day"),
        ]
    )


def _calculate_epsilon(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute adverse and favorable excursion distances for each session.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing:
        - `_direction`
        - `O_ref`
        - `_l_day`
        - `_h_day`

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_epsilon_ae` and `_epsilon_fe` added.

    Notes
    -----
    Excursions are measured relative to `O_ref` and depend on the
    assigned directional regime:
    - bullish: downside is adverse, upside is favorable
    - bearish: upside is adverse, downside is favorable
    - neutral: both use the larger side of the daily range
    """
    return df.with_columns(
        [
            (
                pl.when(pl.col("_direction") == "bullish")
                .then(pl.col("O_ref") - pl.col("_l_day"))
                .when(pl.col("_direction") == "bearish")
                .then(pl.col("_h_day") - pl.col("O_ref"))
                .otherwise(
                    pl.max_horizontal(
                        pl.col("_h_day") - pl.col("O_ref"),
                        pl.col("O_ref") - pl.col("_l_day"),
                    )
                )
            ).alias("_epsilon_ae"),
            (
                pl.when(pl.col("_direction") == "bullish")
                .then(pl.col("_h_day") - pl.col("O_ref"))
                .when(pl.col("_direction") == "bearish")
                .then(pl.col("O_ref") - pl.col("_l_day"))
                .otherwise(
                    pl.max_horizontal(
                        pl.col("_h_day") - pl.col("O_ref"),
                        pl.col("O_ref") - pl.col("_l_day"),
                    )
                )
            ).alias("_epsilon_fe"),
        ]
    )


def _normalize_epsilon(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize excursion distances by historical volatility.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `_epsilon_ae`, `_epsilon_fe`, and
        `Sigma_historical`.

    Returns
    -------
    pl.DataFrame
        Input dataframe with normalized adverse and favorable
        excursion features added.
    """
    return df.with_columns(
        [
            (pl.col("_epsilon_ae") / pl.col("Sigma_historical")).alias(
                "_epsilon_ae_normalized"
            ),
            (pl.col("_epsilon_fe") / pl.col("Sigma_historical")).alias(
                "_epsilon_fe_normalized"
            ),
        ]
    )


def _calculate_mu(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """
    Compute rolling expected excursion centers in normalized space.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing normalized excursion features.

    n : int
        Rolling lookback window size.

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_mu_ae` and `_mu_fe` added.

    Notes
    -----
    Values are shifted by one row before aggregation so the current
    session only uses prior information.
    """
    return df.with_columns(
        [
            (pl.col("_epsilon_ae_normalized").shift(1).rolling_mean(n)).alias("_mu_ae"),
            (pl.col("_epsilon_fe_normalized").shift(1).rolling_mean(n)).alias("_mu_fe"),
        ]
    )


def _calculate_mu_scaled(df: pl.DataFrame) -> pl.DataFrame:
    """
    Scale normalized excursion centers back into price space.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `_mu_ae`, `_mu_fe`, and
        `Sigma_historical`.

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_mu_ae_scaled` and `_mu_fe_scaled` added.
    """
    return df.with_columns(
        [
            (pl.col("_mu_ae") * pl.col("Sigma_historical").shift(1)).alias(
                "_mu_ae_scaled"
            ),
            (pl.col("_mu_fe") * pl.col("Sigma_historical").shift(1)).alias(
                "_mu_fe_scaled"
            ),
        ]
    )


def _calculate_delta_t(
    df: pl.DataFrame, k: float
) -> pl.DataFrame:
    """
    Compute shared band half-width in price space.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `Sigma_historical` and `O_ref`.

    k : float
        Width multiplier applied to lagged historical volatility.

    Returns
    -------
    pl.DataFrame
        Input dataframe with `_delta_t` added.

    Notes
    -----
    `_delta_t` is used as the symmetric offset around both AE and FE
    band centers when constructing upper and lower zone boundaries.
    """
    return df.with_columns(
        [(k * pl.col("Sigma_historical").shift(1) * pl.col("O_ref")).alias("_delta_t")]
    )


def assign_direction(df: pl.DataFrame, n: int, threshold: dict) -> pl.DataFrame:
    """
    Classify each session as bullish, bearish, or neutral.

    Parameters
    ----------
    df : pl.DataFrame
        Input dataframe containing `C_target_2`, `O_ref`, and
        `Sigma_historical`.

    n : int
        Rolling lookback window used in volatility regime scoring.

    threshold : dict
        Threshold configuration containing:
        - tau_0
        - tau_min
        - tau_max

    Returns
    -------
    pl.DataFrame
        Input dataframe with direction-related intermediate features and
        `_direction` added.

    Notes
    -----
    Direction is assigned only when the normalized body magnitude
    exceeds the adaptive threshold.
    """
    _tag_str = "[features/excursion_bands/assign_direction]"
    print(logger(_tag_str, "Assigning Direction"))

    df = _calculate_z_body(df)
    df = _calculate_z_sigma(df, n)
    df = _calculate_threshold(df, threshold["tau_0"], threshold["tau_min"], threshold["tau_max"])

    return df.with_columns(
        pl.when(
            (pl.col("_z_body") > pl.col("_tau"))
            & (pl.col("C_target_2") > pl.col("O_ref"))
        )
        .then(pl.lit("bullish"))
        .when(
            (pl.col("_z_body") > pl.col("_tau"))
            & (pl.col("C_target_2") < pl.col("O_ref"))
        )
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral"))
        .alias("_direction")
    )


def calculate_excursion_bands(
    config_file: dict, df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Compute excursion band centers and zone boundaries.

    Parameters
    ----------
    config_file : dict
        Configuration containing:
        - lookback_window
        - threshold settings (`tau_0`, `tau_min`, `tau_max`, `k`)

    df : pl.DataFrame
        Input dataframe containing aggregated session features,
        `O_ref`, and `Sigma_historical`.

    Returns
    -------
    pl.DataFrame
        Input dataframe with AE and FE band center and boundary columns
        added. Intermediate helper columns prefixed with `_` are removed.

    Notes
    -----
    The pipeline performs:

    1. Direction assignment from normalized session body
    2. Session range extraction across all intraday buckets
    3. Adverse and favorable excursion measurement
    4. Normalization and rolling expectation estimation
    5. Band center construction
    6. Symmetric upper and lower boundary construction
    """
    _tag_str = "[features/excursion_bands/calculate_excursion_bands]"
    print(logger(_tag_str, "Calculating Excursion Bands"))

    n = config_file["lookback_window"]
    threshold = config_file["threshold"]

    df = assign_direction(df, n, threshold)
    df = _get_day_boundaries(df)
    df = _calculate_epsilon(df)
    df = _normalize_epsilon(df)
    df = _calculate_mu(df, n)
    df = _calculate_mu_scaled(df)
    df = _calculate_delta_t(df, threshold["k"])

    df = df.with_columns(
        [
            (pl.col("O_ref") + pl.col("_mu_ae_scaled")).alias("Band_AE_Pos_Center"),
            (pl.col("O_ref") - pl.col("_mu_ae_scaled")).alias("Band_AE_Neg_Center"),
            (pl.col("O_ref") + pl.col("_mu_fe_scaled")).alias("Band_FE_Pos_Center"),
            (pl.col("O_ref") - pl.col("_mu_fe_scaled")).alias("Band_FE_Neg_Center"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("Band_AE_Neg_Center") + pl.col("_delta_t")).alias(
                "Band_AE_Neg_Upper"
            ),
            (pl.col("Band_AE_Neg_Center") - pl.col("_delta_t")).alias(
                "Band_AE_Neg_Lower"
            ),
            (pl.col("Band_AE_Pos_Center") + pl.col("_delta_t")).alias(
                "Band_AE_Pos_Upper"
            ),
            (pl.col("Band_AE_Pos_Center") - pl.col("_delta_t")).alias(
                "Band_AE_Pos_Lower"
            ),
            (pl.col("Band_FE_Neg_Center") + pl.col("_delta_t")).alias(
                "Band_FE_Neg_Upper"
            ),
            (pl.col("Band_FE_Neg_Center") - pl.col("_delta_t")).alias(
                "Band_FE_Neg_Lower"
            ),
            (pl.col("Band_FE_Pos_Center") + pl.col("_delta_t")).alias(
                "Band_FE_Pos_Upper"
            ),
            (pl.col("Band_FE_Pos_Center") - pl.col("_delta_t")).alias(
                "Band_FE_Pos_Lower"
            ),
        ]
    )

    return df.select(pl.exclude("^_"))
