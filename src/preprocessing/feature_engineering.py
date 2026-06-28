"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Feature Engineering
Python  : 3.11
=========================================================

This module creates machine learning features
for demand forecasting.

Features:
1. Date Features
2. Weekend Features
3. Lag Features
4. Rolling Features
5. Trend Features
6. Cyclical Features
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from config import (
    DATE_COLUMN,
    TARGET_COLUMN,
)

# ==========================================================
# Logger
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Sort Dataset
# ==========================================================

def sort_dataset(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Sort dataset chronologically.
    """

    logger.info("Sorting dataset...")

    df = df.sort_values(
        by=DATE_COLUMN
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    logger.info("Sorting completed.")

    return df


# ==========================================================
# Create Date Features
# ==========================================================

def create_date_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create date based features.
    """

    logger.info("Creating date features...")

    df["year"] = df[DATE_COLUMN].dt.year

    df["month"] = df[DATE_COLUMN].dt.month

    df["quarter"] = df[DATE_COLUMN].dt.quarter

    df["week"] = (
        df[DATE_COLUMN]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["day"] = df[DATE_COLUMN].dt.day

    df["day_of_week"] = (
        df[DATE_COLUMN]
        .dt
        .dayofweek
    )

    df["day_name"] = (
        df[DATE_COLUMN]
        .dt
        .day_name()
    )

    df["day_of_year"] = (
        df[DATE_COLUMN]
        .dt
        .dayofyear
    )

    df["month_name"] = (
        df[DATE_COLUMN]
        .dt
        .month_name()
    )

    logger.info(
        "Date features created."
    )

    return df


# ==========================================================
# Weekend Feature
# ==========================================================

def create_weekend_feature(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create weekend indicator.
    """

    logger.info(
        "Creating weekend feature..."
    )

    df["is_weekend"] = (

        df["day_of_week"]

        >= 5

    ).astype(int)

    logger.info(
        "Weekend feature created."
    )

    return df


# ==========================================================
# Month Start / End
# ==========================================================

def create_month_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Month start and month end.
    """

    logger.info(
        "Creating month features..."
    )

    df["is_month_start"] = (
        df[DATE_COLUMN]
        .dt
        .is_month_start
        .astype(int)
    )

    df["is_month_end"] = (
        df[DATE_COLUMN]
        .dt
        .is_month_end
        .astype(int)
    )

    return df


# ==========================================================
# Quarter Start / End
# ==========================================================

def create_quarter_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Quarter indicators.
    """

    logger.info(
        "Creating quarter features..."
    )

    df["is_quarter_start"] = (

        df[DATE_COLUMN]

        .dt

        .is_quarter_start

        .astype(int)

    )

    df["is_quarter_end"] = (

        df[DATE_COLUMN]

        .dt

        .is_quarter_end

        .astype(int)

    )

    return df


# ==========================================================
# Year Start / End
# ==========================================================

def create_year_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Year indicators.
    """

    logger.info(
        "Creating year features..."
    )

    df["is_year_start"] = (

        df[DATE_COLUMN]

        .dt

        .is_year_start

        .astype(int)

    )

    df["is_year_end"] = (

        df[DATE_COLUMN]

        .dt

        .is_year_end

        .astype(int)

    )

    return df


# ==========================================================
# Season Feature
# ==========================================================

def create_season(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create season from month.
    """

    logger.info(
        "Creating season..."
    )

    season_map = {

        12: "Winter",
        1: "Winter",
        2: "Winter",

        3: "Summer",
        4: "Summer",
        5: "Summer",

        6: "Monsoon",
        7: "Monsoon",
        8: "Monsoon",
        9: "Monsoon",

        10: "PostMonsoon",
        11: "PostMonsoon"

    }

    df["derived_season"] = (

        df["month"]

        .map(season_map)

    )

    logger.info(
        "Season feature created."
    )

    return df


# ==========================================================
# Dataset Summary
# ==========================================================

def feature_summary(
    df: pd.DataFrame
) -> None:

    logger.info("=" * 60)

    logger.info(
        "Current Dataset Shape : %s",
        df.shape
    )

    logger.info("=" * 60)


if __name__ == "__main__":

    logger.info(
        "Feature Engineering Module Loaded."
    )

# ==========================================================
# Lag Features
# ==========================================================

def create_lag_features(
    df: pd.DataFrame,
    group_column: str = "product_id"
) -> pd.DataFrame:
    """
    Create lag features for each product.

    Parameters
    ----------
    df : pd.DataFrame
    group_column : str
    """

    logger.info("Creating lag features...")

    df = df.sort_values(
        [group_column, DATE_COLUMN]
    )

    lags = [1, 7, 14, 30]

    for lag in lags:

        column_name = f"lag_{lag}"

        df[column_name] = (

            df.groupby(group_column)[TARGET_COLUMN]

            .shift(lag)

        )

        logger.info(
            "%s created.",
            column_name
        )

    return df


# ==========================================================
# Rolling Mean
# ==========================================================

def create_rolling_mean(
    df: pd.DataFrame,
    group_column: str = "product_id"
) -> pd.DataFrame:

    logger.info(
        "Creating rolling mean..."
    )

    windows = [7, 14, 30]

    for window in windows:

        column_name = f"rolling_mean_{window}"

        df[column_name] = (

            df.groupby(group_column)[TARGET_COLUMN]

            .transform(

                lambda x:

                x.rolling(
                    window=window,
                    min_periods=1
                ).mean()

            )

        )

        logger.info(
            "%s created.",
            column_name
        )

    return df


# ==========================================================
# Rolling Standard Deviation
# ==========================================================

def create_rolling_std(
    df: pd.DataFrame,
    group_column: str = "product_id"
) -> pd.DataFrame:

    logger.info(
        "Creating rolling std..."
    )

    windows = [7, 14, 30]

    for window in windows:

        column = f"rolling_std_{window}"

        df[column] = (

            df.groupby(group_column)[TARGET_COLUMN]

            .transform(

                lambda x:

                x.rolling(
                    window=window,
                    min_periods=1
                ).std()

            )

        )

    return df


# ==========================================================
# Rolling Minimum
# ==========================================================

def create_rolling_min(
    df: pd.DataFrame,
    group_column="product_id"
):

    logger.info(
        "Creating rolling minimum..."
    )

    windows = [7, 30]

    for window in windows:

        name = f"rolling_min_{window}"

        df[name] = (

            df.groupby(group_column)[TARGET_COLUMN]

            .transform(

                lambda x:

                x.rolling(window,1).min()

            )

        )

    return df


# ==========================================================
# Rolling Maximum
# ==========================================================

def create_rolling_max(
    df,
    group_column="product_id"
):

    logger.info(
        "Creating rolling maximum..."
    )

    windows = [7,30]

    for window in windows:

        name = f"rolling_max_{window}"

        df[name] = (

            df.groupby(group_column)[TARGET_COLUMN]

            .transform(

                lambda x:

                x.rolling(window,1).max()

            )

        )

    return df


# ==========================================================
# Exponential Moving Average
# ==========================================================

def create_ema(
    df,
    group_column="product_id"
):

    logger.info(
        "Creating EMA..."
    )

    spans = [7,14,30]

    for span in spans:

        name = f"ema_{span}"

        df[name] = (

            df.groupby(group_column)[TARGET_COLUMN]

            .transform(

                lambda x:

                x.ewm(
                    span=span,
                    adjust=False
                ).mean()

            )

        )

    return df


# ==========================================================
# Expanding Mean
# ==========================================================

def create_expanding_mean(
    df,
    group_column="product_id"
):

    logger.info(
        "Creating expanding mean..."
    )

    df["expanding_mean"] = (

        df.groupby(group_column)[TARGET_COLUMN]

        .transform(

            lambda x:

            x.expanding().mean()

        )

    )

    return df


# ==========================================================
# Demand Growth
# ==========================================================

def create_growth_rate(
    df,
    group_column="product_id"
):

    logger.info(
        "Creating demand growth..."
    )

    df["demand_growth"] = (

        df.groupby(group_column)[TARGET_COLUMN]

        .pct_change()

    )

    return df


# ==========================================================
# Difference Features
# ==========================================================

def create_difference_features(
    df,
    group_column="product_id"
):

    logger.info(
        "Creating difference features..."
    )

    df["difference_1"] = (

        df.groupby(group_column)[TARGET_COLUMN]

        .diff(1)

    )

    df["difference_7"] = (

        df.groupby(group_column)[TARGET_COLUMN]

        .diff(7)

    )

    return df
# ==========================================================
# Cyclical Encoding
# ==========================================================

def create_cyclical_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Encode cyclical features using sine and cosine transformations.
    """

    logger.info("Creating cyclical features...")

    # Month
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Day of Week
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    # Day of Month
    df["day_sin_month"] = np.sin(
        2 * np.pi * df["day"] / 31
    )

    df["day_cos_month"] = np.cos(
        2 * np.pi * df["day"] / 31
    )

    logger.info("Cyclical features created.")

    return df


# ==========================================================
# Holiday Proximity
# ==========================================================

def create_holiday_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create holiday indicator features.
    """

    logger.info("Creating holiday features...")

    if "holiday_flag" in df.columns:

        df["days_from_holiday"] = (

            df["holiday_flag"]

            .rolling(
                window=7,
                min_periods=1
            )

            .sum()

        )

    logger.info("Holiday features completed.")

    return df


# ==========================================================
# Festival Features
# ==========================================================

def create_festival_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Creating festival features..."
    )

    if "festival_flag" in df.columns:

        df["festival_last_7_days"] = (

            df["festival_flag"]

            .rolling(
                window=7,
                min_periods=1
            )

            .sum()

        )

    return df


# ==========================================================
# Fill Missing Values Generated by Lags
# ==========================================================

def fill_generated_missing(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Lag and rolling features create NaN values.
    Fill them appropriately.
    """

    logger.info(
        "Handling generated missing values..."
    )

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    df[numeric_columns] = (

        df[numeric_columns]

        .fillna(0)

    )

    logger.info(
        "Generated missing values handled."
    )

    return df


# ==========================================================
# Validate Engineered Features
# ==========================================================

def validate_engineered_features(
    df: pd.DataFrame
) -> None:

    logger.info("=" * 60)

    logger.info(
        "Feature Engineering Validation"
    )

    logger.info("=" * 60)

    logger.info(
        "Rows : %d",
        df.shape[0]
    )

    logger.info(
        "Columns : %d",
        df.shape[1]
    )

    logger.info(
        "Missing Values : %d",
        df.isna().sum().sum()
    )

    logger.info("=" * 60)


# ==========================================================
# Run Complete Feature Engineering Pipeline
# ==========================================================

def run_feature_engineering(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    """

    logger.info("=" * 70)
    logger.info("START FEATURE ENGINEERING")
    logger.info("=" * 70)

    df = sort_dataset(df)

    df = create_date_features(df)

    df = create_weekend_feature(df)

    df = create_month_features(df)

    df = create_quarter_features(df)

    df = create_year_features(df)

    df = create_season(df)

    df = create_lag_features(df)

    df = create_rolling_mean(df)

    df = create_rolling_std(df)

    df = create_rolling_min(df)

    df = create_rolling_max(df)

    df = create_ema(df)

    df = create_expanding_mean(df)

    df = create_growth_rate(df)

    df = create_difference_features(df)

    df = create_cyclical_features(df)

    df = create_holiday_features(df)

    df = create_festival_features(df)

    df = fill_generated_missing(df)

    validate_engineered_features(df)

    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING COMPLETED")
    logger.info("=" * 70)

    return df


# ==========================================================
# Execute Standalone
# ==========================================================

if __name__ == "__main__":

    logger.info(
        "Feature Engineering Module Loaded."
    )