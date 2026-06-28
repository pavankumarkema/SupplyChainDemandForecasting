"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Data Preprocessing
Python  : 3.11
=========================================================

This module contains reusable preprocessing functions
used throughout the project.

Functions included:

1. Load Dataset
2. Save Dataset
3. Standardize Column Names
4. Convert Data Types
5. Missing Value Handling
6. Duplicate Handling
7. Data Validation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from config import (
    DATE_COLUMN,
    NUMERICAL_COLUMNS,
    LABEL_ENCODING_COLUMNS,
    ONE_HOT_COLUMNS,
)

# ==========================================================
# Logger Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================================================
# Load Dataset
# ==========================================================

def load_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load CSV dataset.

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    pd.DataFrame
    """

    logger.info("Loading dataset...")

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    logger.info(
        "Dataset Loaded Successfully"
    )

    logger.info(
        "Shape : %s",
        df.shape
    )

    return df

# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(
    df: pd.DataFrame,
    output_path: str | Path
) -> None:
    """
    Save dataframe as CSV.

    Parameters
    ----------
    df : DataFrame

    output_path : Path
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        "Dataset saved to:\n%s",
        output_path
    )

# ==========================================================
# Standardize Column Names
# ==========================================================

def standardize_column_names(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert column names into
    lowercase snake_case.
    """

    logger.info(
        "Standardizing column names..."
    )

    df.columns = (

        df.columns

        .str.strip()

        .str.lower()

        .str.replace(" ", "_")

        .str.replace("-", "_")

        .str.replace("/", "_")

    )

    logger.info(
        "Column names standardized."
    )

    return df

# ==========================================================
# Convert Date Column
# ==========================================================

def convert_date_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert date column into datetime.
    """

    logger.info(
        "Converting date column..."
    )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN]
    )

    logger.info(
        "Date conversion completed."
    )

    return df

# ==========================================================
# Display Dataset Information
# ==========================================================

def dataset_summary(
    df: pd.DataFrame
) -> None:

    logger.info("=" * 60)

    logger.info("Dataset Summary")

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
        "\n%s",
        df.dtypes
    )

# ==========================================================
# Check Missing Values
# ==========================================================

def missing_value_summary(
    df: pd.DataFrame
) -> pd.Series:
    """
    Return missing value count.
    """

    logger.info(
        "Checking missing values..."
    )

    missing = df.isna().sum()

    logger.info("\n%s", missing)

    return missing

# ==========================================================
# Missing Percentage
# ==========================================================

def missing_percentage(
    df: pd.DataFrame
) -> pd.Series:

    percent = (

        df.isna().sum()

        / len(df)

        * 100

    )

    logger.info(
        "\n%s",
        percent
    )

    return percent

# ==========================================================
# Duplicate Rows
# ==========================================================

def duplicate_summary(
    df: pd.DataFrame
) -> int:

    duplicates = df.duplicated().sum()

    logger.info(
        "Duplicate Rows : %d",
        duplicates
    )

    return duplicates

# ==========================================================
# Remove Duplicates
# ==========================================================

def remove_duplicates(
    df: pd.DataFrame
) -> pd.DataFrame:

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    logger.info(
        "Removed %d duplicate rows.",
        before - after
    )

    return df

# ==========================================================
# Validate Numerical Columns
# ==========================================================

def validate_numerical_columns(
    df: pd.DataFrame
) -> None:
    """
    Validate whether all numerical columns exist.

    Parameters
    ----------
    df : pd.DataFrame
    """

    logger.info("Validating numerical columns...")

    missing_columns = []

    for column in NUMERICAL_COLUMNS:

        if column not in df.columns:

            missing_columns.append(column)

    if missing_columns:

        logger.warning(
            "Missing numerical columns: %s",
            missing_columns
        )

    else:

        logger.info(
            "All numerical columns are available."
        )


# ==========================================================
# Fill Missing Numerical Values
# ==========================================================

def fill_missing_numerical(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill numerical missing values using median.
    """

    logger.info(
        "Handling numerical missing values..."
    )

    for column in NUMERICAL_COLUMNS:

        if column in df.columns:

            if df[column].isnull().sum() > 0:

                median = df[column].median()

                df[column].fillna(
                    median,
                    inplace=True
                )

                logger.info(
                    "%s -> Filled using median",
                    column
                )

    return df


# ==========================================================
# Fill Missing Categorical Values
# ==========================================================

def fill_missing_categorical(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Fill categorical missing values using mode.
    """

    logger.info(
        "Handling categorical missing values..."
    )

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        if df[column].isnull().sum() > 0:

            mode = df[column].mode()[0]

            df[column].fillna(
                mode,
                inplace=True
            )

            logger.info(
                "%s -> Filled using mode",
                column
            )

    return df


# ==========================================================
# Validate Business Rules
# ==========================================================

def validate_business_rules(
    df: pd.DataFrame
) -> None:
    """
    Validate supply chain business rules.
    """

    logger.info(
        "Running business rule validation..."
    )

    rules = {

        "Negative Selling Price":
        (df["selling_price"] <= 0).sum(),

        "Negative Units Sold":
        (df["units_sold"] < 0).sum(),

        "Negative Inventory":
        (df["inventory_level"] < 0).sum(),

        "Negative Shipping Cost":
        (df["shipping_cost"] < 0).sum(),

        "Negative Procurement Cost":
        (df["procurement_cost"] < 0).sum(),

        "Invalid Discount":
        (
            (df["discount_percent"] < 0)
            |
            (df["discount_percent"] > 100)
        ).sum(),

        "Negative Lead Time":
        (
            df["supplier_lead_time_days"] <= 0
        ).sum()

    }

    logger.info("=" * 50)

    logger.info("Business Rule Report")

    logger.info("=" * 50)

    for rule, violations in rules.items():

        logger.info(
            "%-30s : %d",
            rule,
            violations
        )


# ==========================================================
# Remove Impossible Records
# ==========================================================

def remove_invalid_records(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove impossible records.
    """

    logger.info(
        "Removing invalid records..."
    )

    before = len(df)

    df = df[
        (df["selling_price"] > 0)
    ]

    df = df[
        (df["inventory_level"] >= 0)
    ]

    df = df[
        (df["shipping_cost"] >= 0)
    ]

    df = df[
        (df["procurement_cost"] >= 0)
    ]

    df = df[
        (df["discount_percent"] >= 0)
    ]

    df = df[
        (df["discount_percent"] <= 100)
    ]

    after = len(df)

    logger.info(
        "Removed %d invalid rows.",
        before - after
    )

    return df


# ==========================================================
# Validate Order IDs
# ==========================================================

def validate_order_ids(
    df: pd.DataFrame
) -> None:
    """
    Validate duplicate Order IDs.
    """

    duplicates = (
        df["order_id"]
        .duplicated()
        .sum()
    )

    logger.info(
        "Duplicate Order IDs : %d",
        duplicates
    )


# ==========================================================
# Validate Product Mapping
# ==========================================================

def validate_product_mapping(
    df: pd.DataFrame
) -> None:
    """
    Validate Product ID ↔ Product Name mapping.
    """

    mapping = (

        df.groupby("product_id")

        ["product_name"]

        .nunique()

    )

    inconsistent = mapping[mapping > 1]

    if len(inconsistent) == 0:

        logger.info(
            "Product mapping validation passed."
        )

    else:

        logger.warning(
            "Found inconsistent Product IDs."
        )


# ==========================================================
# Dataset Statistics
# ==========================================================

def numerical_statistics(
    df: pd.DataFrame
) -> None:

    logger.info("=" * 60)

    logger.info(
        "\n%s",
        df[NUMERICAL_COLUMNS].describe()
    )

    # ==========================================================
# Detect Outliers using IQR
# ==========================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Detect outliers using IQR method.

    Parameters
    ----------
    df : pd.DataFrame
    column : str

    Returns
    -------
    pd.DataFrame
    """

    if column not in df.columns:
        raise ValueError(f"{column} not found.")

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR

    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower) |
        (df[column] > upper)
    ]

    logger.info(
        "%s -> %d outliers detected.",
        column,
        len(outliers)
    )

    return outliers


# ==========================================================
# Label Encoding
# ==========================================================

def apply_label_encoding(
    df: pd.DataFrame
):
    """
    Apply Label Encoding.

    Returns
    -------
    DataFrame
    Encoders Dictionary
    """

    logger.info("Applying Label Encoding...")

    encoders = {}

    for column in LABEL_ENCODING_COLUMNS:

        if column in df.columns:

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(
                df[column].astype(str)
            )

            encoders[column] = encoder

            logger.info(
                "%s encoded.",
                column
            )

    return df, encoders


# ==========================================================
# One Hot Encoding
# ==========================================================

def apply_one_hot_encoding(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply One Hot Encoding.
    """

    logger.info(
        "Applying One Hot Encoding..."
    )

    columns = [

        c for c in ONE_HOT_COLUMNS

        if c in df.columns

    ]

    df = pd.get_dummies(

        df,

        columns=columns,

        drop_first=True,

        dtype=int

    )

    logger.info(
        "One Hot Encoding completed."
    )

    return df


# ==========================================================
# Feature Scaling
# ==========================================================

def apply_standard_scaling(
    df: pd.DataFrame
):
    """
    Standardize numerical features.

    Returns
    -------
    DataFrame
    Scaler
    """

    logger.info(
        "Applying StandardScaler..."
    )

    scaler = StandardScaler()

    available_columns = [

        c for c in NUMERICAL_COLUMNS

        if c in df.columns

    ]

    df[available_columns] = scaler.fit_transform(

        df[available_columns]

    )

    logger.info(
        "Scaling completed."
    )

    return df, scaler


# ==========================================================
# Validate Final Dataset
# ==========================================================

def validate_dataset(
    df: pd.DataFrame
) -> None:
    """
    Final validation before ML.
    """

    logger.info("=" * 60)

    logger.info("FINAL DATASET VALIDATION")

    logger.info("=" * 60)

    logger.info(
        "Rows    : %d",
        df.shape[0]
    )

    logger.info(
        "Columns : %d",
        df.shape[1]
    )

    missing = df.isna().sum().sum()

    logger.info(
        "Missing Values : %d",
        missing
    )

    duplicates = df.duplicated().sum()

    logger.info(
        "Duplicate Rows : %d",
        duplicates
    )

    logger.info("=" * 60)


# ==========================================================
# Complete Preprocessing Pipeline
# ==========================================================

def run_preprocessing(
    df: pd.DataFrame
):
    """
    Complete preprocessing pipeline.

    Returns
    -------
    DataFrame
    Label Encoders
    Scaler
    """

    logger.info("=" * 70)
    logger.info("STARTING PREPROCESSING PIPELINE")
    logger.info("=" * 70)

    dataset_summary(df)

    df = standardize_column_names(df)

    df = convert_date_column(df)

    duplicate_summary(df)

    df = remove_duplicates(df)

    missing_value_summary(df)

    fill_missing_numerical(df)

    fill_missing_categorical(df)

    validate_numerical_columns(df)

    validate_business_rules(df)

    df = remove_invalid_records(df)

    validate_order_ids(df)

    validate_product_mapping(df)

    numerical_statistics(df)

    df, encoders = apply_label_encoding(df)

    df = apply_one_hot_encoding(df)

    df, scaler = apply_standard_scaling(df)

    validate_dataset(df)

    logger.info("=" * 70)
    logger.info("PREPROCESSING COMPLETED")
    logger.info("=" * 70)

    return (

        df,

        encoders,

        scaler

    )


# ==========================================================
# Execute Standalone
# ==========================================================

if __name__ == "__main__":

    logger.info(
        "Preprocessing module loaded successfully."
    )

