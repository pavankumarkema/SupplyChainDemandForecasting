"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Main Preprocessing Pipeline
Python  : 3.11
=========================================================
"""

import logging

from preprocessing.config import (
    RAW_DATA_FILE,
    FEATURE_ENGINEERED_DATA_FILE,
    PROCESSED_DATA_DIR,
)

from preprocessing.preprocessing import (
    load_data,
    save_dataset,
    run_preprocessing,
)

from preprocessing.feature_engineering import (
    run_feature_engineering,
)

from preprocessing.split_data import (
    run_split_pipeline,
)

from preprocessing.save_objects import (
    save_scaler,
    save_label_encoders,
    save_feature_list,
)

logger = logging.getLogger(__name__)


def main():
    """
    Execute the complete preprocessing pipeline.
    """

    logger.info("Starting preprocessing pipeline...")

    # Load dataset
    df = load_data(RAW_DATA_FILE)

    # Data preprocessing
    df, label_encoders, scaler = run_preprocessing(df)

    # Feature engineering
    df = run_feature_engineering(df)

    # Save engineered dataset
    save_dataset(
        df,
        FEATURE_ENGINEERED_DATA_FILE
    )

    # Save preprocessing artifacts
    save_scaler(scaler)
    save_label_encoders(label_encoders)
    save_feature_list(df.columns.tolist())

    # Time-series train/validation/test split
    run_split_pipeline(
        df,
        PROCESSED_DATA_DIR
    )

    logger.info("Preprocessing pipeline completed successfully.")

    return df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline execution failed.")
        raise