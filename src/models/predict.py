"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Prediction
Python  : 3.11
=========================================================

Load trained model and predict future demand.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import (
    PREDICTIONS_FILE,
)

from .save_model import (
    load_model,
)

logger = logging.getLogger(__name__)
def predict(
    model,
    X: pd.DataFrame
):
    """
    Predict demand.
    """

    logger.info("Generating predictions...")

    predictions = model.predict(X)

    logger.info("Prediction completed.")

    return predictions
def save_predictions(
    predictions,
    output_file=PREDICTIONS_FILE
):
    """
    Save predictions to CSV.
    """

    prediction_df = pd.DataFrame(
        {
            "Predicted_Units_Sold": predictions
        }
    )

    prediction_df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        "Predictions saved to %s",
        output_file
    )

    return prediction_df
def run_prediction_pipeline(
    X_test: pd.DataFrame
):
    """
    Complete prediction pipeline.
    """

    logger.info(
        "Starting prediction pipeline..."
    )

    model = load_model()

    predictions = predict(
        model,
        X_test
    )

    prediction_df = save_predictions(
        predictions
    )

    logger.info(
        "Prediction pipeline completed."
    )

    return prediction_df
def predict_single_sample(
    sample: pd.DataFrame
):
    """
    Predict a single sample.
    """

    model = load_model()

    prediction = model.predict(sample)

    return prediction[0]
if __name__ == "__main__":

    logger.info(
        "Prediction Module Loaded."
    )