"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Save and Load Model
Python  : 3.11
=========================================================

Save and load trained machine learning models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import (
    BEST_MODEL_FILE,
    FEATURE_IMPORTANCE_FILE,
)

logger = logging.getLogger(__name__)
def save_model(
    model,
    model_name: str
) -> None:
    """
    Save the trained model.
    """

    BEST_MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        BEST_MODEL_FILE
    )

    logger.info(
        "%s saved successfully.",
        model_name
    )
def load_model():
    """
    Load saved model.
    """

    if not BEST_MODEL_FILE.exists():

        raise FileNotFoundError(
            f"{BEST_MODEL_FILE} not found."
        )

    model = joblib.load(
        BEST_MODEL_FILE
    )

    logger.info(
        "Model loaded successfully."
    )

    return model
def save_feature_names(
    feature_names: list
):
    """
    Save feature names.
    """

    output = (
        FEATURE_IMPORTANCE_FILE.parent
        / "feature_names.csv"
    )

    pd.DataFrame(
        {
            "Feature": feature_names
        }
    ).to_csv(
        output,
        index=False
    )

    logger.info(
        "Feature names saved."
    )
def save_feature_importance(
    model,
    feature_names
):
    """
    Save feature importance if supported.
    """

    if not hasattr(
        model,
        "feature_importances_"
    ):

        logger.warning(
            "Feature importance not available."
        )

        return

    importance = pd.DataFrame(

        {

            "Feature": feature_names,

            "Importance":
            model.feature_importances_

        }

    )

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    importance.to_csv(

        FEATURE_IMPORTANCE_FILE,

        index=False

    )

    logger.info(
        "Feature importance saved."
    )
def run_save_pipeline(
    model,
    model_name,
    feature_names
):
    """
    Save model and related artifacts.
    """

    save_model(
        model,
        model_name
    )

    save_feature_names(
        feature_names
    )

    save_feature_importance(
        model,
        feature_names
    )

    logger.info(
        "All artifacts saved."
    )
if __name__ == "__main__":

    logger.info(
        "Save Model Module Loaded."
    )