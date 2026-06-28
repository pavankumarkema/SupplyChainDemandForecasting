"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Save & Load Objects
Python  : 3.11
=========================================================

This module is responsible for saving and loading
machine learning objects.

Objects include:

• StandardScaler
• Label Encoders
• Feature List
• Trained Models (Future)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from config import (
    ARTIFACTS_DIR,
    SCALER_FILE,
    LABEL_ENCODER_FILE,
    FEATURE_LIST_FILE,
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
# Create Artifacts Directory
# ==========================================================

def create_artifacts_directory() -> None:
    """
    Create artifacts directory if it does not exist.
    """

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        "Artifacts directory ready."
    )


# ==========================================================
# Generic Save Object
# ==========================================================

def save_object(
    obj: Any,
    file_path: str | Path
) -> None:
    """
    Save any Python object using Joblib.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        obj,
        file_path
    )

    logger.info(
        "Saved object -> %s",
        file_path
    )


# ==========================================================
# Generic Load Object
# ==========================================================

def load_object(
    file_path: str | Path
) -> Any:
    """
    Load any Joblib object.
    """

    file_path = Path(file_path)

    if not file_path.exists():

        raise FileNotFoundError(
            f"{file_path} not found."
        )

    obj = joblib.load(
        file_path
    )

    logger.info(
        "Loaded object -> %s",
        file_path
    )

    return obj


# ==========================================================
# Save Scaler
# ==========================================================

def save_scaler(
    scaler
) -> None:
    """
    Save fitted StandardScaler.
    """

    save_object(
        scaler,
        SCALER_FILE
    )

    logger.info(
        "Scaler saved successfully."
    )


# ==========================================================
# Load Scaler
# ==========================================================

def load_scaler():

    return load_object(
        SCALER_FILE
    )


# ==========================================================
# Save Label Encoders
# ==========================================================

def save_label_encoders(
    encoders
) -> None:

    save_object(
        encoders,
        LABEL_ENCODER_FILE
    )

    logger.info(
        "Label Encoders saved."
    )


# ==========================================================
# Load Label Encoders
# ==========================================================

def load_label_encoders():

    return load_object(
        LABEL_ENCODER_FILE
    )


# ==========================================================
# Save Feature List
# ==========================================================

def save_feature_list(
    feature_list
) -> None:

    save_object(
        feature_list,
        FEATURE_LIST_FILE
    )

    logger.info(
        "Feature list saved."
    )


# ==========================================================
# Load Feature List
# ==========================================================

def load_feature_list():

    return load_object(
        FEATURE_LIST_FILE
    )


# ==========================================================
# Save Trained Model
# ==========================================================

def save_model(
    model,
    model_name: str
) -> None:
    """
    Save trained ML model.
    """

    model_path = (
        ARTIFACTS_DIR /
        f"{model_name}.pkl"
    )

    save_object(
        model,
        model_path
    )

    logger.info(
        "%s model saved.",
        model_name
    )


# ==========================================================
# Load Trained Model
# ==========================================================

def load_model(
    model_name: str
):

    model_path = (
        ARTIFACTS_DIR /
        f"{model_name}.pkl"
    )

    return load_object(
        model_path
    )


# ==========================================================
# List Saved Artifacts
# ==========================================================

def list_artifacts() -> None:
    """
    Display all saved artifacts.
    """

    logger.info("=" * 60)

    logger.info("Saved Artifacts")

    logger.info("=" * 60)

    for file in ARTIFACTS_DIR.glob("*"):

        logger.info(file.name)

    logger.info("=" * 60)


# ==========================================================
# Delete Artifact
# ==========================================================

def delete_artifact(
    filename: str
) -> None:
    """
    Delete artifact.
    """

    file_path = (
        ARTIFACTS_DIR /
        filename
    )

    if file_path.exists():

        file_path.unlink()

        logger.info(
            "%s deleted.",
            filename
        )

    else:

        logger.warning(
            "%s not found.",
            filename
        )


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    create_artifacts_directory()

    list_artifacts()

    logger.info(
        "Save Objects Module Loaded Successfully."
    )
