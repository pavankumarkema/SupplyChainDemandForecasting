"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Machine Learning Pipeline
Python  : 3.11
=========================================================

Pipeline

Load Dataset
      ↓
Train Models
      ↓
Evaluate Models
      ↓
Select Best Model
      ↓
Save Model
      ↓
Predict Test Dataset
"""

from __future__ import annotations

import logging

from .train import (
    run_training_pipeline,
)

from .evaluate import (
    run_evaluation_pipeline,
)

from .model_selection import (
    run_model_selection_pipeline,
)

from .save_model import (
    run_save_pipeline,
)

from .predict import (
    run_prediction_pipeline,
)

logger = logging.getLogger(__name__)


def main():
    """
    Complete Machine Learning Pipeline.
    """

    logger.info("=" * 70)
    logger.info("MACHINE LEARNING PIPELINE STARTED")
    logger.info("=" * 70)

    # =====================================================
    # Train Models
    # =====================================================

    (
        trained_models,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = run_training_pipeline()

    # =====================================================
    # Evaluate Models
    # =====================================================

    results = run_evaluation_pipeline(
        trained_models,
        X_test,
        y_test,
    )

    # =====================================================
    # Select Best Model
    # =====================================================

    (
        best_model_name,
        best_model,
    ) = run_model_selection_pipeline(
        trained_models,
        results,
    )

    # =====================================================
    # Save Best Model
    # =====================================================

    run_save_pipeline(
        best_model,
        best_model_name,
        X_test.columns.tolist(),
    )

    # =====================================================
    # Predict
    # =====================================================

    prediction_df = run_prediction_pipeline(
        X_test,
    )

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)

    return {
        "results": results,
        "predictions": prediction_df,
        "best_model": best_model_name,
    }


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    try:

        output = main()

        print("\n")
        print("=" * 70)
        print("BEST MODEL")
        print("=" * 70)
        print(output["best_model"])

        print("\n")
        print("=" * 70)
        print("MODEL RESULTS")
        print("=" * 70)
        print(output["results"])

    except Exception:

        logger.exception(
            "Machine Learning Pipeline Failed."
        )

        raise
