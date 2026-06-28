"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Model Evaluation
Python  : 3.11
=========================================================

Evaluate regression models and compare performance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from .config import MODEL_RESULTS_FILE

logger = logging.getLogger(__name__)
def calculate_metrics(
    y_true,
    y_pred
) -> dict:
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_true,
        y_pred
    ) * 100

    return {

        "MAE": round(mae, 4),

        "MSE": round(mse, 4),

        "RMSE": round(rmse, 4),

        "R2": round(r2, 4),

        "MAPE": round(mape, 4)

    }
def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate one model.
    """

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    return metrics
def evaluate_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluate every trained model.
    """

    results = []

    for model_name, model in trained_models.items():

        logger.info(
            "Evaluating %s...",
            model_name
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = model_name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    columns = [

        "Model",

        "MAE",

        "MSE",

        "RMSE",

        "R2",

        "MAPE"

    ]

    results_df = results_df[columns]

    return results_df
def rank_models(
    results_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Rank models by RMSE.
    """

    ranked = results_df.sort_values(
        by="RMSE",
        ascending=True
    ).reset_index(drop=True)

    ranked.insert(
        0,
        "Rank",
        range(
            1,
            len(ranked)+1
        )
    )

    return ranked
def save_results(
    results_df: pd.DataFrame
):
    """
    Save comparison table.
    """

    output = Path(
        MODEL_RESULTS_FILE
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        output,
        index=False
    )

    logger.info(
        "Results saved to %s",
        output
    )
def run_evaluation_pipeline(
    trained_models,
    X_test,
    y_test
):
    """
    Complete evaluation pipeline.
    """

    logger.info(
        "Starting model evaluation..."
    )

    results = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    results = rank_models(
        results
    )

    save_results(
        results
    )

    logger.info(
        "Evaluation completed."
    )

    return results
if __name__ == "__main__":

    logger.info(
        "Evaluation module loaded."
    )