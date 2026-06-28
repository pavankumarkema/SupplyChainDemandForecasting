"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Model Selection
Python  : 3.11
=========================================================

Select the best-performing regression model.
"""

from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)
def select_best_model(
    trained_models: dict,
    results_df: pd.DataFrame
):
    """
    Select the best model based on RMSE.

    Parameters
    ----------
    trained_models : dict
        Dictionary of trained models

    results_df : pd.DataFrame
        Model evaluation results

    Returns
    -------
    best_model_name
    best_model
    """

    logger.info("Selecting best model...")

    best_row = results_df.sort_values(
        by="RMSE",
        ascending=True
    ).iloc[0]

    best_model_name = best_row["Model"]

    best_model = trained_models[best_model_name]

    logger.info(
        "Best Model : %s",
        best_model_name
    )

    logger.info(
        "RMSE : %.4f",
        best_row["RMSE"]
    )

    logger.info(
        "R2 : %.4f",
        best_row["R2"]
    )

    return best_model_name, best_model
def print_leaderboard(
    results_df: pd.DataFrame
):
    """
    Display model ranking.
    """

    logger.info("=" * 70)

    logger.info("MODEL LEADERBOARD")

    logger.info("=" * 70)

    print(results_df)

    logger.info("=" * 70)
def model_summary(
    model_name: str,
    results_df: pd.DataFrame
):
    """
    Print summary of selected model.
    """

    row = results_df[
        results_df["Model"] == model_name
    ].iloc[0]

    logger.info("Selected Model Summary")

    logger.info("----------------------")

    logger.info("Model : %s", model_name)

    logger.info("MAE : %.4f", row["MAE"])

    logger.info("RMSE : %.4f", row["RMSE"])

    logger.info("R2 : %.4f", row["R2"])

    logger.info("MAPE : %.4f", row["MAPE"])
def run_model_selection_pipeline(
    trained_models: dict,
    results_df: pd.DataFrame
):
    """
    Execute model selection.
    """

    print_leaderboard(results_df)

    model_name, best_model = select_best_model(
        trained_models,
        results_df
    )

    model_summary(
        model_name,
        results_df
    )

    return model_name, best_model
if __name__ == "__main__":

    logger.info(
        "Model Selection Module Loaded."
    )
