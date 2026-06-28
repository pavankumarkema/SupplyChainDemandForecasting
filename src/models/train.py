"""
=========================================================
Project : Supply Chain Demand Forecasting
Module  : Model Training
Python  : 3.11
=========================================================

Train all regression models.
"""

from __future__ import annotations

import logging

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from .config import (
    X_TRAIN_FILE,
    Y_TRAIN_FILE,
    X_VALID_FILE,
    Y_VALID_FILE,
    X_TEST_FILE,
    Y_TEST_FILE,
    RANDOM_STATE,
    RF_PARAMS,
    GB_PARAMS,
    XGB_PARAMS,
    LGBM_PARAMS,
    CATBOOST_PARAMS,
)

logger = logging.getLogger(__name__)
def load_datasets():
    """
    Load train, validation and test datasets.
    """

    X_train = pd.read_csv(X_TRAIN_FILE)

    y_train = pd.read_csv(Y_TRAIN_FILE).squeeze("columns")

    X_valid = pd.read_csv(X_VALID_FILE)

    y_valid = pd.read_csv(Y_VALID_FILE).squeeze("columns")

    X_test = pd.read_csv(X_TEST_FILE)

    y_test = pd.read_csv(Y_TEST_FILE).squeeze("columns")

    logger.info("Datasets loaded successfully.")

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )
def build_models():
    """
    Create all regression models.
    """

    models = {

        "Linear Regression":

            LinearRegression(),

        "Decision Tree":

            DecisionTreeRegressor(
                random_state=RANDOM_STATE
            ),

        "Random Forest":

            RandomForestRegressor(
                **RF_PARAMS
            ),

        "Gradient Boosting":

            GradientBoostingRegressor(
                **GB_PARAMS
            ),

        "XGBoost":

            XGBRegressor(
                **XGB_PARAMS
            ),

        "LightGBM":

            LGBMRegressor(
                **LGBM_PARAMS
            ),

        "CatBoost":

            CatBoostRegressor(
                **CATBOOST_PARAMS
            )

    }

    return models
def train_models(
    X_train,
    y_train,
):
    """
    Train all models.
    """

    trained_models = {}

    models = build_models()

    for name, model in models.items():

        logger.info(
            "Training %s...",
            name
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        logger.info(
            "%s training completed.",
            name
        )

    return trained_models
def run_training_pipeline():
    """
    Execute model training.
    """

    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    ) = load_datasets()

    models = train_models(
        X_train,
        y_train,
    )

    logger.info(
        "All models trained successfully."
    )

    return (
        models,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )
if __name__ == "__main__":

    run_training_pipeline()
