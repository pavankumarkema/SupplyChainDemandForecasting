"""
==============================================================================
Supply Chain Demand Forecasting
==============================================================================

Language    : Python 3.11
Project     : End-to-End Supply Chain Demand Forecasting

==============================================================================

Pipeline

1. Load Dataset
2. Validate Dataset
3. Clean Dataset
4. Feature Engineering
5. Encoding
6. Scaling
7. Train/Test Split
8. Train ML Models
9. Evaluate Models
10. Select Best Model
11. Save Artifacts
12. Predict
13. Reports
14. Visualizations

==============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)

from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REPORT_DIR = PROJECT_ROOT / "reports"

# =============================================================================
# CREATE REQUIRED FOLDERS
# =============================================================================

for folder in [
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    ARTIFACTS_DIR,
    REPORT_DIR
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

# =============================================================================
# DATASET PATHS
# =============================================================================

EXCEL_DATASET = (
    RAW_DATA_DIR /
    "enterprise_supply_chain_dataset_india_2023_2025.xlsx"
)

CSV_DATASET = (
    RAW_DATA_DIR /
    "enterprise_supply_chain_dataset_india_2023_2025.csv"
)

TARGET_COLUMN = "units_sold"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# =============================================================================
# LOAD DATASET
# =============================================================================

def load_dataset():

    logger.info("=" * 80)
    logger.info("Loading Dataset")
    logger.info("=" * 80)

    if EXCEL_DATASET.exists():

        logger.info("Excel Dataset Found")

        df = pd.read_excel(EXCEL_DATASET)

    elif CSV_DATASET.exists():

        logger.info("CSV Dataset Found")

        df = pd.read_csv(CSV_DATASET)

    else:

        raise FileNotFoundError(
            f"\nDataset not found.\n\n"
            f"Expected:\n"
            f"{EXCEL_DATASET}\n"
            f"or\n"
            f"{CSV_DATASET}"
        )

    logger.info("Dataset Loaded Successfully")

    logger.info("Rows : %d", df.shape[0])

    logger.info("Columns : %d", df.shape[1])

    return df

# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset(df):

    logger.info("=" * 80)
    logger.info("Dataset Validation")
    logger.info("=" * 80)

    print("\nDataset Shape")
    print(df.shape)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows :", df.duplicated().sum())

    print("\nColumn Names\n")

    for column in df.columns:
        print(column)

# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_dataset(df):

    logger.info("=" * 80)
    logger.info("Cleaning Dataset")
    logger.info("=" * 80)

    df.columns = df.columns.str.strip()

    df.drop_duplicates(
        inplace=True
    )

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"]
        )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    for column in numeric_columns:

        df[column].fillna(
            df[column].median(),
            inplace=True
        )

    for column in categorical_columns:

        df[column].fillna(
            "Unknown",
            inplace=True
        )

    logger.info("Cleaning Completed")

    return df
# =============================================================================
# DATE FEATURE ENGINEERING
# =============================================================================

def create_date_features(df):

    logger.info("=" * 80)
    logger.info("Creating Date Features")
    logger.info("=" * 80)

    if "date" not in df.columns:
        logger.warning("'date' column not found.")
        return df

    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["day"] = df["date"].dt.day

    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = (
        df["date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


# =============================================================================
# INVENTORY FEATURES
# =============================================================================

def create_inventory_features(df):

    logger.info("Creating Inventory Features...")

    if {
        "inventory_level",
        "reorder_point"
    }.issubset(df.columns):

        df["inventory_gap"] = (
            df["inventory_level"]
            -
            df["reorder_point"]
        )

    if {
        "inventory_level",
        "safety_stock"
    }.issubset(df.columns):

        df["inventory_ratio"] = (
            df["inventory_level"]
            /
            (df["safety_stock"] + 1)
        )

    return df


# =============================================================================
# PRICE FEATURES
# =============================================================================

def create_price_features(df):

    logger.info("Creating Pricing Features...")

    if {
        "selling_price",
        "procurement_cost"
    }.issubset(df.columns):

        df["profit_per_unit"] = (
            df["selling_price"]
            -
            df["procurement_cost"]
        )

    if {
        "selling_price",
        "discount_percent"
    }.issubset(df.columns):

        df["discount_amount"] = (
            df["selling_price"]
            *
            df["discount_percent"]
            /
            100
        )

        df["discount_price"] = (
            df["selling_price"]
            -
            df["discount_amount"]
        )

    return df


# =============================================================================
# DELIVERY FEATURES
# =============================================================================

def create_delivery_features(df):

    logger.info("Creating Delivery Features...")

    if {
        "supplier_lead_time_days",
        "delivery_time_days"
    }.issubset(df.columns):

        df["total_delivery_days"] = (

            df["supplier_lead_time_days"]

            +

            df["delivery_time_days"]

        )

    return df


# =============================================================================
# SALES FEATURES
# =============================================================================

def create_sales_features(df):

    logger.info("Creating Sales Features...")

    if {
        "units_sold",
        "selling_price"
    }.issubset(df.columns):

        df["sales_amount"] = (

            df["units_sold"]

            *

            df["selling_price"]

        )

    return df


# =============================================================================
# LAG FEATURES
# =============================================================================

def create_lag_features(df):

    logger.info("Creating Lag Features...")

    if TARGET_COLUMN not in df.columns:
        return df

    df["lag_1"] = df[TARGET_COLUMN].shift(1)

    df["lag_7"] = df[TARGET_COLUMN].shift(7)

    df["lag_30"] = df[TARGET_COLUMN].shift(30)

    return df


# =============================================================================
# ROLLING FEATURES
# =============================================================================

def create_rolling_features(df):

    logger.info("Creating Rolling Features...")

    if TARGET_COLUMN not in df.columns:
        return df

    df["rolling_mean_7"] = (

        df[TARGET_COLUMN]

        .rolling(
            window=7,
            min_periods=1
        )

        .mean()

    )

    df["rolling_std_7"] = (

        df[TARGET_COLUMN]

        .rolling(
            window=7,
            min_periods=1
        )

        .std()

    )

    df["rolling_mean_30"] = (

        df[TARGET_COLUMN]

        .rolling(
            window=30,
            min_periods=1
        )

        .mean()

    )

    return df


# =============================================================================
# CYCLICAL FEATURES
# =============================================================================

def create_cyclic_features(df):

    logger.info("Creating Cyclic Features...")

    if "month" in df.columns:

        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["month_cos"] = np.cos(
            2 * np.pi * df["month"] / 12
        )

    if "day_of_week" in df.columns:

        df["day_sin"] = np.sin(
            2 * np.pi * df["day_of_week"] / 7
        )

        df["day_cos"] = np.cos(
            2 * np.pi * df["day_of_week"] / 7
        )

    return df


# =============================================================================
# FEATURE ENGINEERING PIPELINE
# =============================================================================

def feature_engineering(df):

    logger.info("=" * 80)
    logger.info("Feature Engineering Started")
    logger.info("=" * 80)

    df = create_date_features(df)

    df = create_inventory_features(df)

    df = create_price_features(df)

    df = create_delivery_features(df)

    df = create_sales_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = create_cyclic_features(df)

    df.fillna(0, inplace=True)

    logger.info("Feature Engineering Completed")

    logger.info(
        "Dataset Shape : %s",
        df.shape
    )

    return df
# =============================================================================
# ENCODING
# =============================================================================

def encode_features(df):
    """
    Encode categorical columns.
    """

    logger.info("=" * 80)
    logger.info("Encoding Categorical Features")
    logger.info("=" * 80)

    label_encoders = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Date has already been converted to features
    if "date" in categorical_columns:
        categorical_columns.remove("date")

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

        label_encoders[column] = encoder

    logger.info(
        "%d categorical columns encoded.",
        len(categorical_columns)
    )

    return df, label_encoders


# =============================================================================
# PREPARE FEATURES
# =============================================================================

def prepare_features(df):
    """
    Separate X and y.
    """

    logger.info("=" * 80)
    logger.info("Preparing Features")
    logger.info("=" * 80)

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"{TARGET_COLUMN} not found."
        )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    logger.info(
        "Features : %d",
        X.shape[1]
    )

    logger.info(
        "Samples : %d",
        X.shape[0]
    )

    return X, y


# =============================================================================
# SCALE FEATURES
# =============================================================================

def scale_features(X):
    """
    Standard Scaling
    """

    logger.info("=" * 80)
    logger.info("Scaling Features")
    logger.info("=" * 80)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(

        X_scaled,

        columns=X.columns,

        index=X.index

    )

    return X_scaled, scaler


# =============================================================================
# TRAIN TEST SPLIT
# =============================================================================

def split_dataset(X, y):
    """
    Time-series aware split.
    """

    logger.info("=" * 80)
    logger.info("Train Test Split")
    logger.info("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        shuffle=False,

        random_state=RANDOM_STATE

    )

    logger.info(
        "Training Samples : %d",
        len(X_train)
    )

    logger.info(
        "Testing Samples : %d",
        len(X_test)
    )

    return (

        X_train,

        X_test,

        y_train,

        y_test

    )


# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_artifacts(
    scaler,
    label_encoders,
    feature_columns
):
    """
    Save preprocessing artifacts.
    """

    logger.info("=" * 80)
    logger.info("Saving Artifacts")
    logger.info("=" * 80)

    joblib.dump(
        scaler,
        ARTIFACTS_DIR / "scaler.pkl"
    )

    joblib.dump(
        label_encoders,
        ARTIFACTS_DIR / "label_encoders.pkl"
    )

    joblib.dump(
        feature_columns,
        ARTIFACTS_DIR / "feature_list.pkl"
    )

    logger.info("Artifacts Saved Successfully.")


# =============================================================================
# SAVE PROCESSED DATA
# =============================================================================

def save_processed_dataset(
    X,
    y
):
    """
    Save processed dataset.
    """

    processed = X.copy()

    processed[TARGET_COLUMN] = y.values

    output_path = (

        PROCESSED_DATA_DIR /

        "processed_dataset.csv"

    )

    processed.to_csv(

        output_path,

        index=False

    )

    logger.info(

        "Processed dataset saved."

    )
# =============================================================================
# CREATE MACHINE LEARNING MODELS
# =============================================================================

def get_models():
    """
    Return all regression models.
    """

    models = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "AdaBoost": AdaBoostRegressor(
            random_state=RANDOM_STATE
        ),

        "KNN": KNeighborsRegressor(
            n_neighbors=5
        ),

        "XGBoost": XGBRegressor(

            objective="reg:squarederror",

            n_estimators=300,

            learning_rate=0.05,

            max_depth=8,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

    }

    return models


# =============================================================================
# TRAIN MODELS
# =============================================================================

def train_models(
    X_train,
    y_train
):
    """
    Train all machine learning models.
    """

    logger.info("=" * 80)
    logger.info("Training Machine Learning Models")
    logger.info("=" * 80)

    models = get_models()

    trained_models = {}

    for model_name, model in models.items():

        logger.info(
            "Training %s...",
            model_name
        )

        model.fit(
            X_train,
            y_train
        )

        trained_models[model_name] = model

        logger.info(
            "%s Completed",
            model_name
        )

    logger.info("=" * 80)
    logger.info("All Models Trained Successfully")
    logger.info("=" * 80)

    return trained_models


# =============================================================================
# DISPLAY TRAINED MODELS
# =============================================================================

def display_trained_models(
    trained_models
):
    """
    Print trained models.
    """

    logger.info("=" * 80)
    logger.info("TRAINED MODELS")
    logger.info("=" * 80)

    print()

    for index, model_name in enumerate(

        trained_models.keys(),

        start=1

    ):

        print(f"{index}. {model_name}")

    print()

    logger.info("=" * 80)


# =============================================================================
# TRAINING SUMMARY
# =============================================================================

def training_summary(
    trained_models
):
    """
    Display training summary.
    """

    logger.info("=" * 80)

    logger.info(
        "Total Models Trained : %d",
        len(trained_models)
    )

    logger.info("=" * 80)    
# =============================================================================
# EVALUATE MACHINE LEARNING MODELS
# =============================================================================

def evaluate_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluate all trained models.
    """

    logger.info("=" * 80)
    logger.info("Evaluating Models")
    logger.info("=" * 80)

    results = []

    for model_name, model in trained_models.items():

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({

            "Model": model_name,

            "MAE": round(mae, 4),

            "RMSE": round(rmse, 4),

            "R2 Score": round(r2, 4)

        })

        logger.info(
            "%-20s | R2 = %.4f",
            model_name,
            r2
        )

    results_df = pd.DataFrame(results)

    results_df.sort_values(
        by="R2 Score",
        ascending=False,
        inplace=True
    )

    results_df.reset_index(
        drop=True,
        inplace=True
    )

    logger.info("=" * 80)
    logger.info("Evaluation Completed")
    logger.info("=" * 80)

    return results_df


# =============================================================================
# DISPLAY MODEL COMPARISON
# =============================================================================

def display_model_results(results_df):

    logger.info("=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)

    print()

    print(results_df)

    print()


# =============================================================================
# SELECT BEST MODEL
# =============================================================================

def select_best_model(
    trained_models,
    results_df
):
    """
    Select the model with highest R² score.
    """

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[
        best_model_name
    ]

    logger.info(
        "Best Model : %s",
        best_model_name
    )

    logger.info(
        "Best R2 Score : %.4f",
        results_df.iloc[0]["R2 Score"]
    )

    return (

        best_model_name,

        best_model

    )


# =============================================================================
# SAVE BEST MODEL
# =============================================================================

def save_best_model(
    best_model,
    model_name
):
    """
    Save best trained model.
    """

    model_path = (
        MODEL_DIR /
        "best_model.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    logger.info(
        "Best Model Saved : %s",
        model_name
    )


# =============================================================================
# SAVE MODEL COMPARISON REPORT
# =============================================================================

def save_model_results(
    results_df
):
    """
    Save evaluation report.
    """

    report_path = (
        REPORT_DIR /
        "model_comparison.csv"
    )

    results_df.to_csv(
        report_path,
        index=False
    )

    logger.info(
        "Model comparison report saved."
    )


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def save_feature_importance(
    best_model,
    feature_names
):
    """
    Save feature importance for tree models.
    """

    if not hasattr(
        best_model,
        "feature_importances_"
    ):

        logger.info(
            "Feature importance unavailable."
        )

        return

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
        best_model.feature_importances_

    })

    importance.sort_values(

        by="Importance",

        ascending=False,

        inplace=True

    )

    importance.to_csv(

        REPORT_DIR /
        "feature_importance.csv",

        index=False

    )

    logger.info(
        "Feature importance saved."
    )    
# =============================================================================
# PREDICTION
# =============================================================================

def predict_demand(
    model,
    X_test
):
    """
    Predict demand using trained model.
    """

    logger.info("=" * 80)
    logger.info("Predicting Demand")
    logger.info("=" * 80)

    predictions = model.predict(X_test)

    logger.info("Prediction completed.")

    return predictions


# =============================================================================
# CREATE PREDICTION REPORT
# =============================================================================

def create_prediction_report(
    y_test,
    predictions
):
    """
    Create prediction dataframe.
    """

    report = pd.DataFrame({

        "Actual_Demand": y_test.values,

        "Predicted_Demand": predictions

    })

    report["Error"] = (

        report["Actual_Demand"]

        -

        report["Predicted_Demand"]

    )

    report["Absolute_Error"] = (

        report["Error"]

        .abs()

    )

    report["Percentage_Error"] = (

        report["Absolute_Error"]

        /

        (report["Actual_Demand"] + 1e-8)

        *

        100

    )

    return report


# =============================================================================
# SAVE PREDICTION REPORT
# =============================================================================

def save_prediction_report(
    prediction_report
):

    path = REPORT_DIR / "prediction_report.csv"

    prediction_report.to_csv(

        path,

        index=False

    )

    logger.info(

        "Prediction report saved."

    )


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    predictions
):

    prediction_df = pd.DataFrame({

        "Prediction": predictions

    })

    path = REPORT_DIR / "predictions.csv"

    prediction_df.to_csv(

        path,

        index=False

    )

    logger.info(

        "Predictions saved."

    )


# =============================================================================
# DISPLAY SAMPLE PREDICTIONS
# =============================================================================

def display_predictions(
    prediction_report,
    n=10
):

    logger.info("=" * 80)

    logger.info("Sample Predictions")

    logger.info("=" * 80)

    print()

    print(

        prediction_report.head(n)

    )

    print()

    logger.info("=" * 80)


# =============================================================================
# CALCULATE PREDICTION METRICS
# =============================================================================

def prediction_metrics(
    prediction_report
):

    mae = prediction_report[
        "Absolute_Error"
    ].mean()

    rmse = np.sqrt(

        np.mean(

            prediction_report["Error"] ** 2

        )

    )

    mape = prediction_report[
        "Percentage_Error"
    ].mean()

    logger.info("=" * 80)

    logger.info("Prediction Metrics")

    logger.info("=" * 80)

    logger.info("MAE  : %.4f", mae)

    logger.info("RMSE : %.4f", rmse)

    logger.info("MAPE : %.2f %%", mape)

    logger.info("=" * 80)

    return {

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape

    }


# =============================================================================
# SAVE METRICS
# =============================================================================

def save_prediction_metrics(metrics):
    """
    Save prediction metrics.
    """

    metrics_df = pd.DataFrame([metrics])

    metrics_df.to_csv(
        REPORT_DIR / "prediction_metrics.csv",
        index=False
    )

    logger.info("Prediction metrics saved.")
# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

def plot_actual_vs_predicted(
    y_test,
    predictions
):
    """
    Plot actual vs predicted demand.
    """

    logger.info("Generating Actual vs Predicted Plot...")

    plt.figure(figsize=(12, 6))

    plt.plot(
        y_test.values,
        label="Actual Demand"
    )

    plt.plot(
        predictions,
        label="Predicted Demand"
    )

    plt.title("Actual vs Predicted Demand")

    plt.xlabel("Samples")

    plt.ylabel("Units Sold")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR /
        "actual_vs_predicted.png",
        dpi=300
    )

    plt.close()


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def plot_model_comparison(
    results_df
):
    """
    Plot R² comparison.
    """

    logger.info("Generating Model Comparison Plot...")

    plt.figure(figsize=(12, 6))

    plt.bar(
        results_df["Model"],
        results_df["R2 Score"]
    )

    plt.xticks(rotation=45)

    plt.ylabel("R² Score")

    plt.title("Machine Learning Model Comparison")

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR /
        "model_comparison.png",
        dpi=300
    )

    plt.close()


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    model,
    feature_names
):
    """
    Plot feature importance.
    """

    if not hasattr(
        model,
        "feature_importances_"
    ):
        logger.info(
            "Feature importance unavailable."
        )
        return

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
        model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    ).head(20)

    plt.figure(figsize=(12, 8))

    plt.barh(

        importance["Feature"],

        importance["Importance"]

    )

    plt.gca().invert_yaxis()

    plt.title("Top 20 Important Features")

    plt.tight_layout()

    plt.savefig(

        REPORT_DIR /
        "feature_importance.png",

        dpi=300

    )

    plt.close()


# =============================================================================
# RESIDUAL PLOT
# =============================================================================

def plot_residuals(
    y_test,
    predictions
):

    residuals = y_test.values - predictions

    plt.figure(figsize=(10, 6))

    plt.scatter(

        predictions,

        residuals,

        alpha=0.5

    )

    plt.axhline(

        y=0,

        linestyle="--"

    )

    plt.xlabel("Predicted")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(

        REPORT_DIR /
        "residual_plot.png",

        dpi=300

    )

    plt.close()


# =============================================================================
# GENERATE ALL PLOTS
# =============================================================================

def generate_visualizations(

    best_model,

    X_train,

    y_test,

    predictions,

    results_df

):

    logger.info("=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)

    plot_actual_vs_predicted(
        y_test,
        predictions
    )

    plot_model_comparison(
        results_df
    )

    plot_feature_importance(
        best_model,
        X_train.columns
    )

    plot_residuals(
        y_test,
        predictions
    )

    logger.info("Visualizations Generated.")


# =============================================================================
# PIPELINE SUMMARY
# =============================================================================

def pipeline_summary():

    logger.info("=" * 80)

    logger.info("PIPELINE COMPLETED SUCCESSFULLY")

    logger.info("=" * 80)

    logger.info("Generated Files")

    logger.info("------------------------------")

    logger.info("models/best_model.pkl")

    logger.info("artifacts/scaler.pkl")

    logger.info("artifacts/label_encoders.pkl")

    logger.info("artifacts/feature_list.pkl")

    logger.info("reports/model_comparison.csv")

    logger.info("reports/prediction_report.csv")

    logger.info("reports/prediction_metrics.csv")

    logger.info("reports/feature_importance.csv")

    logger.info("reports/actual_vs_predicted.png")

    logger.info("reports/model_comparison.png")

    logger.info("reports/feature_importance.png")

    logger.info("reports/residual_plot.png")

    logger.info("=" * 80)   
# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():

    logger.info("=" * 80)
    logger.info("SUPPLY CHAIN DEMAND FORECASTING")
    logger.info("END-TO-END MACHINE LEARNING PIPELINE")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1 : Load Dataset
    # -------------------------------------------------------------------------

    df = load_dataset()

    # -------------------------------------------------------------------------
    # STEP 2 : Validate Dataset
    # -------------------------------------------------------------------------

    validate_dataset(df)

    # -------------------------------------------------------------------------
    # STEP 3 : Clean Dataset
    # -------------------------------------------------------------------------

    df = clean_dataset(df)

    # -------------------------------------------------------------------------
    # STEP 4 : Feature Engineering
    # -------------------------------------------------------------------------

    df = feature_engineering(df)

    # -------------------------------------------------------------------------
    # STEP 5 : Encode Features
    # -------------------------------------------------------------------------

    df, label_encoders = encode_features(df)

    # -------------------------------------------------------------------------
    # STEP 6 : Prepare Features
    # -------------------------------------------------------------------------

    X, y = prepare_features(df)

    # -------------------------------------------------------------------------
    # STEP 7 : Scale Features
    # -------------------------------------------------------------------------

    X, scaler = scale_features(X)

    # -------------------------------------------------------------------------
    # STEP 8 : Save Processed Dataset
    # -------------------------------------------------------------------------

    save_processed_dataset(X, y)

    # -------------------------------------------------------------------------
    # STEP 9 : Train Test Split
    # -------------------------------------------------------------------------

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y
    )

    # -------------------------------------------------------------------------
    # STEP 10 : Save Artifacts
    # -------------------------------------------------------------------------

    save_artifacts(
        scaler=scaler,
        label_encoders=label_encoders,
        feature_columns=X_train.columns.tolist()
    )

    # -------------------------------------------------------------------------
    # STEP 11 : Train Models
    # -------------------------------------------------------------------------

    trained_models = train_models(
        X_train,
        y_train
    )

    display_trained_models(
        trained_models
    )

    training_summary(
        trained_models
    )

    # -------------------------------------------------------------------------
    # STEP 12 : Evaluate Models
    # -------------------------------------------------------------------------

    results_df = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    display_model_results(
        results_df
    )

    # -------------------------------------------------------------------------
    # STEP 13 : Select Best Model
    # -------------------------------------------------------------------------

    best_model_name, best_model = select_best_model(
        trained_models,
        results_df
    )

    # -------------------------------------------------------------------------
    # STEP 14 : Save Best Model
    # -------------------------------------------------------------------------

    save_best_model(
        best_model,
        best_model_name
    )

    # -------------------------------------------------------------------------
    # STEP 15 : Save Feature Importance
    # -------------------------------------------------------------------------

    save_feature_importance(
        best_model,
        X_train.columns.tolist()
    )

    # -------------------------------------------------------------------------
    # STEP 16 : Prediction
    # -------------------------------------------------------------------------
    

    predictions = predict_demand(
        best_model,
        X_test
    )

    prediction_report = create_prediction_report(
        y_test,
        predictions
    )

    save_predictions(
        predictions
    )

    save_prediction_report(
       prediction_report
    )

    metrics = prediction_metrics(
       prediction_report
    )

    save_prediction_metrics(
       metrics
    )

    display_predictions(
       prediction_report
    )
    # -------------------------------------------------------------------------
    # STEP 17 : Prediction Report
    # -------------------------------------------------------------------------

    prediction_report = create_prediction_report(
        y_test,
        predictions
    )

    save_predictions(
        predictions
    )

    save_prediction_report(
        prediction_report
    )

    metrics = prediction_metrics(
        prediction_report
    )

    save_prediction_metrics(
        metrics
    )

    display_predictions(
        prediction_report
    )
    # -------------------------------------------------------------------------
    # STEP 18 : Save Reports
    # -------------------------------------------------------------------------

    save_model_results(
        results_df
    )

    # -------------------------------------------------------------------------
    # STEP 19 : Visualizations
    # -------------------------------------------------------------------------

    generate_visualizations(
        best_model,
        X_train,
        y_test,
        predictions,
        results_df
    )

    # -------------------------------------------------------------------------
    # STEP 20 : Summary
    # -------------------------------------------------------------------------

    pipeline_summary()

    logger.info("=" * 80)
    logger.info("PIPELINE EXECUTED SUCCESSFULLY")
    logger.info("=" * 80)

    return {

        "best_model": best_model,

        "best_model_name": best_model_name,

        "results": results_df,

        "predictions": prediction_report

    }


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        output = main()

    except Exception as error:

        logger.exception("Pipeline Failed.")

        raise        