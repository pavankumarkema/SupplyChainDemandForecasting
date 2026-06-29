"""
==============================================================================
Supply Chain Demand Forecasting Dashboard
==============================================================================

Framework : Streamlit

==============================================================================

"""

# =============================================================================
# IMPORTS
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Supply Chain Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_DIR = PROJECT_ROOT / "models"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

REPORT_DIR = PROJECT_ROOT / "reports"

DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# MODEL PATHS
# =============================================================================

MODEL_PATH = MODELS_DIR / "best_model.pkl"

SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

ENCODER_PATH = ARTIFACTS_DIR / "label_encoders.pkl"

FEATURE_PATH = ARTIFACTS_DIR / "feature_list.pkl"

# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource
def load_model():

    if MODEL_PATH.exists():

        return joblib.load(MODEL_PATH)

    return None


# =============================================================================
# LOAD SCALER
# =============================================================================

@st.cache_resource
def load_scaler():

    if SCALER_PATH.exists():

        return joblib.load(SCALER_PATH)

    return None


# =============================================================================
# LOAD LABEL ENCODERS
# =============================================================================

@st.cache_resource
def load_encoders():

    if ENCODER_PATH.exists():

        return joblib.load(ENCODER_PATH)

    return {}


# =============================================================================
# LOAD FEATURE LIST
# =============================================================================

@st.cache_resource
def load_feature_list():

    if FEATURE_PATH.exists():

        return joblib.load(FEATURE_PATH)

    return []


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

model = load_model()

scaler = load_scaler()

label_encoders = load_encoders()

feature_list = load_feature_list()

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("📦 Supply Chain")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Dashboard",

        "📈 Predict Demand",

        "📂 Batch Prediction",

        "📊 Analytics",

        "📑 Reports",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

# =============================================================================
# ARTIFACT STATUS
# =============================================================================

status = []

status.append(

    "✅ Model"

    if model is not None

    else "❌ Model"

)

status.append(

    "✅ Scaler"

    if scaler is not None

    else "❌ Scaler"

)

status.append(

    "✅ Encoders"

    if len(label_encoders) > 0

    else "❌ Encoders"

)

status.append(

    "✅ Features"

    if len(feature_list) > 0

    else "❌ Features"

)

for item in status:

    st.sidebar.write(item)

st.sidebar.markdown("---")

st.sidebar.info(

    "Supply Chain Demand Forecasting\n\n"
    "Machine Learning Dashboard"

)

# =============================================================================
# DASHBOARD
# =============================================================================

if page == "🏠 Dashboard":

    st.title("📦 Supply Chain Demand Forecasting")

    st.markdown(
        """
Welcome to the Supply Chain Demand Forecasting Dashboard.

This application allows you to:

- Predict product demand
- Batch predict using CSV
- Compare ML models
- Download reports
- Visualize model performance
"""
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(

        "Model",

        type(model).__name__

        if model

        else "Not Loaded"

    )

    c2.metric(

        "Features",

        len(feature_list)

    )

    c3.metric(

        "Reports",

        len(list(REPORT_DIR.glob("*")))

    )

    c4.metric(

        "Artifacts",

        len(list(ARTIFACTS_DIR.glob("*")))

    )

    st.divider()

    st.subheader("Project Structure")

    st.code(
"""
SupplyChainDemandForecasting/

├── app.py
├── main.py
├── data/
├── models/
├── artifacts/
├── reports/
"""
    )

    st.success("Dashboard Loaded Successfully.")
# =============================================================================
# PREDICT DEMAND
# =============================================================================

if page == "📈 Predict Demand":

    st.title("📈 Demand Prediction")

    st.write(
        "Fill in the product and supply chain details to predict demand."
    )

    st.divider()

    left, right = st.columns(2)

    # ==========================================================
    # PRODUCT DETAILS
    # ==========================================================

    with left:

        st.subheader("📦 Product")

        product_category = st.selectbox(
            "Product Category",
            [
                "Electronics",
                "Fashion",
                "Grocery",
                "Home",
                "Beauty",
                "Sports"
            ]
        )

        selling_price = st.number_input(
            "Selling Price",
            min_value=0.0,
            value=500.0,
            step=10.0
        )

        procurement_cost = st.number_input(
            "Procurement Cost",
            min_value=0.0,
            value=350.0,
            step=10.0
        )

        discount_percent = st.slider(
            "Discount (%)",
            0,
            80,
            10
        )

        inventory_level = st.number_input(
            "Inventory Level",
            min_value=0,
            value=250
        )

        safety_stock = st.number_input(
            "Safety Stock",
            min_value=0,
            value=80
        )

        reorder_point = st.number_input(
            "Reorder Point",
            min_value=0,
            value=120
        )

    # ==========================================================
    # SUPPLY CHAIN DETAILS
    # ==========================================================

    with right:

        st.subheader("🚚 Supply Chain")

        supplier_lead_time_days = st.slider(
            "Supplier Lead Time",
            1,
            30,
            7
        )

        delivery_time_days = st.slider(
            "Delivery Time",
            1,
            15,
            4
        )

        weather = st.selectbox(
            "Weather",
            [
                "Sunny",
                "Rainy",
                "Cloudy",
                "Cold",
                "Hot"
            ]
        )

        promotion_type = st.selectbox(
            "Promotion",
            [
                "No Promotion",
                "Flat Discount",
                "Cashback",
                "BOGO",
                "Festival Offer"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "UPI",
                "Credit Card",
                "Debit Card",
                "Wallet",
                "Net Banking"
            ]
        )

        region = st.selectbox(
            "Region",
            [
                "North",
                "South",
                "East",
                "West"
            ]
        )

        holiday = st.selectbox(
            "Holiday",
            [
                "No",
                "Yes"
            ]
        )

        festival = st.selectbox(
            "Festival",
            [
                "No",
                "Yes"
            ]
        )

        weekend = st.selectbox(
            "Weekend",
            [
                "No",
                "Yes"
            ]
        )

        month = st.slider(
            "Month",
            1,
            12,
            6
        )

        day_of_week = st.slider(
            "Day of Week",
            0,
            6,
            2
        )

    st.divider()

    predict_button = st.button(
        "🔮 Predict Demand",
        use_container_width=True
    )
    # ==========================================================
    # PREDICTION
    # ==========================================================

    if predict_button:

        if model is None:
            st.error("❌ Model not found. Please run main.py first.")
            st.stop()

        if scaler is None:
            st.error("❌ Scaler not found. Please run main.py first.")
            st.stop()

        # ------------------------------------------------------
        # Create Input DataFrame
        # ------------------------------------------------------

        input_df = pd.DataFrame({

            "product_category": [product_category],
            "selling_price": [selling_price],
            "procurement_cost": [procurement_cost],
            "discount_percent": [discount_percent],
            "inventory_level": [inventory_level],
            "safety_stock": [safety_stock],
            "reorder_point": [reorder_point],
            "supplier_lead_time_days": [supplier_lead_time_days],
            "delivery_time_days": [delivery_time_days],
            "weather": [weather],
            "promotion_type": [promotion_type],
            "payment_method": [payment_method],
            "region": [region],
            "holiday": [holiday],
            "festival": [festival],
            "weekend": [weekend],
            "month": [month],
            "day_of_week": [day_of_week]

        })

        # ------------------------------------------------------
        # Encode categorical columns
        # ------------------------------------------------------

        for column, encoder in label_encoders.items():

            if column in input_df.columns:

                value = str(input_df.loc[0, column])

                if value in encoder.classes_:

                    input_df[column] = encoder.transform(
                        input_df[column]
                    )

                else:

                    input_df[column] = 0

        # ------------------------------------------------------
        # Feature Engineering
        # ------------------------------------------------------

        input_df["profit_per_unit"] = (
            input_df["selling_price"]
            -
            input_df["procurement_cost"]
        )

        input_df["discount_amount"] = (
            input_df["selling_price"]
            *
            input_df["discount_percent"]
            / 100
        )

        input_df["discount_price"] = (
            input_df["selling_price"]
            -
            input_df["discount_amount"]
        )

        input_df["inventory_gap"] = (
            input_df["inventory_level"]
            -
            input_df["reorder_point"]
        )

        input_df["inventory_ratio"] = (
            input_df["inventory_level"]
            /
            (input_df["safety_stock"] + 1)
        )

        input_df["total_delivery_days"] = (
            input_df["supplier_lead_time_days"]
            +
            input_df["delivery_time_days"]
        )

        input_df["month_sin"] = np.sin(
            2 * np.pi * input_df["month"] / 12
        )

        input_df["month_cos"] = np.cos(
            2 * np.pi * input_df["month"] / 12
        )

        input_df["day_sin"] = np.sin(
            2 * np.pi * input_df["day_of_week"] / 7
        )

        input_df["day_cos"] = np.cos(
            2 * np.pi * input_df["day_of_week"] / 7
        )

        # Features unavailable during live prediction

        input_df["lag_1"] = 0
        input_df["lag_7"] = 0
        input_df["lag_30"] = 0
        input_df["rolling_mean_7"] = 0
        input_df["rolling_std_7"] = 0
        input_df["rolling_mean_30"] = 0

        # ------------------------------------------------------
        # Match Training Feature Order
        # ------------------------------------------------------

        for feature in feature_list:

            if feature not in input_df.columns:

                input_df[feature] = 0

        input_df = input_df.reindex(
            columns=feature_list,
            fill_value=0
        )

        # ------------------------------------------------------
        # Scale
        # ------------------------------------------------------

        input_scaled = scaler.transform(input_df)

        # ------------------------------------------------------
        # Predict
        # ------------------------------------------------------

        prediction = model.predict(input_scaled)[0]

        st.divider()

        st.success("✅ Prediction Completed")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Demand",
                f"{prediction:.2f} Units"
            )

        with col2:

            if prediction > 500:

                st.success("High Demand")

            elif prediction > 200:

                st.warning("Medium Demand")

            else:

                st.info("Low Demand")

        st.divider()

        st.subheader("Processed Input")

        st.dataframe(
            input_df,
            use_container_width=True
        )
# =============================================================================
# BATCH PREDICTION
# =============================================================================

if page == "📂 Batch Prediction":

    st.title("📂 Batch Demand Prediction")

    st.write(
        "Upload a CSV file containing product information."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            # =====================================================
            # Encode Categorical Columns
            # =====================================================

            for column, encoder in label_encoders.items():

                if column in batch_df.columns:

                    batch_df[column] = batch_df[column].astype(str)

                    batch_df[column] = batch_df[column].apply(

                        lambda value:

                        encoder.transform([value])[0]

                        if value in encoder.classes_

                        else 0

                    )

            # =====================================================
            # Feature Engineering
            # =====================================================

            if {
                "selling_price",
                "procurement_cost"
            }.issubset(batch_df.columns):

                batch_df["profit_per_unit"] = (

                    batch_df["selling_price"]

                    -

                    batch_df["procurement_cost"]

                )

            if {
                "selling_price",
                "discount_percent"
            }.issubset(batch_df.columns):

                batch_df["discount_amount"] = (

                    batch_df["selling_price"]

                    *

                    batch_df["discount_percent"]

                    / 100

                )

                batch_df["discount_price"] = (

                    batch_df["selling_price"]

                    -

                    batch_df["discount_amount"]

                )

            if {
                "inventory_level",
                "reorder_point"
            }.issubset(batch_df.columns):

                batch_df["inventory_gap"] = (

                    batch_df["inventory_level"]

                    -

                    batch_df["reorder_point"]

                )

            if {
                "inventory_level",
                "safety_stock"
            }.issubset(batch_df.columns):

                batch_df["inventory_ratio"] = (

                    batch_df["inventory_level"]

                    /

                    (batch_df["safety_stock"] + 1)

                )

            if {
                "supplier_lead_time_days",
                "delivery_time_days"
            }.issubset(batch_df.columns):

                batch_df["total_delivery_days"] = (

                    batch_df["supplier_lead_time_days"]

                    +

                    batch_df["delivery_time_days"]

                )

            if "month" in batch_df.columns:

                batch_df["month_sin"] = np.sin(
                    2*np.pi*batch_df["month"]/12
                )

                batch_df["month_cos"] = np.cos(
                    2*np.pi*batch_df["month"]/12
                )

            if "day_of_week" in batch_df.columns:

                batch_df["day_sin"] = np.sin(
                    2*np.pi*batch_df["day_of_week"]/7
                )

                batch_df["day_cos"] = np.cos(
                    2*np.pi*batch_df["day_of_week"]/7
                )

            # =====================================================
            # Missing Features
            # =====================================================

            missing_features = [

                "lag_1",
                "lag_7",
                "lag_30",
                "rolling_mean_7",
                "rolling_std_7",
                "rolling_mean_30"

            ]

            for feature in missing_features:

                if feature not in batch_df.columns:

                    batch_df[feature] = 0

            # =====================================================
            # Match Feature Order
            # =====================================================

            for feature in feature_list:

                if feature not in batch_df.columns:

                    batch_df[feature] = 0

            batch_df = batch_df.reindex(
                columns=feature_list,
                fill_value=0
            )

            # =====================================================
            # Scale Features
            # =====================================================

            batch_scaled = scaler.transform(batch_df)

            # =====================================================
            # Predict
            # =====================================================

            predictions = model.predict(batch_scaled)

            result_df = batch_df.copy()

            result_df["Predicted_Demand"] = predictions

            st.success("Batch Prediction Completed")

            st.subheader("Prediction Results")

            st.dataframe(
                result_df.head(20),
                use_container_width=True
            )

            csv = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(

                "📥 Download Predictions",

                data=csv,

                file_name="batch_predictions.csv",

                mime="text/csv"

            )

        except Exception as error:

            st.error(error)
# =============================================================================
# ANALYTICS DASHBOARD
# =============================================================================

if page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    st.markdown("## Model Performance")

    # ==========================================================
    # LOAD REPORTS
    # ==========================================================

    metrics_path = REPORT_DIR / "prediction_metrics.csv"

    comparison_path = REPORT_DIR / "model_comparison.csv"

    prediction_path = REPORT_DIR / "prediction_report.csv"

    feature_path = REPORT_DIR / "feature_importance.csv"

    # ==========================================================
    # KPI METRICS
    # ==========================================================

    if metrics_path.exists():

        metrics = pd.read_csv(metrics_path)

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                "MAE",

                f"{metrics.iloc[0]['MAE']:.3f}"

            )

        with col2:

            st.metric(

                "RMSE",

                f"{metrics.iloc[0]['RMSE']:.3f}"

            )

        with col3:

            st.metric(

                "MAPE",

                f"{metrics.iloc[0]['MAPE']:.2f}%"

            )

    else:

        st.warning("Prediction metrics not found.")

    st.divider()

    # ==========================================================
    # MODEL COMPARISON
    # ==========================================================

    st.subheader("🏆 Model Comparison")

    if comparison_path.exists():

        comparison = pd.read_csv(comparison_path)

        st.dataframe(

            comparison,

            use_container_width=True

        )

        st.bar_chart(

            comparison.set_index("Model")["R2 Score"]

        )

    else:

        st.warning("Model comparison report not found.")

    st.divider()

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    st.subheader("⭐ Top Features")

    if feature_path.exists():

        feature_df = pd.read_csv(feature_path)

        st.dataframe(

            feature_df.head(20),

            use_container_width=True

        )

        st.bar_chart(

            feature_df.head(20).set_index("Feature")["Importance"]

        )

    else:

        st.info("Feature importance unavailable.")

    st.divider()

    # ==========================================================
    # ACTUAL VS PREDICTED
    # ==========================================================

    st.subheader("📈 Prediction Analysis")

    if prediction_path.exists():

        prediction_df = pd.read_csv(prediction_path)

        chart_df = prediction_df[

            [

                "Actual_Demand",

                "Predicted_Demand"

            ]

        ]

        st.line_chart(chart_df)

        st.dataframe(

            prediction_df.head(20),

            use_container_width=True

        )

    else:

        st.warning("Prediction report not found.")
# =============================================================================
# REPORTS
# =============================================================================

if page == "📑 Reports":

    st.title("📑 Reports")

    st.markdown(
        "Download generated reports from the Machine Learning pipeline."
    )

    st.divider()

    reports = {

        "Prediction Report":
            REPORT_DIR / "prediction_report.csv",

        "Prediction Metrics":
            REPORT_DIR / "prediction_metrics.csv",

        "Model Comparison":
            REPORT_DIR / "model_comparison.csv",

        "Feature Importance":
            REPORT_DIR / "feature_importance.csv"

    }

    for report_name, report_path in reports.items():

        st.subheader(report_name)

        if report_path.exists():

            report_df = pd.read_csv(report_path)

            st.dataframe(

                report_df.head(),

                use_container_width=True

            )

            with open(report_path, "rb") as file:

                st.download_button(

                    label=f"📥 Download {report_name}",

                    data=file,

                    file_name=report_path.name,

                    mime="text/csv",

                    use_container_width=True

                )

        else:

            st.warning(

                f"{report_name} not found."

            )

        st.divider()

    # =====================================================
    # ZIP DOWNLOAD INFORMATION
    # =====================================================

    st.subheader("📂 Generated Reports")

    files = list(REPORT_DIR.glob("*"))

    if len(files) > 0:

        file_info = []

        for file in files:

            file_info.append({

                "File Name": file.name,

                "Size (KB)": round(file.stat().st_size / 1024, 2)

            })

        st.dataframe(

            pd.DataFrame(file_info),

            use_container_width=True

        )

    else:

        st.info("No report files available.")
# =============================================================================
# ABOUT
# =============================================================================

if page == "ℹ About":

    st.title("ℹ Supply Chain Demand Forecasting")

    st.markdown("""
## 📦 Project Overview

Supply Chain Demand Forecasting is an AI-powered Machine Learning application that predicts future product demand using historical sales and supply chain data.

This dashboard helps businesses:

- 📈 Forecast future demand
- 📦 Improve inventory planning
- 🚚 Optimize supply chain operations
- 💰 Reduce stockouts and overstock
- 📊 Analyze model performance
- 📁 Generate prediction reports

---
""")

    st.subheader("🛠 Technologies Used")

    tech_df = pd.DataFrame({

        "Technology":[

            "Python",

            "Pandas",

            "NumPy",

            "Scikit-Learn",

            "XGBoost",

            "Streamlit",

            "Joblib",

            "Matplotlib"

        ],

        "Purpose":[

            "Programming Language",

            "Data Analysis",

            "Numerical Computing",

            "Machine Learning",

            "Boosting Algorithm",

            "Web Dashboard",

            "Model Serialization",

            "Visualization"

        ]

    })

    st.dataframe(

        tech_df,

        use_container_width=True

    )

    st.divider()

    st.subheader("🤖 Machine Learning Workflow")

    workflow = [

        "Load Dataset",

        "Data Cleaning",

        "Feature Engineering",

        "Label Encoding",

        "Feature Scaling",

        "Train-Test Split",

        "Model Training",

        "Model Evaluation",

        "Best Model Selection",

        "Save Model",

        "Deploy using Streamlit"

    ]

    for step in workflow:

        st.write("✅", step)

    st.divider()

    st.subheader("📊 Algorithms Used")

    algo_df = pd.DataFrame({

        "Algorithm":[

            "Linear Regression",

            "Decision Tree",

            "Random Forest",

            "Extra Trees",

            "Gradient Boosting",

            "AdaBoost",

            "KNN",

            "XGBoost"

        ],

        "Purpose":[

            "Baseline",

            "Regression",

            "Ensemble",

            "Ensemble",

            "Boosting",

            "Boosting",

            "Distance Based",

            "Advanced Boosting"

        ]

    })

    st.dataframe(

        algo_df,

        use_container_width=True

    )

    st.divider()

    st.subheader("📂 Project Structure")

    st.code("""

SupplyChainDemandForecasting/

│

├── app.py

├── main.py

├── requirements.txt

├── README.md

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   └── best_model.pkl

│

├── artifacts/

│   ├── scaler.pkl

│   ├── label_encoders.pkl

│   └── feature_list.pkl

│

├── reports/

│

└── assets/

""")

    st.divider()

    st.subheader("📌 Current Status")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Model",

            "Loaded" if model else "Missing"

        )

    with col2:

        st.metric(

            "Features",

            len(feature_list)

        )

    with col3:

        st.metric(

            "Artifacts",

            len(list(ARTIFACTS_DIR.glob("*")))

        )

    st.divider()

    st.info(
        """
This project demonstrates an end-to-end Machine Learning pipeline
for Supply Chain Demand Forecasting using Streamlit.
        """
    )

    st.success("Application Ready 🚀")                                            