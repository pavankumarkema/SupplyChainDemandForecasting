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

import streamlit as st
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoders.pkl"
FEATURE_PATH = ARTIFACTS_DIR / "feature_list.pkl"
DEMAND_PROFILE_PATH = ARTIFACTS_DIR / "demand_profile.pkl"

# =============================================================================
# PAGE CONSTANTS
# =============================================================================

PAGE_DASHBOARD = "Dashboard"
PAGE_PREDICT = "Predict Demand"
PAGE_BATCH = "Batch Prediction"
PAGE_ANALYTICS = "Analytics"
PAGE_REPORTS = "Reports"
PAGE_ABOUT = "About"

PAGE_OPTIONS = [
    PAGE_DASHBOARD,
    PAGE_PREDICT,
    PAGE_BATCH,
    PAGE_ANALYTICS,
    PAGE_REPORTS,
    PAGE_ABOUT,
]

PAGE_ICONS = {
    PAGE_DASHBOARD: "🏠",
    PAGE_PREDICT: "📈",
    PAGE_BATCH: "📂",
    PAGE_ANALYTICS: "📊",
    PAGE_REPORTS: "📑",
    PAGE_ABOUT: "ℹ️",
}

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #1E88E5;
    }
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

@st.cache_resource(show_spinner="📦 Loading model and artifacts...")
def load_artifacts():
    artifacts = {
        "model": None,
        "scaler": None,
        "encoders": {},
        "features": [],
        "demand_profile": None,
    }

    if MODEL_PATH.exists():
        artifacts["model"] = joblib.load(MODEL_PATH)

    if SCALER_PATH.exists():
        artifacts["scaler"] = joblib.load(SCALER_PATH)

    if ENCODER_PATH.exists():
        artifacts["encoders"] = joblib.load(ENCODER_PATH)

    if FEATURE_PATH.exists():
        artifacts["features"] = joblib.load(FEATURE_PATH)

    if DEMAND_PROFILE_PATH.exists():
        artifacts["demand_profile"] = joblib.load(DEMAND_PROFILE_PATH)

    return artifacts

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
label_encoders = artifacts["encoders"]
feature_list = artifacts["features"]
demand_profile = artifacts["demand_profile"]

# =============================================================================
# DEMAND PROFILE DEFAULTS
# =============================================================================

if demand_profile is None:
    demand_profile = {
        "min": 0, "max": 100, "mean": 50,
        "median": 50, "std": 15,
        "q25": 30, "q33": 35, "q50": 50,
        "q66": 65, "q75": 75,
    }

def classify_demand(prediction):
    low_cutoff = demand_profile["q33"]
    high_cutoff = demand_profile["q66"]

    if prediction <= low_cutoff:
        return "💧 Low", "Low"
    elif prediction <= high_cutoff:
        return "⚡ Medium", "Medium"
    else:
        return "🔥 High", "High"

# =============================================================================
# CATEGORICAL OPTIONS
# =============================================================================

CATEGORY_OPTIONS = ["Electronics", "Fashion", "Grocery", "Home", "Beauty", "Sports"]
WEATHER_OPTIONS = ["Sunny", "Rainy", "Cloudy", "Cold", "Hot"]
PROMOTION_OPTIONS = ["No Promotion", "Flat Discount", "Cashback", "BOGO", "Festival Offer"]
PAYMENT_OPTIONS = ["UPI", "Credit Card", "Debit Card", "Wallet", "Net Banking"]
REGION_OPTIONS = ["North", "South", "East", "West"]
YES_NO = ["No", "Yes"]

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📦 Navigation")
    st.markdown("---")

    selected_page = st.radio(
        "Go to",
        PAGE_OPTIONS,
        format_func=lambda p: f"{PAGE_ICONS[p]} {p}"
    )

    st.markdown("---")
    st.markdown("### 🛠 Artifact Status")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model", "✅" if model else "❌")
    with c2:
        st.metric("Scaler", "✅" if scaler else "❌")

    c3, c4 = st.columns(2)
    with c3:
        st.metric("Encoders", f"✅ {len(label_encoders)}" if label_encoders else "❌")
    with c4:
        st.metric("Features", f"✅ {len(feature_list)}" if feature_list else "❌")

    st.markdown("---")

    st.markdown("### 📊 Demand Profile")
    st.write(f"Min: {demand_profile['min']:.0f} | "
             f"Median: {demand_profile['median']:.0f} | "
             f"Max: {demand_profile['max']:.0f}")

    st.markdown("---")
    st.info("💡 **Tip:** Run `python main.py` first to train and save artifacts.")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_encode(value, encoder):
    value_str = str(value)
    classes = set(encoder.classes_.astype(str))
    if value_str in classes:
        return int(encoder.transform([value_str])[0])
    return 0


def build_features(raw_input):
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
    elif isinstance(raw_input, list):
        df = pd.DataFrame(raw_input)
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(raw_input)}")

    # Ensure required columns exist
    required_defaults = {
        "month": 6, "day_of_week": 2, "year": 2024,
        "day": 15, "week_of_year": 24,
        "selling_price": 0, "procurement_cost": 0,
        "discount_percent": 0, "inventory_level": 0,
        "reorder_point": 0, "safety_stock": 0,
        "supplier_lead_time_days": 0, "delivery_time_days": 0,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Date features
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Inventory features
    df["inventory_gap"] = df["inventory_level"] - df["reorder_point"]
    df["inventory_ratio"] = df["inventory_level"] / (df["safety_stock"] + 1)

    # Pricing features
    df["profit_per_unit"] = df["selling_price"] - df["procurement_cost"]
    df["discount_amount"] = df["selling_price"] * df["discount_percent"] / 100
    df["discount_price"] = df["selling_price"] - df["discount_amount"]

    # Delivery features
    df["total_delivery_days"] = df["supplier_lead_time_days"] + df["delivery_time_days"]

    # Cyclic features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Lag/Rolling: use demand profile median/std as default
    lag_defaults = {
        "lag_1": demand_profile["median"],
        "lag_7": demand_profile["median"],
        "lag_30": demand_profile["median"],
        "rolling_mean_7": demand_profile["median"],
        "rolling_mean_30": demand_profile["median"],
        "rolling_std_7": demand_profile["std"],
    }

    for col, default_value in lag_defaults.items():
        if col not in df.columns:
            df[col] = default_value
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default_value)

    return df


def predict_pipeline(input_df):
    # Encode categorical columns
    for col, enc in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda v: safe_encode(v, enc))

    # Convert Yes/No flags
    flag_cols = ["holiday", "festival", "weekend", "promotion"]
    for col in flag_cols:
        if col in input_df.columns:
            input_df[col] = (
                input_df[col].astype(str)
                .map({"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0})
                .fillna(0).astype(int)
            )

    # Align features
    for feat in feature_list:
        if feat not in input_df.columns:
            input_df[feat] = 0

    input_df = input_df.reindex(columns=feature_list, fill_value=0)
    input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    input_scaled = scaler.transform(input_df)
    preds = model.predict(input_scaled)

    return preds, input_df

# =============================================================================
# CHART HELPERS (matplotlib to avoid altair/rpds bugs)
# =============================================================================

def bar_chart_mpl(data_series, title="", xlabel="", ylabel="Value", color="#1E88E5"):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(data_series.index.astype(str), data_series.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

def line_chart_mpl(df, title=""):
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in df.columns:
        ax.plot(df.index, df[col], label=col)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

if selected_page == PAGE_DASHBOARD:

    st.markdown(
        '<div class="main-header">📦 Supply Chain Demand Forecasting</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### Welcome to the AI-Powered Forecasting Dashboard

    This application predicts future product demand using historical 
    sales and supply chain data — helping you reduce stockouts, 
    optimize inventory, and improve operational efficiency.
    """)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("🤖 Model", type(model).__name__ if model else "Not Loaded")
    with c2:
        st.metric("📊 Features", len(feature_list))
    with c3:
        st.metric("📑 Reports", len(list(REPORT_DIR.glob("*"))))
    with c4:
        st.metric("💾 Artifacts", len(list(ARTIFACTS_DIR.glob("*"))))

    st.markdown("---")

    st.subheader("📊 Historical Demand Profile")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Min Demand", f"{demand_profile['min']:.0f}")
    with d2:
        st.metric("Median Demand", f"{demand_profile['median']:.0f}")
    with d3:
        st.metric("Mean Demand", f"{demand_profile['mean']:.0f}")
    with d4:
        st.metric("Max Demand", f"{demand_profile['max']:.0f}")

    st.markdown("---")

    st.subheader("🚀 Capabilities")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📈 Single Prediction**")
        st.write("Predict demand for an individual product configuration.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📂 Batch Prediction**")
        st.write("Upload CSV and predict demand for thousands of records.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📊 Analytics**")
        st.write("Explore model performance and feature importance.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.success("✅ Dashboard loaded successfully.")

# =============================================================================
# PAGE: PREDICT DEMAND
# =============================================================================

elif selected_page == PAGE_PREDICT:

    st.markdown(
        '<div class="main-header">📈 Demand Prediction</div>',
        unsafe_allow_html=True
    )

    if model is None or scaler is None:
        st.error("❌ Model or scaler not found. Run `python main.py` first.")
        st.stop()

    st.markdown("Fill in the product and supply chain details below to predict demand.")
    st.markdown("---")

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            st.subheader("📦 Product Details")

            product_category = st.selectbox("Product Category", CATEGORY_OPTIONS)
            selling_price = st.number_input("Selling Price (₹)", min_value=0.0, value=500.0, step=10.0)
            procurement_cost = st.number_input("Procurement Cost (₹)", min_value=0.0, value=350.0, step=10.0)
            discount_percent = st.slider("Discount (%)", 0, 80, 10)
            inventory_level = st.number_input("Inventory Level (units)", min_value=0, value=250, step=10)
            safety_stock = st.number_input("Safety Stock (units)", min_value=0, value=80, step=5)
            reorder_point = st.number_input("Reorder Point (units)", min_value=0, value=120, step=5)

            st.markdown("---")
            st.subheader("📊 Recent Demand Context")

            recent_demand = st.number_input(
                "Recent / Previous Demand (units)",
                min_value=0,
                value=int(demand_profile["median"]),
                step=1,
                help="Approximate recent demand. If unsure, leave the default."
            )

            avg_7_day_demand = st.number_input(
                "Average Demand - Last 7 Days",
                min_value=0,
                value=int(demand_profile["median"]),
                step=1,
                help="Average demand over the last 7 days."
            )

            avg_30_day_demand = st.number_input(
                "Average Demand - Last 30 Days",
                min_value=0,
                value=int(demand_profile["median"]),
                step=1,
                help="Average demand over the last 30 days."
            )

        with right:
            st.subheader("🚚 Supply Chain & Market")

            supplier_lead_time_days = st.slider("Supplier Lead Time (days)", 1, 30, 7)
            delivery_time_days = st.slider("Delivery Time (days)", 1, 15, 4)
            weather = st.selectbox("Weather", WEATHER_OPTIONS)
            promotion_type = st.selectbox("Promotion", PROMOTION_OPTIONS)
            promotion = 1 if promotion_type != "No Promotion" else 0
            payment_method = st.selectbox("Payment Method", PAYMENT_OPTIONS)
            region = st.selectbox("Region", REGION_OPTIONS)
            holiday = st.selectbox("Is Holiday?", YES_NO)
            festival = st.selectbox("Is Festival?", YES_NO)
            weekend = st.selectbox("Is Weekend?", YES_NO)
            month = st.slider("Month", 1, 12, 6)
            day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("🔮 Predict Demand", use_container_width=True)

    if submitted:

        raw = {
            "category": product_category,
            "weather": weather,
            "promotion": promotion,
            "promotion_type": promotion_type,
            "payment_method": payment_method,
            "region": region,
            "holiday": 1 if holiday == "Yes" else 0,
            "festival": 1 if festival == "Yes" else 0,
            "weekend": 1 if weekend == "Yes" else 0,
            "selling_price": selling_price,
            "procurement_cost": procurement_cost,
            "discount_percent": discount_percent,
            "inventory_level": inventory_level,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "supplier_lead_time_days": supplier_lead_time_days,
            "delivery_time_days": delivery_time_days,
            "month": month,
            "day_of_week": day_of_week,
            "year": 2024,
            "day": 15,
            "week_of_year": 24,
            "lag_1": recent_demand,
            "lag_7": avg_7_day_demand,
            "lag_30": avg_30_day_demand,
            "rolling_mean_7": avg_7_day_demand,
            "rolling_mean_30": avg_30_day_demand,
            "rolling_std_7": demand_profile["std"],
        }

        try:
            input_df = build_features(raw)
            preds, processed_df = predict_pipeline(input_df)
            prediction = float(preds[0])

            st.markdown("---")
            st.markdown(
                '<div class="success-box">✅ Prediction completed successfully!</div>',
                unsafe_allow_html=True
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("📦 Predicted Demand", f"{prediction:,.0f} units")

            with m2:
                demand_label, demand_level = classify_demand(prediction)
                st.metric("📊 Demand Level", demand_label)
                st.caption(
                    f"Low ≤ {demand_profile['q33']:.0f} | "
                    f"Medium ≤ {demand_profile['q66']:.0f} | "
                    f"High > {demand_profile['q66']:.0f}"
                )

            with m3:
                revenue = prediction * selling_price
                st.metric("💰 Est. Revenue", f"₹{revenue:,.0f}")

            st.markdown("### 💡 Recommendation")

            if inventory_level < prediction:
                st.error(
                    f"⚠️ **Stockout Risk:** Predicted demand ({prediction:.0f}) "
                    f"exceeds inventory ({inventory_level}). "
                    f"Restock at least {int(prediction - inventory_level)} units."
                )
            elif prediction < 0.3 * inventory_level:
                st.warning("⚠️ **Overstock Risk:** Demand is below inventory. Consider promotions.")
            else:
                st.success("✅ Inventory is well-aligned with expected demand.")

            with st.expander("🔍 View Processed Input Features"):
                st.dataframe(processed_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.exception(e)

# =============================================================================
# PAGE: BATCH PREDICTION
# =============================================================================

elif selected_page == PAGE_BATCH:

    st.markdown(
        '<div class="main-header">📂 Batch Demand Prediction</div>',
        unsafe_allow_html=True
    )

    if model is None or scaler is None:
        st.error("❌ Model or scaler not found. Run `main.py` first.")
        st.stop()

    st.write("Upload a CSV file containing product and supply chain data.")

    template_data = pd.DataFrame([{
        "category": "Electronics",
        "selling_price": 500,
        "procurement_cost": 350,
        "discount_percent": 10,
        "inventory_level": 250,
        "safety_stock": 80,
        "reorder_point": 120,
        "supplier_lead_time_days": 7,
        "delivery_time_days": 4,
        "weather": "Sunny",
        "promotion": 0,
        "promotion_type": "No Promotion",
        "payment_method": "UPI",
        "region": "North",
        "holiday": 0,
        "festival": 0,
        "weekend": 0,
        "month": 6,
        "day_of_week": 2,
        "year": 2024,
        "day": 15,
        "week_of_year": 24,
    }])

    csv_template = template_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download CSV Template",
        data=csv_template,
        file_name="batch_template.csv",
        mime="text/csv"
    )

    st.markdown("---")

    uploaded = st.file_uploader("📤 Upload CSV File", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.subheader(f"📋 Uploaded Data ({len(batch_df)} rows)")

            with st.expander("Preview data"):
                st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", use_container_width=True):

                with st.spinner("⏳ Processing batch..."):

                    engineered = build_features(batch_df)
                    preds, _ = predict_pipeline(engineered)

                    batch_df["Predicted_Demand"] = np.round(preds, 2)
                    batch_df["Demand_Level"] = pd.cut(
                        batch_df["Predicted_Demand"],
                        bins=[
                            -np.inf,
                            demand_profile["q33"],
                            demand_profile["q66"],
                            np.inf
                        ],
                        labels=["Low", "Medium", "High"]
                    )

                st.success(f"✅ Predicted demand for {len(batch_df)} records!")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Demand", f"{batch_df['Predicted_Demand'].sum():,.0f}")
                with col2:
                    st.metric("Average Demand", f"{batch_df['Predicted_Demand'].mean():,.0f}")
                with col3:
                    st.metric("Max Demand", f"{batch_df['Predicted_Demand'].max():,.0f}")

                st.subheader("📊 Prediction Results (preview)")
                st.dataframe(batch_df.head(50), use_container_width=True)

                if "Demand_Level" in batch_df.columns:
                    st.subheader("📈 Demand Distribution")
                    dist = batch_df["Demand_Level"].value_counts()
                    bar_chart_mpl(dist, title="Demand Level Distribution",
                                  xlabel="Demand Level", ylabel="Count", color="#43A047")

                csv_out = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions CSV",
                    data=csv_out,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

# =============================================================================
# PAGE: ANALYTICS
# =============================================================================

elif selected_page == PAGE_ANALYTICS:

    st.markdown(
        '<div class="main-header">📊 Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    metrics_path = REPORT_DIR / "prediction_metrics.csv"
    comparison_path = REPORT_DIR / "model_comparison.csv"
    prediction_path = REPORT_DIR / "prediction_report.csv"
    feature_path = REPORT_DIR / "feature_importance.csv"

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 KPIs", "🏆 Models", "⭐ Features", "📉 Predictions"
    ])

    with tab1:
        st.subheader("🎯 Prediction Metrics")

        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{metrics.iloc[0]['MAE']:.3f}")
            with c2:
                st.metric("RMSE", f"{metrics.iloc[0]['RMSE']:.3f}")
            with c3:
                st.metric("MAPE", f"{metrics.iloc[0]['MAPE']:.2f}%")
            st.dataframe(metrics, use_container_width=True)
        else:
            st.warning("No metrics available. Run `main.py` first.")

    with tab2:
        st.subheader("🏆 Model Comparison")

        if comparison_path.exists():
            comp = pd.read_csv(comparison_path).sort_values("R2 Score", ascending=False)
            st.dataframe(comp, use_container_width=True)

            bar_chart_mpl(
                comp.set_index("Model")["R2 Score"],
                title="R² Score by Model",
                xlabel="Model",
                ylabel="R² Score",
                color="#1E88E5"
            )
        else:
            st.warning("No comparison report found.")

    with tab3:
        st.subheader("⭐ Feature Importance")

        if feature_path.exists():
            feat_df = pd.read_csv(feature_path).head(20)
            st.dataframe(feat_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            feat_sorted = feat_df.sort_values("Importance", ascending=True)
            ax.barh(feat_sorted["Feature"], feat_sorted["Importance"], color="#1E88E5")
            ax.set_xlabel("Importance")
            ax.set_title("Top 20 Features")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Feature importance unavailable.")

    with tab4:
        st.subheader("📉 Prediction Analysis")

        if prediction_path.exists():
            pred_df = pd.read_csv(prediction_path)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Samples", len(pred_df))
            with c2:
                st.metric("Avg Actual", f"{pred_df['Actual_Demand'].mean():.0f}")
            with c3:
                st.metric("Avg Predicted", f"{pred_df['Predicted_Demand'].mean():.0f}")

            line_chart_mpl(
                pred_df[["Actual_Demand", "Predicted_Demand"]].head(100),
                title="Actual vs Predicted Demand (first 100 samples)"
            )

            st.dataframe(pred_df.head(20), use_container_width=True)
        else:
            st.warning("Prediction report not found.")

# =============================================================================
# PAGE: REPORTS
# =============================================================================

elif selected_page == PAGE_REPORTS:

    st.markdown(
        '<div class="main-header">📑 Reports</div>',
        unsafe_allow_html=True
    )

    st.write("Download generated reports from the machine learning pipeline.")

    reports = {
        "Prediction Report": REPORT_DIR / "prediction_report.csv",
        "Prediction Metrics": REPORT_DIR / "prediction_metrics.csv",
        "Model Comparison": REPORT_DIR / "model_comparison.csv",
        "Feature Importance": REPORT_DIR / "feature_importance.csv",
    }

    cols = st.columns(2)

    for idx, (name, path) in enumerate(reports.items()):
        with cols[idx % 2]:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.subheader(f"📄 {name}")

            if path.exists():
                df = pd.read_csv(path)
                st.write(f"**Rows:** {len(df)} | **Size:** {path.stat().st_size / 1024:.1f} KB")
                with open(path, "rb") as f:
                    st.download_button(
                        f"📥 Download {name}",
                        data=f.read(),
                        file_name=path.name,
                        mime="text/csv",
                        key=f"dl_{name}",
                        use_container_width=True
                    )

                with st.expander(f"Preview {name}"):
                    st.dataframe(df.head(), use_container_width=True)
            else:
                st.warning(f"❌ File not found.")
            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE: ABOUT
# =============================================================================

elif selected_page == PAGE_ABOUT:

    st.markdown(
        '<div class="main-header">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    ### 📦 Supply Chain Demand Forecasting
    
    An end-to-end Machine Learning system that predicts future product 
    demand using historical sales, inventory, pricing, and supply chain data.
    """)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🛠 Tech Stack")
        st.markdown("""
        - 🐍 **Python 3.11**
        - 🐼 **Pandas & NumPy**
        - 🤖 **Scikit-Learn, XGBoost**
        - 🎨 **Streamlit** (this dashboard)
        - 💾 **Joblib** (model serialization)
        - 📊 **Matplotlib**
        """)

    with c2:
        st.subheader("🤖 ML Pipeline")
        st.markdown("""
        1. Load & Validate Dataset
        2. Clean (handle missing, dedupe)
        3. Drop unusable columns (IDs, leaked features)
        4. Feature Engineering (date, lag, rolling, cyclic)
        5. Encode categorical features
        6. Train/Test Split (time-series aware)
        7. Scale features
        8. Train 8 models
        9. Evaluate (MAE, RMSE, R²)
        10. Select best model
        11. Save artifacts & predict
        """)

    st.divider()

    st.subheader("📊 Models Used")

    algo_df = pd.DataFrame({
        "Algorithm": [
            "Linear Regression", "Decision Tree", "Random Forest",
            "Extra Trees", "Gradient Boosting", "AdaBoost",
            "KNN", "XGBoost"
        ],
        "Type": [
            "Linear", "Tree", "Ensemble (Bagging)",
            "Ensemble (Bagging)", "Boosting", "Boosting",
            "Instance-based", "Boosting"
        ]
    })

    st.dataframe(algo_df, use_container_width=True, hide_index=True)

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model", "✅ Loaded" if model else "❌ Missing")
    with c2:
        st.metric("Features", len(feature_list))
    with c3:
        st.metric("Artifacts", len(list(ARTIFACTS_DIR.glob("*"))))

    st.divider()
    st.success("🚀 Dashboard ready — start predicting from the sidebar!")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<center><sub>Built with ❤️ using Streamlit | "
    "Enterprise Supply Chain Demand Forecasting 2026</sub></center>",
    unsafe_allow_html=True
)