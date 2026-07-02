# Supply Chain Demand Forecasting

End-to-end machine learning project for predicting product demand from sales, inventory, pricing, and supply-chain signals. The repository includes the training pipeline, saved artifacts, reports, and a Streamlit dashboard.

Live demo: [Supply Chain Demand Forecasting Streamlit App](https://supply-chanin-demand-forecasting.streamlit.app/)

## Executive Summary

This project forecasts demand for supply chain planning and inventory optimization. It is designed to reduce stockouts, limit overstock, and help operational teams make faster decisions using historical and engineered signals.

## Key Capabilities

- Train and compare multiple regression models for demand forecasting.
- Engineer time-based, inventory, pricing, delivery, lag, rolling, and cyclic features.
- Save reusable inference artifacts such as the scaler, encoders, feature list, and demand profile.
- Run single predictions and batch predictions in Streamlit.
- Review analytics, reports, and model outputs from the dashboard.

## Project Structure

```text
SupplyChainDemandForecasting/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
└── reports/
```

<<<<<<< HEAD
## End-to-End Workflow

1. Load the raw dataset from data/raw.
2. Validate the dataset structure, null values, duplicates, and columns.
3. Clean the data and standardize column names.
4. Remove columns that are unusable or may leak target information.
5. Create engineered features from date, inventory, pricing, delivery, lag, rolling, and cyclic patterns.
6. Encode categorical features and scale the final feature matrix.
7. Split the data in a time-aware way for training and testing.
8. Train multiple candidate models and compare evaluation metrics.
9. Save the best model and supporting inference artifacts.
10. Use the Streamlit dashboard to forecast demand and inspect reports.

## Main Files

- [main.py](main.py): training pipeline, model comparison, artifact generation, and reports.
- [app.py](app.py): Streamlit dashboard for prediction, analytics, and downloads.
- [requirements.txt](requirements.txt): Python dependencies.
- [models/](models): serialized model files.
- [artifacts/](artifacts): scaler, encoders, feature list, and demand profile.
- [reports/](reports): metrics, comparisons, feature importance, predictions, and charts.

## How to Run Locally

1. Install dependencies with pip install -r requirements.txt.
2. Train or refresh artifacts with python main.py.
3. Launch the dashboard with streamlit run app.py.

## Streamlit Pages

- Dashboard: project summary, artifact status, and demand profile.
- Predict Demand: single-record demand forecasting.
- Batch Prediction: CSV upload and bulk forecasting.
- Analytics: metrics, model comparison, feature importance, and actual versus predicted analysis.
- Reports: downloadable CSV outputs.
- About: stack summary and ML pipeline overview.

## Saved Outputs

- models/best_model.pkl
- artifacts/scaler.pkl
- artifacts/label_encoders.pkl
- artifacts/feature_list.pkl
- artifacts/demand_profile.pkl
- data/processed/processed_dataset.csv
- reports/prediction_metrics.csv
- reports/model_comparison.csv
- reports/prediction_report.csv
- reports/feature_importance.csv
- reports/*.png visualizations

## Modeling Summary

The training pipeline compares models such as linear regression, decision tree, random forest, extra trees, gradient boosting, AdaBoost, KNN, and XGBoost. The best model in the current results is XGBoost, based on the repository metrics output.

## Documentation Source

This repository keeps documentation in this README only. Separate docs were removed so the project has one canonical source of truth.

## Deployment

The live Streamlit deployment is available here:

- https://supply-chanin-demand-forecasting.streamlit.app/

## Troubleshooting

- If the dashboard cannot load predictions, rerun python main.py so the artifacts exist.
- If batch upload fails, confirm the CSV uses the same column names as the template.
- If metrics or reports are missing, verify the files still exist in reports.

## Maintenance Notes

- Keep the filenames in artifacts aligned with the loaders in app.py.
- Keep the raw dataset in data/raw using one of the loader-supported file formats.
- Review the reports after each retraining cycle.
- Update this README when the feature set, pipeline, or dashboard changes.
=======
## Pipeline

1. Load the raw dataset from `data/raw/`.
2. Validate the schema, missing values, duplicates, and column names.
3. Clean the dataset and handle missing values.
4. Remove columns that are not useful for modeling or may leak target information.
5. Engineer features from date, inventory, pricing, delivery, lag, rolling, and cyclic patterns.
6. Encode categorical variables and scale numeric features.
7. Split the data using a time-aware train/test strategy.
8. Train several regression models and compare performance.
9. Save the best model and supporting inference artifacts.
10. Use the Streamlit dashboard for prediction and analysis.

## Main Files

- [main.py](main.py): training, evaluation, artifact generation, and report creation.
- [app.py](app.py): Streamlit dashboard for prediction and analytics.
- [requirements.txt](requirements.txt): project dependencies.
- [reports/](reports): model comparison, prediction metrics, feature importance, and prediction outputs.
- [artifacts/](artifacts): scaler, label encoders, feature list, and demand profile used by the app.

## Run Locally

1. Install dependencies with `pip install -r requirements.txt`.
2. Train the pipeline with `python main.py`.
3. Launch the app with `streamlit run app.py`.

## Dashboard Pages

| Page | Purpose |
| --- | --- |
| Dashboard | Summary metrics and project overview |
| Predict Demand | Single-record demand prediction |
| Batch Prediction | CSV upload and bulk inference |
| Analytics | Model metrics, feature importance, and prediction charts |
| Reports | Downloadable CSV reports |
| About | Model, pipeline, and stack summary |

## Generated Outputs

- `models/best_model.pkl`
- `artifacts/scaler.pkl`
- `artifacts/label_encoders.pkl`
- `artifacts/feature_list.pkl`
- `artifacts/demand_profile.pkl`
- `data/processed/processed_dataset.csv`
- CSV and PNG outputs in `reports/`

## Deployment Notes

- The Streamlit app expects trained artifacts to exist before prediction pages are used.
- The pipeline looks for the raw dataset in `data/raw/` and will use either the Excel or CSV source if present.
- The hosted demo is available at [supply-chanin-demand-forecasting.streamlit.app](https://supply-chanin-demand-forecasting.streamlit.app/).
