# Model Card: Corporación Favorita Sales Forecaster

This model card details the primary machine learning model (LightGBM) utilized in the Corporación Favorita Sales Forecasting dashboard. It is structured based on Google's Model Card framework.

## Model Details

*   **Developer:** Internal Machine Learning Team (Placeholder)
*   **Model Date:** August 2023 (Based on dataset timeline ending August 2017)
*   **Model Version:** 1.0.0
*   **Model Type:** LightGBM (Light Gradient Boosting Machine) Regressor
*   **Architecture Details:** Histogram-based decision tree ensemble leveraging leaf-wise growth.
*   **License:** MIT

## Intended Use

*   **Primary Intended Uses:** Forecasting daily retail sales volume at the store and product-family level to inform inventory management, supply chain logistics, and staffing for Corporación Favorita stores in Ecuador.
*   **Primary Intended Users:** Supply chain analysts, store managers, and regional directors at Corporación Favorita.
*   **Out-of-Scope Use Cases:** This model is strictly designed for the specific product categories and store footprints present in the dataset. It should not be used to forecast sales in other countries, for unrelated retail chains, or during periods of unprecedented macroeconomic collapse not represented in the training distribution.

## Factors

*   **Relevant Factors:**
    *   **Time:** Seasonality (day of week, month, year, holidays).
    *   **Geography:** Store location (city, state), store type, and cluster.
    *   **Externalities:** Oil price fluctuations (WTI), major natural disasters (2016 Ecuador earthquake), and national economic shifts (public sector paydays).
*   **Evaluation Factors:** The model's performance was evaluated across different product families (e.g., Grocery vs. Automotive) and store clusters to ensure robustness across diverse sales profiles.

## Metrics

*   **Primary Evaluation Metrics:**
    *   **MAE (Mean Absolute Error):** Measures the average magnitude of the errors in predictions, without considering their direction.
    *   **RMSE (Root Mean Squared Error):** Penalizes larger errors more severely than MAE, which is important for inventory management where massive stockouts or overstocks are disproportionately costly.
    *   **MAPE (Mean Absolute Percentage Error):** Provides a relative measure of error, making it easier to compare performance across product families with vastly different baseline sales volumes.
*   **Performance:**
    *   *Note: Detailed performance metrics are available in the `reports/model_comparison_table.csv` and the README.md.*
    *   LightGBM excels at capturing granular, non-linear interactions (e.g., specific promotions at specific stores on specific days) compared to classical top-down models (ARIMA/SARIMA).

## Evaluation Data

*   **Datasets:** The model was evaluated on the trailing 16 days of the dataset (August 1, 2017 – August 15, 2017), which mirrors the Kaggle competition's public test set duration.
*   **Motivation:** This chronological split ensures the model is tested purely on its ability to forecast future unseen data, preventing data leakage from future events.

## Training Data

*   **Datasets:** Corporación Favorita Store Sales dataset (Kaggle).
*   **Timeframe:** January 1, 2013 – July 31, 2017.
*   **Preprocessing:**
    *   Missing oil prices were linearly interpolated.
    *   Dates were decomposed into time-series features (day of week, month, year).
    *   Holidays were mapped and joined, prioritizing transferred dates over traditional dates.
    *   Rolling window statistics (e.g., 7-day rolling mean, lag features) were generated to capture recent momentum.

## Quantitative Analyses

*   **Feature Importance (SHAP):** SHAP (SHapley Additive exPlanations) analysis reveals that the most critical drivers of predictions are typically recent sales momentum (lag features, rolling means) and promotion intensity (`onpromotion`). Holiday flags and day-of-week indicators heavily influence specific spikes.

## Ethical Considerations

*   **Data Privacy:** The dataset contains aggregated store-level sales data. It does not contain personally identifiable information (PII) about individual consumers.
*   **Bias and Fairness:** The model's predictions dictate inventory allocation. If the model systematically under-predicts sales in certain regions (e.g., lower-income areas vs. affluent areas), it could lead to product shortages in those specific communities, degrading service quality.

## Caveats and Recommendations

*   **External Shocks:** The model learned the impact of the April 2016 earthquake because it was present in the training data. It cannot proactively forecast the impact of *future*, novel macroeconomic or natural shocks without manual intervention or scenario planning.
*   **Deep Learning Alternatives:** While LightGBM provides an excellent baseline, state-of-the-art architectures for multivariate time-series forecasting (e.g., Temporal Fusion Transformers or N-BEATS) have not yet been fully evaluated in this pipeline and represent a logical next step for improvement.
