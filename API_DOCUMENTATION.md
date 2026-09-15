# Internal API Documentation: GenAI Tools

This document outlines the internal tool functions provided to the Generative AI Analyst (powered by the Gemini API) within the Streamlit application. These functions serve as a de facto internal API, allowing the agent to dynamically query the dataset, run models, and retrieve contextual facts.

All tools are defined in `app/genai/tools.py`.

---

## `get_model_comparison`

Returns the model comparison table, providing head-to-head error metrics across different forecasting models (ARIMA, SARIMA, Prophet, LightGBM).

*   **Parameters:** None
*   **Returns:** A list of dictionaries representing the rows of the comparison table.

**Example Response:**
```json
[
  {
    "Model": "SARIMA (2,1,2)(1,1,1)[7] [National]",
    "MAE": 86617.78,
    "RMSE": 136302.64,
    "MAPE": 33.36,
    "Notes": "Strong baseline capturing weekly seasonality on national aggregate."
  }
]
```

---

## `get_shap_drivers`

Retrieves the actual sales value and the top 5 SHAP (SHapley Additive exPlanations) feature contributions driving the LightGBM prediction for a specific store, product family, and date.

*   **Parameters:**
    *   `store` (int): Store number (e.g., `3`)
    *   `family` (int): Product family encoded value (e.g., `12` for GROCERY I)
    *   `date` (str): Date in `YYYY-MM-DD` format (e.g., `"2017-08-01"`)
*   **Returns:** A dictionary containing actual sales and a list of top driving features with their impact.

**Example Response:**
```json
{
  "actual_sales": 3450.5,
  "top_drivers": [
    {"feature": "Promotion Intensity", "shap_contribution": 1250.2},
    {"feature": "Rolling 7-Day Mean", "shap_contribution": 800.5},
    {"feature": "Day of Week", "shap_contribution": 450.1}
  ]
}
```

---

## `get_anomalies`

Lists all dates flagged as statistical sales anomalies within the loaded evaluation window for a specific store and product family.

*   **Parameters:**
    *   `store` (int): Store number.
    *   `family` (int): Product family encoded value.
*   **Returns:** A list of dates and their corresponding sales volumes where `is_anomaly = 1`.

**Example Response:**
```json
{
  "anomaly_dates": [
    {"date": "2016-04-18", "sales": "15400.0"},
    {"date": "2016-04-19", "sales": "12800.5"}
  ]
}
```

---

## `get_forecast_metrics`

Computes the LightGBM prediction accuracy metrics over the currently loaded evaluation window for a specific store and product family slice.

*   **Parameters:**
    *   `store` (int): Store number.
    *   `family` (int): Product family encoded value.
*   **Returns:** A dictionary containing the computed MAE, RMSE, average daily sales, and the number of days evaluated.

**Example Response:**
```json
{
  "store": 3,
  "family": "GROCERY I",
  "mae": 105.4,
  "rmse": 150.2,
  "avg_daily_sales": 2500.5,
  "n_days": 16
}
```

---

## `search_context`

Queries a lightweight TF-IDF knowledge base to retrieve specific real-world domain context relevant to the dataset (e.g., earthquake impacts, holiday shifts, payday effects).

*   **Parameters:**
    *   `query` (str): The search term or question from the user (e.g., `"earthquake"`).
*   **Returns:** A list of relevant contextual text snippets.

**Example Response:**
```json
{
  "context_snippets": [
    "On April 16, 2016, a 7.8-magnitude earthquake struck Ecuador near Pedernales. Relief efforts in the following weeks led people to donate water and other essential goods, which significantly affected supermarket sales nationally for several weeks after the earthquake."
  ]
}
```
