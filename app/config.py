import os

# Paths (relative to repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(REPO_ROOT, "models", "lgbm_model.pkl")
DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "feature_store.csv")
COMPARISON_PATH = os.path.join(REPO_ROOT, "reports", "model_comparison_table.csv")
FEATURE_IMPORTANCE_IMG = os.path.join(REPO_ROOT, "reports", "lgbm_feature_importance.png")
PROPHET_IMG = os.path.join(REPO_ROOT, "reports", "prophet_components_holidays.png")

# Precomputed classical-model artifacts
ARIMA_SUMMARY_PATH = os.path.join(REPO_ROOT, "reports", "arima_summary.txt")
SARIMA_SUMMARY_PATH = os.path.join(REPO_ROOT, "reports", "sarima_summary.txt")
PROPHET_SUMMARY_PATH = os.path.join(REPO_ROOT, "reports", "prophet_summary.json")
ARIMA_FORECAST_PATH = os.path.join(REPO_ROOT, "reports", "arima_forecast.csv")
SARIMA_FORECAST_PATH = os.path.join(REPO_ROOT, "reports", "sarima_forecast.csv")
PROPHET_FORECAST_PATH = os.path.join(REPO_ROOT, "reports", "prophet_forecast.csv")

# Exact feature order the trained LightGBM Booster expects.
MODEL_FEATURES = [
    "day_of_week", "month", "is_weekend", "is_month_start", "is_month_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    "expanding_mean_sales", "dcoilwtico",
    "promo_lag_1", "promo_lag_7", "promo_rolling_mean_7",
    "is_holiday",
]

# Plain-English labels used in the SHAP anomaly narrative
FEATURE_LABELS = {
    "day_of_week": "day of the week",
    "month": "month of year",
    "is_weekend": "weekend flag",
    "is_month_start": "start-of-month flag",
    "is_month_end": "end-of-month flag",
    "month_sin": "seasonal month position (sin)",
    "month_cos": "seasonal month position (cos)",
    "dow_sin": "seasonal weekday position (sin)",
    "dow_cos": "seasonal weekday position (cos)",
    "sales_lag_1": "sales 1 day earlier",
    "sales_lag_7": "sales 7 days earlier (same weekday)",
    "sales_lag_14": "sales 14 days earlier",
    "sales_lag_28": "sales 28 days earlier",
    "rolling_mean_7": "7-day rolling average sales",
    "rolling_mean_28": "28-day rolling average sales",
    "rolling_std_7": "7-day sales volatility",
    "rolling_std_28": "28-day sales volatility",
    "expanding_mean_sales": "all-time average sales to date",
    "dcoilwtico": "oil price (WTI)",
    "promo_lag_1": "on-promotion flag, 1 day earlier",
    "promo_lag_7": "on-promotion flag, 7 days earlier",
    "promo_rolling_mean_7": "7-day rolling promo intensity",
    "is_holiday": "local/regional holiday flag",
    "is_national_holiday": "national holiday flag",
    "days_to_next_holiday": "days until next holiday",
    "is_anomaly": "flagged-anomaly indicator",
}

FAMILY_NAMES = {
    0: "AUTOMOTIVE", 1: "BABY CARE", 2: "BEAUTY", 3: "BEVERAGES", 4: "BOOKS",
    5: "BREAD/BAKERY", 6: "CELEBRATION", 7: "CLEANING", 8: "DAIRY", 9: "DELI",
    10: "EGGS", 11: "FROZEN FOODS", 12: "GROCERY I", 13: "GROCERY II",
    14: "HARDWARE", 15: "HOME AND KITCHEN I", 16: "HOME AND KITCHEN II",
    17: "HOME APPLIANCES", 18: "HOME CARE", 19: "LADIESWEAR",
    20: "LAWN AND GARDEN", 21: "LINGERIE", 22: "LIQUOR,WINE,BEER",
    23: "MAGAZINES", 24: "MEATS", 25: "PERSONAL CARE",
    26: "PET SUPPLIES", 27: "PLAYERS AND ELECTRONICS", 28: "POULTRY",
    29: "PREPARED FOODS", 30: "PRODUCE", 31: "SCHOOL AND OFFICE SUPPLIES",
    32: "SEAFOOD",
}
