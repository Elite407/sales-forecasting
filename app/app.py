import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Favorita Sales Forecasting", layout="wide", page_icon="📈")

# ---------------------------------------------------------
# Paths (relative to repo root — Streamlit Cloud runs from there)
# ---------------------------------------------------------
MODEL_PATH = "models/lgbm_model.pkl"
DATA_PATH = "data/processed/feature_store.csv"
COMPARISON_PATH = "reports/model_comparison_table.csv"
FEATURE_IMPORTANCE_IMG = "reports/lgbm_feature_importance.png"
PROPHET_IMG = "reports/prophet_components_holidays.png"

# Exact feature order the trained LightGBM Booster expects.
# (Verified directly against the model with model.feature_name() —
# do not reorder or rename these.)
MODEL_FEATURES = [
    "day_of_week", "month", "is_weekend", "is_month_start", "is_month_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7", "rolling_std_28",
    "expanding_mean_sales", "dcoilwtico",
    "promo_lag_1", "promo_lag_7", "promo_rolling_mean_7",
    "is_holiday", "is_national_holiday", "days_to_next_holiday", "is_anomaly",
]

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
# Note: exact label-to-code mapping should be double-checked against
# models/encoders/family_encoder.pkl if available — this list follows the
# Favorita dataset's standard alphabetical family order as a best estimate.

# ---------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)

# ---------------------------------------------------------
# Load with error handling
# ---------------------------------------------------------
try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model from `{MODEL_PATH}`: {e}")
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"Could not load data from `{DATA_PATH}`: {e}")
    st.stop()

missing_cols = [c for c in MODEL_FEATURES if c not in df.columns]
if missing_cols:
    st.error(f"feature_store.csv is missing columns the model needs: {missing_cols}")
    st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📈 Corporación Favorita — Sales Forecasting")
st.caption(
    "**Project status:** Days 1–12 of a 38-day roadmap complete — data pipeline, "
    "feature engineering, and classical baselines (ARIMA, SARIMA, Prophet, LightGBM). "
    "Deep learning models (LSTM / TFT / N-BEATS) are still in progress. "
    "This dashboard is an interim checkpoint built on the LightGBM baseline."
)

date_min = df["date"].min().date()
date_max = df["date"].max().date()
st.info(f"Data window shown: **{date_min} → {date_max}** ({len(df):,} rows, "
        f"{df['store_nbr'].nunique()} stores × {df['family'].nunique()} product families)")

# ---------------------------------------------------------
# Interactive forecast explorer
# ---------------------------------------------------------
st.header("🔎 Explore a Store / Product Family")

col1, col2 = st.columns(2)
with col1:
    store = st.selectbox("Store number", sorted(df["store_nbr"].unique()))
with col2:
    family_options = sorted(df["family"].unique())
    family = st.selectbox(
        "Product family",
        family_options,
        format_func=lambda code: FAMILY_NAMES.get(code, f"Family {code}"),
    )

subset = df[(df["store_nbr"] == store) & (df["family"] == family)].sort_values("date")

if subset.empty:
    st.warning("No data available for this store / family combination in the current window.")
else:
    # Force numeric dtypes regardless of how pandas inferred them on this
    # environment — protects against "bad pandas dtypes" errors if a column
    # got read as object/string here but not elsewhere.
    X = subset[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()
        st.error(f"Non-numeric values found in columns: {bad_cols}. Check feature_store.csv for stray text/blank cells in these columns.")
        st.stop()
    preds = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["sales"], name="Actual",
                              line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Scatter(x=subset["date"], y=preds, name="LightGBM Predicted",
                              line=dict(color="#ff7f0e", width=2, dash="dash")))
    fig.update_layout(
        title=f"Store {store} — {FAMILY_NAMES.get(family, f'Family {family}')}",
        xaxis_title="Date", yaxis_title="Sales", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    mae = float(np.mean(np.abs(subset["sales"] - preds)))
    rmse = float(np.sqrt(np.mean((subset["sales"] - preds) ** 2)))
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (this slice)", f"{mae:,.2f}")
    m2.metric("RMSE (this slice)", f"{rmse:,.2f}")
    m3.metric("Avg. daily sales", f"{subset['sales'].mean():,.2f}")

st.divider()

# ---------------------------------------------------------
# Model comparison summary
# ---------------------------------------------------------
st.header("📊 Model Comparison So Far")
try:
    comp = load_comparison()
    st.dataframe(comp, use_container_width=True)
except FileNotFoundError:
    st.info(f"`{COMPARISON_PATH}` not found in the repo.")
except Exception as e:
    st.warning(f"Could not load model comparison table: {e}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("LightGBM Feature Importance")
    try:
        st.image(FEATURE_IMPORTANCE_IMG, use_container_width=True)
    except Exception:
        st.info("Feature importance image not found.")
with c2:
    st.subheader("Prophet Components")
    try:
        st.image(PROPHET_IMG, use_container_width=True)
    except Exception:
        st.info("Prophet components image not found.")

st.divider()
st.caption(
    "Built as an interim checkpoint for a 38-day sales forecasting project on the "
    "Corporación Favorita Kaggle dataset. Repo: "
    "[github.com/Elite407/sales-forecasting](https://github.com/Elite407/sales-forecasting)"
)
