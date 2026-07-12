import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Favorita Sales Forecasting", layout="wide")

# ---------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/feature_store.csv", parse_dates=["date"])
    return df

model = load_model()
df = load_data()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("📈 Corporación Favorita — Sales Forecasting")
st.caption(
    "Interim checkpoint (Days 1–12): data pipeline, feature store, and classical "
    "models (ARIMA / SARIMA / Prophet / LightGBM) complete. Deep learning phase "
    "(LSTM / TFT / N-BEATS) is in progress — this dashboard reflects the LightGBM "
    "baseline as a temporary submission."
)

# ---------------------------------------------------------
# Interactive forecast explorer
# ---------------------------------------------------------
st.header("Explore Store / Product Family")

col1, col2 = st.columns(2)
with col1:
    store = st.selectbox("Store number", sorted(df["store_nbr"].unique()))
with col2:
    family = st.selectbox("Product family", sorted(df["family"].unique()))

subset = df[(df["store_nbr"] == store) & (df["family"] == family)].sort_values("date")

if subset.empty:
    st.warning("No data available for this store / family combination.")
else:
    # Use the model's own feature list if available, otherwise infer it
    if hasattr(model, "feature_name_"):
        feature_cols = list(model.feature_name_)
    else:
        feature_cols = [c for c in subset.columns if c not in ["date", "store_nbr", "family", "sales"]]

    X = subset[feature_cols]
    preds = model.predict(X)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["date"], y=subset["sales"], name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=subset["date"], y=preds, name="LightGBM Predicted", line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(title=f"Store {store} — {family}", xaxis_title="Date", yaxis_title="Sales", height=450)
    st.plotly_chart(fig, use_container_width=True)

    mae = float(np.mean(np.abs(subset["sales"] - preds)))
    st.metric("MAE (this slice)", f"{mae:,.2f}")

st.divider()

# ---------------------------------------------------------
# Model comparison summary (static, from reports/)
# ---------------------------------------------------------
st.header("Model Comparison So Far")
try:
    comp = pd.read_csv("reports/model_comparison.csv")
    st.dataframe(comp, use_container_width=True)
except FileNotFoundError:
    st.info("reports/model_comparison.csv not found.")

st.subheader("LightGBM Feature Importance")
try:
    st.image("reports/figures/lgbm_feature_importance.png", use_container_width=True)
except FileNotFoundError:
    st.info("Feature importance image not found.")

st.subheader("Prophet Components")
try:
    st.image("reports/figures/prophet_components.png", use_container_width=True)
except FileNotFoundError:
    st.info("Prophet components image not found.")
