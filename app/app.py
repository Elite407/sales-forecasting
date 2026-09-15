import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config FIRST, before any other st commands or heavy imports
st.set_page_config(page_title="Favorita Sales Forecasting", layout="wide", page_icon="📈")

import sys
import os

# Ensure the repository root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Remove the app directory from sys.path so 'import app' doesn't find app.py
app_dir = os.path.abspath(os.path.dirname(__file__))
while app_dir in sys.path:
    sys.path.remove(app_dir)
if "" in sys.path:
    sys.path.remove("")

# Local imports
from app.data_loader import initialize_data_and_model
from app.tabs import (
    forecast_explorer,
    category_trends,
    classical_models,
    shap_explainer,
    model_comparison
)
from app.genai import agent
from app.genai.agent import _validate_gemini_key, _get_gemini_api_key

def main():
    # ---------------------------------------------------------
    # Sidebar — Gemini API key input
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("🔑 Gemini API Key")
        _existing_key = _get_gemini_api_key()
        _key_ok, _key_msg = _validate_gemini_key(_existing_key) if _existing_key else (False, "")
        
        if _key_ok:
            st.success("API key loaded ✓", icon="✅")
        elif _key_msg:
            st.warning(_key_msg, icon="⚠️")
            
        st.text_input(
            "Paste your Gemini API key",
            type="password",
            key="sidebar_gemini_key",
            help=(
                "Get a free key at https://aistudio.google.com/apikey — "
                "it should start with `AIzaSy…`. This overrides any key in "
                "secrets.toml for this session."
            ),
        )
        st.caption("Or add `GEMINI_API_KEY` to `.streamlit/secrets.toml` for persistence.")

    # ---------------------------------------------------------
    # Load Data and Model
    # ---------------------------------------------------------
    model, df = initialize_data_and_model()

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------
    st.title("📈 Corporación Favorita — Sales Forecasting")
    st.caption(
        "Classical baselines (ARIMA, SARIMA, Prophet) and a LightGBM model, "
        "compared on a granularity-consistent basis, plus a SHAP explainability "
        "layer for the LightGBM model. Deep learning models (LSTM / TFT / "
        "N-BEATS) are still in progress."
    )

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    st.info(
        f"Data window shown: **{date_min} → {date_max}** ({len(df):,} rows, "
        f"{df['store_nbr'].nunique()} stores × {df['family'].nunique()} product families). "
        "This is a trimmed sample bundled with the repo for size reasons — the "
        "classical-model diagnostics below are exported from the notebook, which "
        "trains on the full multi-year national series."
    )

    # ---------------------------------------------------------
    # Tabs rendering
    # ---------------------------------------------------------
    tab_explore, tab_categories, tab_classical, tab_shap, tab_compare, tab_genai = st.tabs(
        ["🔎 Forecast Explorer", "🗂️ Category Trends", "📐 Classical Models", "🧠 SHAP & Anomalies",
         "📊 Model Comparison", "🤖 GenAI Analyst"]
    )

    with tab_explore:
        forecast_explorer.render(df, model)

    with tab_categories:
        category_trends.render(df, agent.generate_quick_insight_gemini)

    with tab_classical:
        classical_models.render()

    with tab_shap:
        shap_explainer.render(df, model)

    with tab_compare:
        model_comparison.render()

    with tab_genai:
        agent.render_genai_tab(df, model)

    st.divider()
    st.caption(
        "Corporación Favorita Kaggle dataset. Repo: "
        "[github.com/Elite407/sales-forecasting](https://github.com/Elite407/sales-forecasting)"
    )

if __name__ == "__main__":
    main()
