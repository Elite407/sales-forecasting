import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config FIRST, before any other st commands or heavy imports
st.set_page_config(
    page_title="Favorita AI — Enterprise Sales Forecasting",
    layout="wide",
    page_icon="🛒"
)

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
    overview,
    forecast_explorer,
    category_trends,
    classical_models,
    shap_explainer,
    model_comparison
)
from app.genai import agent
from app.genai.agent import _validate_gemini_key, _get_gemini_api_key

def inject_custom_styles():
    st.markdown(
        """
        <style>
        /* Scoped Metric Card Styling */
        div[data-testid="stMetric"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 10px 14px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            color: #475569 !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: #0F172A !important;
        }
        /* Tab bar crispness */
        button[data-baseweb="tab"] {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    inject_custom_styles()

    # ---------------------------------------------------------
    # Sidebar Modernization
    # ---------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🛒 Favorita AI")
        st.caption("Enterprise Demand Forecasting & Causal Intelligence")
        
        _existing_key = _get_gemini_api_key()
        _key_ok, _key_msg = _validate_gemini_key(_existing_key) if _existing_key else (False, "")

        # Dynamic Status Pill
        if _key_ok:
            st.markdown(
                '<div style="margin: 6px 0 12px 0;">'
                '<span style="background-color: #DCFCE7; color: #166534; padding: 4px 12px; '
                'border-radius: 9999px; font-weight: 600; font-size: 0.82rem; display: inline-block;">'
                '🟢 Gemini Active</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="margin: 6px 0 12px 0;">'
                '<span style="background-color: #FEF3C7; color: #92400E; padding: 4px 12px; '
                'border-radius: 9999px; font-weight: 600; font-size: 0.82rem; display: inline-block;">'
                '🟡 Demo Mode</span></div>',
                unsafe_allow_html=True
            )

        with st.expander("🔑 Configure Gemini API Key", expanded=not _key_ok):
            st.text_input(
                "Gemini API key",
                type="password",
                key="sidebar_gemini_key",
                help="Get a free key at https://aistudio.google.com/apikey"
            )
            st.caption("Or configure `GEMINI_API_KEY` in `.streamlit/secrets.toml`.")

        st.divider()

        # Dataset Metadata Card
        with st.container(border=True):
            st.markdown("##### 📊 Dataset Metadata")
            st.markdown("- **Scope**: Corporación Favorita (Kaggle)")
            st.markdown("- **Timeline**: Jan 1, 2013 – Aug 31, 2017")
            st.markdown("- **Granularity**: Daily Store × Family Sales")
            st.markdown("- **Exogenous Signals**: WTI Oil, Earthquakes, Quincena Paydays, National Holidays")

        st.divider()

        # Project Resources & Links
        st.markdown("##### 📚 Project Resources")
        st.markdown("- [GitHub Repository](https://github.com/Elite407/sales-forecasting)")
        st.markdown("- [Model Architecture Document](#)")
        st.markdown("- [Notebook Suite (01–08)](#)")

    # ---------------------------------------------------------
    # Load Data and Model
    # ---------------------------------------------------------
    model, df = initialize_data_and_model()

    # ---------------------------------------------------------
    # Header Title Banner
    # ---------------------------------------------------------
    st.title("🛒 Corporación Favorita — Enterprise Demand Forecasting")
    st.caption(
        "Dual-track demand intelligence combining classical macro baselines (ARIMA, SARIMA, Prophet) "
        "with granular LightGBM gradient boosted trees, Explainable AI (SHAP), and an autonomous GenAI conversational analyst."
    )

    # ---------------------------------------------------------
    # Top Responsive KPI Ribbon
    # ---------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        with st.container(border=True):
            st.metric(
                "Total Sales Volume",
                "1.07B+",
                help="All-time unit sales volume across 54 Ecuadorian stores (2013–2017)"
            )
    with kpi2:
        with st.container(border=True):
            st.metric(
                "Active Stores",
                "54",
                help="Supermarket locations in 17 Ecuadorian cities across 5 store types"
            )
    with kpi3:
        with st.container(border=True):
            st.metric(
                "Product Families",
                "33",
                help="Merchandise categories spanning staple groceries to automotive goods"
            )
    with kpi4:
        with st.container(border=True):
            st.metric(
                "Champion Model MAE",
                "43,947",
                delta="-49.3% vs SARIMA",
                delta_color="inverse",
                help="LightGBM national bottom-up daily aggregate MAE (Baseline SARIMA: 86,618)"
            )

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 7 Dashboard Tabs
    # ---------------------------------------------------------
    tab_overview, tab_explore, tab_categories, tab_classical, tab_shap, tab_compare, tab_genai = st.tabs(
        [
            "🏠 Overview",
            "🔎 Forecast Explorer",
            "🗂️ Category Trends",
            "📐 Classical Models",
            "🧠 SHAP Explainer",
            "📊 Model Comparison",
            "🤖 GenAI Analyst",
        ]
    )

    with tab_overview:
        overview.render(df, model)

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
        "Corporación Favorita Grocery Sales Forecasting Showcase | "
        "Repository: [github.com/Elite407/sales-forecasting](https://github.com/Elite407/sales-forecasting)"
    )

if __name__ == "__main__":
    main()
