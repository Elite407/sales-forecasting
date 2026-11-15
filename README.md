# 🛒 Corporación Favorita — Enterprise Sales Forecasting & GenAI Decision Intelligence

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-02569B?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-ARIMA%20%7C%20SARIMA-4A154B?style=for-the-badge)](https://www.statsmodels.org/)
[![Meta Prophet](https://img.shields.io/badge/Meta%20Prophet-v1.1+-0081FB?style=for-the-badge&logo=meta&logoColor=white)](https://facebook.github.io/prophet/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet?style=for-the-badge)](https://shap.readthedocs.io/)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-GenAI%20Copilot-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Corporaci%C3%B3n%20Favorita-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Enterprise-grade dual-track forecasting platform bridging classical statistical time-series baselines (ARIMA, SARIMA, Meta Prophet) with granular gradient-boosted decision trees (LightGBM), SHAP Explainable AI (XAI), and an autonomous Gemini GenAI decision copilot.**

---

## Executive Summary & Operational Footprint

Predicting retail grocery demand across developing economies requires navigating macroeconomic volatility, supply chain shocks, and complex seasonal dynamics. This platform provides an end-to-end forecasting and decision intelligence solution based on the **Corporación Favorita** dataset (Ecuador's premier supermarket chain).

```
┌─────────────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐
│   TOTAL SALES VOLUME    │      ACTIVE STORES      │    PRODUCT FAMILIES     │   CHAMPION MODEL MAE    │
│       1.07B+ Units      │        54 Stores        │       33 Families       │      43,947 Units       │
│  3.4M+ Transactions     │ 16 Provinces, 22 Cities │   Groceries to Staples  │   -49.3% vs SARIMA      │
└─────────────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘
```

The system addresses the fundamental **granularity mismatch fallacy** in retail forecasting: classical statistical methods excel at macro national aggregates, whereas business operations demand store-and-sku-level granularity. By deploying **bottom-up aggregation** ($\hat{Y}_{t}^{\text{nat}} = \sum \sum \hat{y}_{s,f,t}$), our champion LightGBM model achieves a **5.42% MAPE** on national totals—cutting the error of the best aggregate statistical baseline (SARIMA) in half.

---

## 🖥️ Interactive Showcase & Dashboard Tour

The platform features an enterprise Streamlit analytical suite configured in a clean, high-contrast light theme with an executive KPI ribbon, sidebar telemetry, and 7 specialized analytical modules.

### Tab 1: Executive Overview & Macro Shocks
Provides strategic orientation, operational metrics, the mathematical formulation of bottom-up aggregation, and a historical timeline of macroeconomic shocks (WTI crude oil collapse, April 2016 7.8M earthquake, and biweekly public sector paydays).

![Tab 0 — Executive Overview](assets/dashboard_overview.png)

---

### Tab 2: Interactive Forecast Explorer
Enables store managers and supply chain planners to drill into any of the **1,782 Store × Product Family** combinations. Features an interactive date window filter slider, dynamic recalculation of slice-level MAE/RMSE, actual vs. predicted Plotly curves, and one-click CSV export.

![Tab 1 — Forecast Explorer](assets/dashboard_forecast_explorer.png)

---

### Tab 3: Multi-Category Macro Dynamics
Aggregates and visualizes sales across multiple merchandise families simultaneously (e.g., Grocery I, Beverages, Cleaning, Produce), tracking macro demand shifts, category market basket share, and cross-category substitution trends over time.

![Tab 2 — Category Trends](assets/dashboard_category_trends.png)

---

### Tab 4: Responsive Classical Statistical Diagnostics
Presents responsive bordered card containers for **ARIMA(4,1,0)**, **SARIMA(2,1,2)(1,1,1)₇**, and **Meta Prophet (with 281 holiday windows)**. Displays benchmark deltas, test-window multi-model overlays, and expandable Statsmodels diagnostic summaries (Ljung-Box, Jarque-Bera, AIC/BIC).

![Tab 3 — Classical Models](assets/dashboard_classical_models.png)

---

### Tab 5: Explainable AI (SHAP) — Global & Local Attribution
Delivers transparent machine learning interpretability via a side-by-side analytical layout:
- **Global Feature Importance (Left):** TreeExplainer ranking of the top 12 sales drivers (promotions, autoregressive lags, rolling statistics, calendar harmonics).
- **Local Anomaly Deep Dive (Right):** Interactive anomaly event selector with a directional waterfall breakdown identifying exact causal factors behind demand anomalies.

![Tab 4 — SHAP Explainer](assets/dashboard_shap_explainer.png)

---

### Tab 6: Comprehensive Model Benchmark & Component Comparison
Renders the complete cross-model benchmark matrix under apples-to-apples national aggregation, comparing LightGBM, SARIMA, Prophet + Holidays, Prophet Basic, and ARIMA with error distributions, component analyses, and residual spreads.

![Tab 5 — Model Comparison](assets/dashboard_model_comparison.png)

---

### Tab 7: Autonomous GenAI Analyst & Decision Copilot
A conversational decision copilot powered by the **Google Gemini API** grounded via Retrieval-Augmented Generation (RAG) in Ecuadorian retail history. Features dynamic tool use and function calling to execute live database queries, correlate external macroeconomic events, and provide prescriptive inventory recommendations.

![Tab 6 — GenAI Analyst](assets/dashboard_genai_analyst.png)

---

## 🏗️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline["1. Ingestion & Preprocessing"]
        RAW["Raw Kaggle Data<br/>(3.4M Transactions, Oil, Holidays, Stores)"]
        CLEAN["Date Coercion & Imputation<br/>(Linear Oil Resample, Locale Hierarchy)"]
        FEAT["Feature Store Engine<br/>(Lags [1,7,14,28], Rolling Stats, Sin/Cos Cycles)"]
        RAW --> CLEAN --> FEAT
    end

    subgraph Dual_Track_Modeling["2. Dual-Track Forecasting Engines"]
        FEAT -->|"Top-Down National Sum"| NAT_SERIES["National Daily Series<br/>(1 Aggregate Series)"]
        FEAT -->|"Bottom-Up Granular Slices"| GRAN_SERIES["Store × Family Slices<br/>(1,782 Independent Series)"]
        
        NAT_SERIES --> ARIMA["ARIMA(4,1,0)<br/>Auto-regressive Baseline"]
        NAT_SERIES --> SARIMA["SARIMA(2,1,2)(1,1,1)7<br/>Weekly Seasonality + 5-Fold Walk-Forward"]
        NAT_SERIES --> PROPHET["Meta Prophet<br/>Multiplicative + 281 Holidays"]
        
        GRAN_SERIES --> LGBM["LightGBM Regressor<br/>Leaf-wise Histogram Boosting"]
    end

    subgraph Granularity_Alignment["3. Granularity Alignment Engine"]
        LGBM -->|"Bottom-Up Summation<br/>∑(Stores × Families)"| LGBM_NAT["LightGBM National Aggregate<br/>MAE: 43,947 | MAPE: 5.42%"]
        ARIMA --> COMP["Master Model Benchmark<br/>Apples-to-Apples Evaluation"]
        SARIMA --> COMP
        PROPHET --> COMP
        LGBM_NAT --> COMP
    end

    subgraph XAI_GenAI["4. Explainability & Autonomous Intelligence"]
        LGBM --> SHAP["SHAP TreeExplainer<br/>(Global Beeswarm & Anomaly Waterfall)"]
        FEAT --> RAG["TF-IDF Knowledge Base<br/>(Earthquake 2016, Oil Slump, Payday)"]
        RAG --> GEMINI["Gemini GenAI Analyst<br/>(Dynamic Tool Use & Function Calling)"]
    end

    subgraph UI["5. Streamlit Delivery Platform"]
        COMP --> DASH["Streamlit Interactive Dashboard<br/>(Responsive KPI Ribbon, 7 Analytical Tabs)"]
        SHAP --> DASH
        GEMINI --> DASH
    end
```

---

## 📊 Visual Proof of Work Gallery

### Part A: Exploratory Data Analysis & Macro Volatility

| Analysis | Visualization | Strategic Takeaway |
|---|---|---|
| **Multi-Year National Sales Trajectory** | ![National Sales Trend](assets/eda_national_sales_trend.png) | Sales grew steadily from 2013 through 2017 with strong annual December holiday volume surges and consistent intra-year compounding. |
| **WTI Crude Oil Collapse vs. Retail Sales** | ![Oil vs Sales Macro](assets/eda_oil_vs_sales_macro.png) | Despite crude oil crashing from $105 to under $30/barrel in 2014–2015, grocery revenue expanded counter-cyclically due to essential consumer staples demand. |
| **April 2016 7.8M Earthquake Shock** | ![Earthquake Impact](assets/eda_earthquake_impact.png) | The April 16, 2016 Pedernales earthquake triggered an immediate +300% emergency demand spike across canned goods, bottled water, and staples. |
| **7-Day Seasonal Decomposition** | ![Seasonal Decomposition](assets/eda_seasonal_decomposition.png) | 4-panel multiplicative decomposition isolating strong 7-day cyclical oscillations, macro trend, and low variance residuals. |
| **Retail Calendar & Periodic Patterns** | ![Retail Calendar](assets/eda_retail_calendar_patterns.png) | Pronounced weekend demand peaks (Saturdays and Sundays account for 38% of weekly volume) combined with massive December holiday lifts. |
| **Public Sector Payday ("Quincena") Effect** | ![Payday Effect](assets/eda_payday_effect.png) | Ecuadorian biweekly government paydays on the 15th and last day of each month generate a statistically significant **+14.8% sales lift**. |

---

### Part B: Forecasting Models & Explainable AI

| Model / Methodology | Visualization | Diagnostic & Evaluation Details |
|---|---|---|
| **ARIMA(4,1,0) Baseline Diagnostics** | ![ARIMA Forecast Diagnostics](assets/arima_forecast_diagnostics.png) | Out-of-sample forecast with 95% confidence intervals, residual histogram, and Normal Q-Q plot showing unmodeled weekly autocorrelation. |
| **SARIMA(2,1,2)(1,1,1)₇ Walk-Forward CV** | ![SARIMA Walk-Forward](assets/sarima_forecast_walkforward.png) | 5-fold expanding window cross-validation validating robust convergence across varying seasonal retail horizons. |
| **Meta Prophet + 281 Holiday Windows** | ![Prophet Forecast](assets/prophet_forecast_vs_actual.png) | Multiplicative seasonal decomposition capturing complex holiday windows ([-2, +1] days) including Carnaval, Semana Santa, and Navidad. |
| **Global SHAP Beeswarm Summary** | ![SHAP Beeswarm Summary](assets/shap_beeswarm_summary.png) | TreeExplainer attribution showing that 7-day autoregressive lag (`sales_lag_7`), promotions (`onpromotion`), and 14-day rolling mean dominate model gain. |
| **Local Anomaly Attribution Waterfall** | ![SHAP Anomaly Breakdown](assets/shap_anomaly_breakdown.png) | High-resolution local attribution isolating why specific promotion days drove severe positive demand spikes beyond baseline expectations. |
| **Store 1 / Grocery I Forecast Slice** | ![Forecast Explorer Sample](assets/forecast_explorer_sample.png) | Fine-grained Store 1 Grocery I predictions tracking actual sales through volatile promotional periods with tight tracking bounds. |

---

## 🏆 Head-to-Head Model Benchmark & Granularity Derivation

### Master Performance Benchmark

All models are evaluated on the exact same holdout test window (last 16 days of available historical data across all 54 stores and 33 product families).

| Model Architecture | Evaluation Level | MAE (Units) | RMSE (Units) | MAPE (%) | SMAPE (%) | Key Architectural Strengths |
|---|---|---|---|---|---|---|
| 🥇 **LightGBM (Champion)** | **Bottom-Up National Aggregate** | **43,947.25** | **52,489.81** | **5.42%** | **4.81%** *(Row: 70.33%)* | Captures complex non-linear lag interactions and promotion elasticity across 1,782 series simultaneously. |
| 🥈 **SARIMA(2,1,2)(1,1,1)₇** | National Daily Aggregate | 86,617.79 | 136,302.65 | 33.36% | 10.14% | Parametric modeling of 7-day weekly retail periodicity with seasonal moving average terms. |
| 🥉 **Prophet (Basic)** | National Daily Aggregate | 88,684.28 | 131,053.92 | 36.42% | — | Automatic piecewise linear trend changepoint detection capturing macroeconomic shifts. |
| 4️⃣ **Prophet + Holidays** | National Daily Aggregate | 92,550.50 | 132,817.01 | 32.77% | — | Multiplicative seasonality explicitly parameterized with 281 Ecuadorian national and regional holiday windows. |
| 5️⃣ **ARIMA(4,1,0)** | National Daily Aggregate | 151,425.15 | 186,176.22 | 38.59% | 17.83% | First-differenced autoregressive baseline without seasonal terms. |

---

### The Granularity Alignment Derivation: Resolving the Retail Fallacy

A frequent error in retail time-series evaluation is comparing row-level machine learning metrics directly to national-level statistical metrics:
- **Granular ML Formulation:** The dataset contains $S = 54$ stores and $F = 33$ merchandise families ($S \times F = 1,782$ individual series). LightGBM predicts unit demand $y_{s,f,t}$ for store $s$, family $f$, on date $t$. At this granular slice, daily volume frequently averages 200–500 units, yielding a row MAE of ~300–450.
- **Classical Formulation:** Classical models (ARIMA, SARIMA, Prophet) are fitted to the aggregated national time series $Y_t^{\text{nat}} = \sum_{s=1}^{54} \sum_{f=1}^{33} y_{s,f,t}$, where daily volume fluctuates between 700,000 and 1,200,000+ units. Naturally, national MAEs are in the tens of thousands.

#### Bottom-Up Aggregation Proof
To establish a mathematically rigorous, apples-to-apples comparison, we aggregate granular LightGBM predictions to the national level for each test date $t$:

$$\hat{Y}_{t}^{\text{nat}} = \sum_{s=1}^{54} \sum_{f=1}^{33} \hat{y}_{s,f,t}$$

Actual national volume is defined as:

$$Y_{t}^{\text{nat}} = \sum_{s=1}^{54} \sum_{f=1}^{33} y_{s,f,t}$$

The national error metrics are then computed over the test horizon $T$:

$$\text{MAE}^{\text{nat}} = \frac{1}{|T|} \sum_{t \in T} \left| Y_{t}^{\text{nat}} - \hat{Y}_{t}^{\text{nat}} \right| = 43,947.25$$

$$\text{MAPE}^{\text{nat}} = \frac{1}{|T|} \sum_{t \in T} \frac{\left| Y_{t}^{\text{nat}} - \hat{Y}_{t}^{\text{nat}} \right|}{Y_{t}^{\text{nat}}} \times 100\% = 5.42\%$$

#### Why Bottom-Up Machine Learning Wins
1. **Preservation of Micro-Dynamics:** Aggregate national models smooth out localized phenomena (e.g., store-specific stockouts, regional holiday transfers, or local promotions). Bottom-up ML models each store and family individually, capturing localized nuances before summation.
2. **Cancellation of Uncorrelated Residuals:** Under bottom-up summation, uncorrelated over-predictions in Store $A$ cancel against under-predictions in Store $B$:
   $$\text{Var}\left(\sum_{i=1}^N \epsilon_i\right) = \sum_{i=1}^N \text{Var}(\epsilon_i) \quad \text{(for independent errors)}$$
   This statistical smoothing drives national aggregated error down to **43,947**, achieving a **-49.3% error reduction** compared to SARIMA (**86,618**).

---

## 📓 Modular Notebook Suite Guide

The end-to-end research, engineering, and model validation pipeline is organized into 8 modular, self-contained Jupyter notebooks in `notebooks/`:

| Notebook | Scope & Title | Core Methodology & Techniques | Generated Artifacts & Reports |
|---|---|---|---|
| [`01_data_cleaning.ipynb`](notebooks/01_data_cleaning.ipynb) | Data Cleaning & Preprocessing | Ingestion of 3.4M raw transactions, linear interpolation & backward fill for missing oil prices, holiday deduplication with regional hierarchy resolution, and master relational joins. | `clean_merged.csv` |
| [`02_exploratory_data_analysis.ipynb`](notebooks/02_exploratory_data_analysis.ipynb) | Exploratory Data Analysis (EDA) | National sales aggregation, WTI macro oil correlation, April 2016 7.8M earthquake shock analysis, multiplicative 7-day seasonal decomposition, and retail calendar analysis. | `assets/eda_*.png` (6 high-res plots) |
| [`03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb) | Stationarity & Feature Engineering | Augmented Dickey-Fuller (ADF) stationarity testing, differencing engine, sin/cos cyclical calendar transforms, leakage-safe autoregressive lags (1, 7, 14, 28), and rolling/expanding statistics. | `feature_store.csv` (23 features) |
| [`04_arima_forecasting.ipynb`](notebooks/04_arima_forecasting.ipynb) | Classical ARIMA Forecasting | Chronological train/test split, ARIMA(4,1,0) model fitting on national sales, residual diagnostics (Ljung-Box, Jarque-Bera, Normal Q-Q), and out-of-sample forecast with 95% confidence intervals. | `arima_forecast_diagnostics.png`, `reports/arima_forecast.csv` |
| [`05_sarima_forecasting.ipynb`](notebooks/05_sarima_forecasting.ipynb) | Seasonal ARIMA (SARIMA) | SARIMA(2,1,2)(1,1,1)₇ model fitting capturing 7-day weekly retail periodicity, WMAPE & SMAPE metric tracking, and 5-fold expanding window walk-forward validation grid. | `sarima_forecast_walkforward.png`, `reports/sarima_forecast.csv` |
| [`06_prophet_forecasting.ipynb`](notebooks/06_prophet_forecasting.ipynb) | Meta Prophet & Event Modeling | Prophet data preparation, additive vs. multiplicative seasonality selection, modeling 281 Ecuadorian holidays with explicit windows ([-2, +1] days), changepoint detection, and component exports. | `prophet_forecast_vs_actual.png`, `reports/prophet_summary.json` |
| [`07_lightgbm_forecasting.ipynb`](notebooks/07_lightgbm_forecasting.ipynb) | LightGBM & SHAP Explainability | Feature specification, 80/20 chronological partition, LightGBM booster training, row-level evaluation metrics, feature gain importance, SHAP TreeExplainer beeswarm, and local anomaly waterfall breakdown. | `lgbm_model.pkl`, `shap_beeswarm_summary.png`, `shap_anomaly_breakdown.png` |
| [`08_model_comparison_and_evaluation.ipynb`](notebooks/08_model_comparison_and_evaluation.ipynb) | Model Comparison & Validation | Bottom-up granularity alignment (summing row-level predictions to national daily totals), master comparison table compilation, comparative visualizations, automated test suite, and SHA-256 model checksum manifest. | `model_comparison_table.csv`, `models/checksums.json` |

---

## 🔒 Production Verification & Model Integrity

To ensure reproducibility, model governance, and zero data leakage across production environments, every serialized artifact is verified against an immutable SHA-256 cryptographic manifest recorded in `models/checksums.json`:

```json
{
  "cluster_encoder.pkl": "705731407764bdf8b0c5ef55c71e7ec6a60e283423ac216c9d36c4543cf86d38",
  "family_encoder.pkl": "8bdc907334fbb53f2d65d79d080c7bf8c5abf903b58678ce147451dfc94565d2",
  "holiday_type_encoder.pkl": "98b6e0b7f35631567d6643f4378cfb92173946fc358f732beef944925133cf63",
  "lgbm_model.pkl": "ebbd9cd704c45d153a523c557d57aeff5f261d3cd963803575351020dc9e4ab8",
  "oil_scaler.pkl": "e518bb7606e5e202eb99f287964a06a1d8297790583119e4f9a1807f7c781b46",
  "sales_scaler.pkl": "c0c77fbe7b4671a2e19859109f212916de48b95fc4267ab442cda509d42ddb20",
  "sarima_212_1117_store1_grocery12.pkl": "4df073da665a7128fb7f4e12831f084290f99cf6f8921d8b4039f135d7fb56d6",
  "type_x_encoder.pkl": "05753f7996739209586dcb853eb6a0c1f4cfe0a9ee83421d4764b37f639d1299",
  "vanilla_lstm_best.keras": "bb076cc146f9d76e0dcdbc8ab0acabd38f01424fdb05982d02781bd68e475f76",
  "vanilla_lstm_final.keras": "d7c63b8af81c0193b7dbbc7033936ead755d444cb05d962b49b4848a0ac00ab5"
}
```

---

## 🚀 Setup, Installation & Execution Guide

### Prerequisites
- **Python**: Version `3.11` or higher recommended.
- **Git**: For version control and cloning.

### 1. Clone the Repository
```bash
git clone https://github.com/Elite407/sales-forecasting.git
cd sales-forecasting
```

### 2. Create and Activate a Virtual Environment
```bash
# On Linux/macOS
python -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Streamlit Secrets (for GenAI Copilot)
Create a `.streamlit/secrets.toml` file from the provided template:
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```
Open `.streamlit/secrets.toml` and supply your Google Gemini API key:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```
*(Note: If no API key is provided, the dashboard automatically boots in **Demo Mode**, allowing full inspection of all classical, machine learning, and explainability tabs).*

### 5. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/app.py
```
The application will launch locally at `http://localhost:8501`.

### 6. Run the Modular Jupyter Notebook Suite
```bash
jupyter lab notebooks/
```

---

## 📁 Repository Structure

```
sales-forecasting/
├── .streamlit/
│   ├── config.toml                     # Clean crisp light mode configuration
│   ├── secrets.toml.example            # Template for Gemini API secrets
│   └── secrets.toml                    # Local secrets (git-ignored)
├── app/
│   ├── app.py                          # Streamlit application with KPI ribbon & sidebar
│   ├── config.py                       # Global paths and styling tokens
│   ├── data_loader.py                  # Cached data loader and feature store connector
│   ├── tabs/
│   │   ├── overview.py                 # Tab 0: Executive overview & macro narrative
│   │   ├── forecast_explorer.py        # Tab 1: Interactive store/family slider & export
│   │   ├── category_trends.py          # Tab 2: Multi-family category trends
│   │   ├── classical_models.py         # Tab 3: Responsive card containers for ARIMA/SARIMA/Prophet
│   │   ├── shap_explainer.py           # Tab 4: Side-by-side global & local SHAP
│   │   └── model_comparison.py        # Tab 5: Master model benchmark & components
│   └── genai/
│       ├── agent.py                    # Tab 6: Conversational Gemini decision agent
│       └── tools.py                    # Tool calling & data layer functions
├── assets/                             # 12 proof-of-work plots + 7 dashboard tab screenshots
├── models/
│   └── checksums.json                  # SHA-256 cryptographic manifest of 11 model artifacts
├── notebooks/                          # Modular suite (01_ through 08_)
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_arima_forecasting.ipynb
│   ├── 05_sarima_forecasting.ipynb
│   ├── 06_prophet_forecasting.ipynb
│   ├── 07_lightgbm_forecasting.ipynb
│   └── 08_model_comparison_and_evaluation.ipynb
├── reports/
│   ├── prophet_summary.json            # Prophet configuration & evaluated metrics
│   ├── model_comparison_table.csv      # National benchmark CSV
│   ├── arima_forecast.csv
│   ├── sarima_forecast.csv
│   └── prophet_forecast.csv
├── scripts/
│   ├── generate_all_plots.py           # Generates all 12 high-res EDA and model plots
│   └── capture_dashboard_screenshots.py # Headless Chrome CDP screenshot pipeline
├── requirements.txt                    # Python package dependencies
├── LICENSE                             # Standard MIT open source license
└── README.md                           # Modernized enterprise showcase
```

---

## 📜 License & Acknowledgments

- **Dataset:** Corporación Favorita Grocery Sales Forecasting on [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).
- **License:** Open-sourced under the [MIT License](LICENSE).
