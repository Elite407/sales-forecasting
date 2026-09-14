# Corporación Favorita Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square&logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)


This repository contains a comprehensive time-series forecasting pipeline and interactive web application for predicting retail store sales, based on the Corporación Favorita Kaggle dataset.

*(Placeholder: Add a 15-second demo GIF of the Streamlit app here)*

The project evaluates both classical statistical baselines (ARIMA, SARIMA, Prophet) and modern machine learning approaches (LightGBM), ensuring a granularity-consistent comparison. It also features a fully interactive Streamlit dashboard enhanced with SHAP explainability and an integrated Retrieval-Augmented Generation (RAG) Generative AI Analyst.

## Features

- **Interactive Forecast Explorer:** Analyze historical sales and future predictions at the granularity of individual stores and product families.
- **Category Trends:** Evaluate aggregated sales trends across multiple product categories to monitor national-level demand shifts.
- **Explainable AI (SHAP):** 
  - Global feature importance to identify primary drivers of sales.
  - Local anomaly explanations detailing which features pushed statistical anomalies up or down.
- **GenAI Analyst:** A built-in chat agent powered by the Gemini API and RAG. The agent grounds its answers in real-world context (e.g., the 2016 Ecuador earthquake, WTI oil price collapse, localized holiday transfers) and queries the underlying dataset dynamically via function calling.

## Model Results

Head-to-head performance evaluation between classical baselines (ARIMA, SARIMA, Prophet) and modern ML (LightGBM).

| Model | MAE | RMSE | MAPE (%) | Notes |
|-------|-----|------|----------|-------|
| **LightGBM** | ~108* | ~529* | 95.5% | Fast, captures nonlinear interactions. *Note: Currently evaluated at store/family row-level vs. national aggregate.* |
| **SARIMA (2,1,2)(1,1,1)[7]** | 86,617 | 136,302 | 33.3% | Strong baseline capturing weekly seasonality on national aggregate. |
| **Prophet + Holidays** | 92,550 | 132,817 | 32.7% | Excellent handling of massive holiday spikes using explicit event flags. |
| **ARIMA (4,1,0)** | 151,425 | 186,176 | 38.5% | Basic auto-regressive baseline. |

*(Deep learning approaches including LSTM, TFT, and N-BEATS are currently under development).*

## Repository Structure

- `/app`: Contains the main Streamlit application script (`app.py`).
- `/data`: Includes the processed datasets (e.g., `feature_store.csv`).
- `/models`: Contains the serialized LightGBM models and scikit-learn encoders (`lgbm_model.pkl`, `family_encoder.pkl`, etc.).
- `/notebooks`: Contains `sales_forecasting.ipynb` for data exploration, full-scale model training, and exporting artifacts.
- `/reports`: Stores exported classical model summaries, error metrics, and model comparison charts.
- `requirements.txt`: Python dependencies required to run the dashboard.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Elite407/sales-forecasting.git
   cd sales-forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Copy the example secrets file and configure your API keys to enable the GenAI Analyst feature.
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Add your `GEMINI_API_KEY` to the `.streamlit/secrets.toml` file.

4. **Run the application:**
   ```bash
   streamlit run app/app.py
   ```
