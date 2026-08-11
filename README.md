# sales-forecasting
Sales Forecasting using Time Series Models
# Corporación Favorita Sales Forecasting

This repository contains a comprehensive time-series forecasting pipeline and interactive web application for predicting retail store sales, based on the Corporación Favorita Kaggle dataset.

The project evaluates both classical statistical baselines (ARIMA, SARIMA, Prophet) and modern machine learning approaches (LightGBM), ensuring a granularity-consistent comparison. It also features a fully interactive Streamlit dashboard enhanced with SHAP explainability and an integrated Retrieval-Augmented Generation (RAG) Generative AI Analyst.

## Features

- **Interactive Forecast Explorer:** Analyze historical sales and future predictions at the granularity of individual stores and product families.
- **Category Trends:** Evaluate aggregated sales trends across multiple product categories to monitor national-level demand shifts.
- **Model Comparison:** Head-to-head performance evaluation between classical baselines (ARIMA, SARIMA, Prophet) and LightGBM models. (Note: Deep learning approaches including LSTM, TFT, and N-BEATS are currently under development).
- **Explainable AI (SHAP):** 
  - Global feature importance to identify primary drivers of sales.
  - Local anomaly explanations detailing which features pushed statistical anomalies up or down.
- **GenAI Analyst:** A built-in chat agent powered by the Gemini API and RAG. The agent grounds its answers in real-world context (e.g., the 2016 Ecuador earthquake, WTI oil price collapse, localized holiday transfers) and queries the underlying dataset dynamically via function calling.

## Repository Structure

- `/app`: Contains the main Streamlit application script (`app.py`).
- `/data`: Includes the processed datasets (e.g., `feature_store.csv`).
- `/models`: Contains the serialized LightGBM models and scikit-learn encoders (`lgbm_model.pkl`, `family_encoder.pkl`, etc.).
- `/notebooks`: Contains the Jupyter notebooks used for data exploration, full-scale model training, and exporting artifacts.
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
