import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from ..data_loader import load_comparison
from ..config import MODEL_FEATURES, FEATURE_LABELS, FAMILY_NAMES

GENAI_KNOWLEDGE_BASE = [
    "On April 16, 2016, a 7.8-magnitude earthquake struck Ecuador near Pedernales. "
    "Relief efforts in the following weeks led people to donate water and other "
    "essential goods, which significantly affected supermarket sales nationally "
    "for several weeks after the earthquake.",

    "Ecuador's economy is oil-dependent and WTI oil prices collapsed from over "
    "$100/barrel in mid-2014 to under $30/barrel by early 2016, straining "
    "government spending and consumer demand through much of the sample period.",

    "Holidays in this dataset are typed as Local, Regional, or National in scope, "
    "and some are 'transferred' to a different date than their traditional one, or "
    "marked as a 'Bridge' day added to extend a holiday, or a 'Work Day' that makes "
    "up for a bridge day elsewhere — the transferred date is the one that actually "
    "affects sales, not the traditional date.",

    "Public sector wages in Ecuador are paid on the 15th and on the last day of "
    "each month, which tends to produce a recurring sales bump in the days "
    "immediately following those paydays.",

    "Ecuador has used the US dollar as its official currency since 2000, so sales "
    "figures in this dataset are not affected by local currency devaluation risk "
    "the way some other Latin American retail datasets can be.",

    "The 'onpromotion' signal indicates how many items in a product family were "
    "on promotion on a given day; promo intensity is one of the stronger drivers "
    "the LightGBM model picks up in SHAP, alongside recent lag and rolling-mean sales.",

    "December is the strongest seasonal period in this dataset across most "
    "product families, especially GROCERY, BEVERAGES, and LIQUOR/WINE/BEER, "
    "consistent with Christmas and New Year retail demand.",

    "Rows flagged `is_anomaly = 1` in this pipeline are statistically detected "
    "outliers (unusually large deviations from recent rolling behavior), not "
    "confirmed data-quality errors — a flagged spike can reflect a real event "
    "(like the earthquake or a big promotion) rather than a mistake in the data.",

    "LightGBM in this project is trained at row level (store x product family x "
    "day), while ARIMA, SARIMA, and Prophet are all fit on the national daily "
    "aggregate — so their error metrics are not directly comparable on a like-for-"
    "like basis without accounting for that granularity difference.",

    "Back-to-school shopping season in Ecuador falls at different times in "
    "different regions — around April-May on the coast and around August-"
    "September in the highlands — which can show up as regional demand bumps "
    "in school/office supplies and related categories that a national-only "
    "model would smooth over.",
]

@st.cache_resource
def _get_genai_kb_index():
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(GENAI_KNOWLEDGE_BASE)
    return vectorizer, doc_matrix

def genai_retrieve_context(query, k=3):
    vectorizer, doc_matrix = _get_genai_kb_index()
    q_vec = vectorizer.transform([query])
    sims = (doc_matrix @ q_vec.T).toarray().ravel()
    top_idx = sims.argsort()[::-1][:k]
    return [GENAI_KNOWLEDGE_BASE[i] for i in top_idx if sims[i] > 0]

# Try importing shap for the tool
try:
    import shap as _genai_shap

    @st.cache_resource
    def _get_genai_shap_explainer(_model):
        return _genai_shap.TreeExplainer(_model)

    _GENAI_SHAP_AVAILABLE = True
except ImportError:
    _GENAI_SHAP_AVAILABLE = False


def build_tools_dict(df, model):
    def _tool_get_model_comparison(**_):
        try:
            comp = load_comparison()
            return comp.to_dict(orient="records")
        except Exception as e:
            return {"error": f"Could not load model comparison table: {e}"}

    def _tool_get_shap_drivers(store=None, family=None, date=None, **_):
        if not _GENAI_SHAP_AVAILABLE:
            return {"error": "The `shap` package isn't installed in this environment."}
        try:
            target_date = pd.to_datetime(date)
        except Exception:
            return {"error": f"Could not parse date '{date}'. Use YYYY-MM-DD format."}

        row = df[(df["store_nbr"] == store) & (df["family"] == family) & (df["date"] == target_date)]
        if row.empty:
            return {
                "error": (
                    f"No row found for store={store}, family={family}, date={date} in the "
                    f"loaded data window."
                )
            }
        row = row.iloc[0]
        X_row = row[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        X_row_df = pd.DataFrame([X_row.values], columns=MODEL_FEATURES)

        explainer = _get_genai_shap_explainer(model)
        shap_vals = explainer.shap_values(X_row_df)[0]
        contrib = sorted(zip(MODEL_FEATURES, shap_vals), key=lambda t: -abs(t[1]))[:5]

        return {
            "actual_sales": float(row["sales"]),
            "top_drivers": [
                {"feature": FEATURE_LABELS.get(f, f), "shap_contribution": round(float(v), 2)}
                for f, v in contrib
            ],
        }

    def _tool_get_anomalies(store=None, family=None, **_):
        subset = df[(df["store_nbr"] == store) & (df["family"] == family) & (df["is_anomaly"] == 1)]
        if subset.empty:
            return {"anomaly_dates": [], "note": "No flagged anomalies for this store/family in the loaded window."}
        return {"anomaly_dates": subset[["date", "sales"]].astype(str).to_dict(orient="records")}

    def _tool_get_forecast_metrics(store=None, family=None, **_):
        subset = df[(df["store_nbr"] == store) & (df["family"] == family)].sort_values("date")
        if subset.empty:
            return {"error": "No data for this store/family in the loaded window."}
        X = subset[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        valid_mask = ~X.isna().any(axis=1)
        X = X.loc[valid_mask]
        subset = subset.loc[valid_mask]
        if X.empty:
            return {"error": "All rows had missing features after coercion — cannot predict."}
        preds = model.predict(X.values)
        mae = float(np.mean(np.abs(subset["sales"].values - preds)))
        rmse = float(np.sqrt(np.mean((subset["sales"].values - preds) ** 2)))
        return {
            "store": store,
            "family": FAMILY_NAMES.get(family, family),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "avg_daily_sales": round(float(subset["sales"].mean()), 2),
            "n_days": int(len(subset)),
        }

    def _tool_search_context(query=None, **_):
        return {"context_snippets": genai_retrieve_context(query or "")}

    return {
        "get_model_comparison": _tool_get_model_comparison,
        "get_shap_drivers": _tool_get_shap_drivers,
        "get_anomalies": _tool_get_anomalies,
        "get_forecast_metrics": _tool_get_forecast_metrics,
        "search_context": _tool_search_context,
    }

def get_genai_declarations():
    try:
        from google import genai as _google_genai
    except ImportError:
        return []

    return [
        _google_genai.types.Tool(
            function_declarations=[
                _google_genai.types.FunctionDeclaration(
                    name="get_model_comparison",
                    description="Returns the model comparison table (error metrics per model, e.g. ARIMA/SARIMA/Prophet/LightGBM) as a list of records.",
                    parameters=_google_genai.types.Schema(
                        type=_google_genai.types.Type.OBJECT,
                        properties={},
                        required=[],
                    ),
                ),
                _google_genai.types.FunctionDeclaration(
                    name="get_shap_drivers",
                    description=(
                        "Returns the actual sales value and the top SHAP feature contributions driving "
                        "the LightGBM prediction for a specific store, product family, and date."
                    ),
                    parameters=_google_genai.types.Schema(
                        type=_google_genai.types.Type.OBJECT,
                        properties={
                            "store": _google_genai.types.Schema(type=_google_genai.types.Type.INTEGER, description="Store number, e.g. 3"),
                            "family": _google_genai.types.Schema(type=_google_genai.types.Type.INTEGER, description="Product family code (0-32)"),
                            "date": _google_genai.types.Schema(type=_google_genai.types.Type.STRING, description="Date in YYYY-MM-DD format"),
                        },
                        required=["store", "family", "date"],
                    ),
                ),
                _google_genai.types.FunctionDeclaration(
                    name="get_anomalies",
                    description="Lists dates flagged as statistical sales anomalies for a specific store and product family.",
                    parameters=_google_genai.types.Schema(
                        type=_google_genai.types.Type.OBJECT,
                        properties={
                            "store": _google_genai.types.Schema(type=_google_genai.types.Type.INTEGER),
                            "family": _google_genai.types.Schema(type=_google_genai.types.Type.INTEGER),
                        },
                        required=["store", "family"],
                    ),
                ),
                _google_genai.types.FunctionDeclaration(
                    name="get_forecast_metrics",
                    description="Returns LightGBM prediction accuracy (MAE, RMSE, average daily sales) for a specific store and product family over the loaded data window.",
                    parameters=_google_genai.types.Schema(
                        type=_google_genai.types.Type.OBJECT,
                        properties={
                            "store": _google_genai.types.Schema(type=_google_genai.types.Type.INTEGER),
                            "family": _google_genai.types.Schema(type=_google_genai.types.Type.INTEGER),
                        },
                        required=["store", "family"],
                    ),
                ),
                _google_genai.types.FunctionDeclaration(
                    name="search_context",
                    description=(
                        "Searches a knowledge base of real-world context about this dataset (the 2016 "
                        "Ecuador earthquake, oil price shocks, holiday calendar quirks, payday effects, "
                        "etc.) for information relevant to a query."
                    ),
                    parameters=_google_genai.types.Schema(
                        type=_google_genai.types.Type.OBJECT,
                        properties={
                            "query": _google_genai.types.Schema(type=_google_genai.types.Type.STRING),
                        },
                        required=["query"],
                    ),
                ),
            ]
        )
    ]
