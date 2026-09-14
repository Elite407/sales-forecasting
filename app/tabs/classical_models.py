import streamlit as st
import plotly.graph_objects as go
from ..data_loader import load_text_artifact, load_json_artifact, load_csv_artifact
from ..config import (
    ARIMA_SUMMARY_PATH, SARIMA_SUMMARY_PATH, PROPHET_SUMMARY_PATH,
    ARIMA_FORECAST_PATH, SARIMA_FORECAST_PATH, PROPHET_FORECAST_PATH,
    PROPHET_IMG
)

def render():
    st.header("Classical Model Diagnostics")
    st.caption(
        "ARIMA, SARIMA, and Prophet were all refit on the same national daily "
        "aggregate for a granularity-consistent comparison. These summaries "
        "are exported once from the notebook (full multi-year series) rather "
        "than re-fit here, since the sample data bundled with this app is "
        "only a few weeks long."
    )

    arima_summary = load_text_artifact(ARIMA_SUMMARY_PATH)
    sarima_summary = load_text_artifact(SARIMA_SUMMARY_PATH)
    prophet_summary = load_json_artifact(PROPHET_SUMMARY_PATH)
    arima_fc = load_csv_artifact(ARIMA_FORECAST_PATH)
    sarima_fc = load_csv_artifact(SARIMA_FORECAST_PATH)
    prophet_fc = load_csv_artifact(PROPHET_FORECAST_PATH)

    have_any_forecast = any(x is not None for x in (arima_fc, sarima_fc, prophet_fc))

    if have_any_forecast:
        st.subheader("Test-window forecasts vs. actual (national daily sales)")
        fig = go.Figure()
        base = next(x for x in (arima_fc, sarima_fc, prophet_fc) if x is not None)
        if "actual" in base.columns:
            fig.add_trace(go.Scatter(x=base["date"], y=base["actual"], name="Actual",
                                      line=dict(color="black", width=2)))
        for name, fc, color in [
            ("ARIMA", arima_fc, "#1f77b4"),
            ("SARIMA", sarima_fc, "#2ca02c"),
            ("Prophet", prophet_fc, "#d62728"),
        ]:
            if fc is not None:
                y_col = "predicted" if "predicted" in fc.columns else "yhat"
                if y_col in fc.columns:
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc[y_col], name=name,
                                              line=dict(color=color, width=2, dash="dash")))
        fig.update_layout(height=450, xaxis_title="Date", yaxis_title="National daily sales")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_s, col_p = st.columns(3)

    with col_a:
        st.subheader("ARIMA")
        if arima_summary:
            with st.expander("Full statsmodels summary", expanded=False):
                st.code(arima_summary, language=None)
        else:
            st.info(f"`{ARIMA_SUMMARY_PATH}` not found yet — see export snippet below.")

    with col_s:
        st.subheader("SARIMA")
        if sarima_summary:
            with st.expander("Full statsmodels summary", expanded=False):
                st.code(sarima_summary, language=None)
        else:
            st.info(f"`{SARIMA_SUMMARY_PATH}` not found yet — see export snippet below.")

    with col_p:
        st.subheader("Prophet")
        if prophet_summary:
            st.write(f"**Growth:** {prophet_summary.get('growth', 'n/a')}")
            st.write(f"**Seasonalities:** {', '.join(prophet_summary.get('seasonalities', [])) or 'n/a'}")
            st.write(f"**Holidays modeled:** {prophet_summary.get('n_holidays', 'n/a')}")
            st.write(f"**Changepoints:** {prophet_summary.get('n_changepoints', 'n/a')}")
        else:
            st.info(f"`{PROPHET_SUMMARY_PATH}` not found yet — see export snippet below.")
        try:
            st.image(PROPHET_IMG, use_container_width=True, caption="Prophet components")
        except Exception:
            pass

    if not (arima_summary and sarima_summary and prophet_summary):
        with st.expander("📋 How to generate these artifacts from the notebook"):
            st.markdown(
                "Run this after fitting each model on the national aggregate "
                "(same cell that builds `national_classical_df`), then commit "
                "the output files under `reports/`:"
            )
            st.code(
                '''# --- ARIMA / SARIMA text summaries ---
with open("reports/arima_summary.txt", "w") as f:
    f.write(str(arima_model.summary()))
with open("reports/sarima_summary.txt", "w") as f:
    f.write(str(sarima_model.summary()))

# --- Prophet summary (Prophet objects aren't easily pickled/displayed) ---
import json
prophet_summary = {
    "growth": m_holidays.growth,
    "seasonalities": list(m_holidays.seasonalities.keys()),
    "n_holidays": len(m_holidays.holidays) if m_holidays.holidays is not None else 0,
    "n_changepoints": len(m_holidays.changepoints),
}
with open("reports/prophet_summary.json", "w") as f:
    json.dump(prophet_summary, f, indent=2)

# --- Forecast vs actual on the shared test window (for the overlay chart) ---
pd.DataFrame({"date": test.index, "actual": test.values,
              "predicted": arima_forecast.values}).to_csv("reports/arima_forecast.csv", index=False)
pd.DataFrame({"date": test.index, "actual": test.values,
              "predicted": sarima_forecast.values}).to_csv("reports/sarima_forecast.csv", index=False)
prophet_fc_test = m_holidays.predict(prophet_test[["ds"]])
pd.DataFrame({"date": prophet_test["ds"], "actual": prophet_test["y"].values,
              "yhat": prophet_fc_test["yhat"].values}).to_csv("reports/prophet_forecast.csv", index=False)''',
                language="python",
            )
