import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..config import MODEL_FEATURES, FEATURE_LABELS, FAMILY_NAMES

def render(df, model):
    st.header("SHAP Explainability")

    try:
        import shap
    except ImportError:
        st.error("The `shap` package isn't installed. Add `shap` to requirements.txt.")
        return

    @st.cache_resource
    def get_explainer(_model):
        return shap.TreeExplainer(_model)

    @st.cache_data
    def get_shap_sample(_df, sample_size=500, seed=42):
        clean = _df.dropna(subset=MODEL_FEATURES)
        n = min(sample_size, len(clean))
        sample = clean.sample(n=n, random_state=seed)
        X_sample = sample[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        return sample, X_sample

    explainer = get_explainer(model)
    sample_rows, X_sample = get_shap_sample(df)

    with st.spinner("Computing SHAP values on a 500-row sample…"):
        shap_values_sample = explainer.shap_values(X_sample)

    st.subheader("Global feature impact")
    mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "label": [FEATURE_LABELS.get(f, f) for f in MODEL_FEATURES],
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=True).tail(15)

    fig_imp = go.Figure(go.Bar(
        x=importance_df["mean_abs_shap"], y=importance_df["label"],
        orientation="h", marker_color="#636EFA",
    ))
    fig_imp.update_layout(height=500, xaxis_title="Mean |SHAP value| (impact on predicted sales)",
                           yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    # -------------------------------------------------------
    # Anomaly explanations
    # -------------------------------------------------------
    st.subheader("Why was this flagged as an anomaly?")

    if "is_anomaly" not in df.columns or df["is_anomaly"].sum() == 0:
        st.info("No rows are flagged with `is_anomaly = 1` in the current data window.")
    else:
        anomalies = df[df["is_anomaly"] == 1].copy()
        st.caption(f"{len(anomalies):,} rows are flagged as anomalies in this data window.")

        max_options = 300
        if len(anomalies) > max_options:
            anomalies_display = anomalies.sample(n=max_options, random_state=1).sort_values("date")
            st.caption(f"Showing a random sample of {max_options} for the picker below.")
        else:
            anomalies_display = anomalies.sort_values("date")

        anomalies_display = anomalies_display.reset_index(drop=True)
        anomalies_display["label"] = anomalies_display.apply(
            lambda r: f"{r['date'].date()} — Store {int(r['store_nbr'])} — "
                      f"{FAMILY_NAMES.get(int(r['family']), r['family'])} — actual sales {r['sales']:.1f}",
            axis=1,
        )

        chosen_label = st.selectbox("Pick a flagged anomaly to explain", anomalies_display["label"])
        chosen_row = anomalies_display[anomalies_display["label"] == chosen_label].iloc[0]

        row_X = chosen_row[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        row_X_df = pd.DataFrame([row_X.values], columns=MODEL_FEATURES)

        row_shap = explainer.shap_values(row_X_df)[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        predicted = base_value + row_shap.sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Actual sales", f"{chosen_row['sales']:.1f}")
        m2.metric("Model predicted", f"{predicted:.1f}")
        m3.metric("Baseline (avg. prediction)", f"{base_value:.1f}")

        contrib_df = pd.DataFrame({
            "feature": MODEL_FEATURES,
            "label": [FEATURE_LABELS.get(f, f) for f in MODEL_FEATURES],
            "value": row_X.values,
            "shap": row_shap,
        })
        contrib_df["abs_shap"] = contrib_df["shap"].abs()
        top_contrib = contrib_df.sort_values("abs_shap", ascending=False).head(8)

        fig_wf = go.Figure(go.Bar(
            x=top_contrib.sort_values("shap")["shap"],
            y=top_contrib.sort_values("shap")["label"],
            orientation="h",
            marker_color=["#d62728" if v < 0 else "#2ca02c" for v in top_contrib.sort_values("shap")["shap"]],
        ))
        fig_wf.update_layout(
            height=400, xaxis_title="SHAP contribution (push toward higher / lower sales)",
            yaxis_title="", title="Top drivers for this prediction",
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("**In plain terms:**")
        narrative_top = contrib_df.sort_values("abs_shap", ascending=False).head(4)
        bullets = []
        for _, r in narrative_top.iterrows():
            direction = "pushed the prediction **up**" if r["shap"] > 0 else "pushed the prediction **down**"
            bullets.append(
                f"- **{r['label']}** (value: {r['value']:.2f}) {direction} by about {abs(r['shap']):.1f} units."
            )
        st.markdown("\n".join(bullets))

        residual = chosen_row["sales"] - predicted
        if abs(residual) > 0.1 * max(abs(predicted), 1):
            st.caption(
                f"Note: the model's prediction still misses the actual value by "
                f"~{residual:.1f} units — the flagged features above explain the "
                "model's prediction, not necessarily why the true sales spiked/dropped."
            )
