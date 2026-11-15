import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..config import MODEL_FEATURES, FEATURE_LABELS, FAMILY_NAMES

def render(df, model):
    st.header("SHAP Explainability & Causal Attribution")
    st.caption(
        "Interpreting LightGBM predictions using Shapley Additive Explanations (SHAP) — "
        "bridging global model mechanics and local anomaly attribution."
    )

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

    with st.spinner("Computing SHAP values on 500-sample slice…"):
        shap_values_sample = explainer.shap_values(X_sample)

    # ---------------------------------------------------------
    # Side-by-Side 2-Column Layout
    # ---------------------------------------------------------
    col_global, col_local = st.columns([1, 1], gap="medium")

    # ---------------------------------------------------------
    # Left Column: Global Feature Importance
    # ---------------------------------------------------------
    with col_global:
        with st.container(border=True):
            st.subheader("🌐 Global Feature Importance")
            st.caption("Mean |SHAP value| impact on predicted daily sales across 500 sample observations.")

            mean_abs_shap = np.abs(shap_values_sample).mean(axis=0)
            importance_df = pd.DataFrame({
                "feature": MODEL_FEATURES,
                "label": [FEATURE_LABELS.get(f, f) for f in MODEL_FEATURES],
                "mean_abs_shap": mean_abs_shap,
            }).sort_values("mean_abs_shap", ascending=True).tail(12)

            fig_imp = go.Figure(go.Bar(
                x=importance_df["mean_abs_shap"],
                y=importance_df["label"],
                orientation="h",
                marker_color="#2563EB",
                text=[f"{v:.1f}" for v in importance_df["mean_abs_shap"]],
                textposition="auto",
            ))
            fig_imp.update_layout(
                height=480,
                xaxis_title="Mean |SHAP value| (Units of Daily Sales)",
                yaxis_title="",
                margin=dict(l=10, r=10, t=20, b=30),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown(
                """
                **Key Global Observations:**
                - **Lag Dominance**: `sales_lag_1` and `sales_lag_7` drive the overwhelming majority of predictive variance.
                - **Promotions**: `onpromotion` provides significant uplift, especially when interacting with holiday weeks.
                - **Macro Factors**: Oil price changes (`dcoilwtico`) and earthquake indicators exhibit localized non-linear influences.
                """
            )

    # ---------------------------------------------------------
    # Right Column: Local Anomaly Deep Dive
    # ---------------------------------------------------------
    with col_local:
        with st.container(border=True):
            st.subheader("🎯 Local Anomaly Deep Dive")
            st.caption("Shapley decomposition of individual flagged anomalous demand events.")

            if "is_anomaly" not in df.columns or df["is_anomaly"].sum() == 0:
                st.info("No rows are flagged with `is_anomaly = 1` in the current data window.")
            else:
                anomalies = df[df["is_anomaly"] == 1].copy()
                st.caption(f"Detected **{len(anomalies):,}** demand anomaly events in this data window.")

                max_options = 300
                if len(anomalies) > max_options:
                    anomalies_display = anomalies.sample(n=max_options, random_state=1).sort_values("date")
                else:
                    anomalies_display = anomalies.sort_values("date")

                anomalies_display = anomalies_display.reset_index(drop=True)
                anomalies_display["label"] = anomalies_display.apply(
                    lambda r: f"{r['date'].date()} — Store {int(r['store_nbr'])} — "
                              f"{FAMILY_NAMES.get(int(r['family']), r['family'])} (Sales: {r['sales']:,.0f})",
                    axis=1,
                )

                chosen_label = st.selectbox(
                    "Select Flagged Anomaly Event:",
                    anomalies_display["label"],
                    key="shap_anomaly_select"
                )
                chosen_row = anomalies_display[anomalies_display["label"] == chosen_label].iloc[0]

                row_X = chosen_row[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
                row_X_df = pd.DataFrame([row_X.values], columns=MODEL_FEATURES)

                row_shap = explainer.shap_values(row_X_df)[0]
                base_value = explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
                predicted = base_value + row_shap.sum()

                # 3 Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Actual Sales", f"{chosen_row['sales']:,.0f}")
                with m2:
                    st.metric("Model Predicted", f"{predicted:,.0f}")
                with m3:
                    st.metric("Expected Baseline", f"{base_value:,.0f}")

                # Waterfall contribution chart
                contrib_df = pd.DataFrame({
                    "feature": MODEL_FEATURES,
                    "label": [FEATURE_LABELS.get(f, f) for f in MODEL_FEATURES],
                    "value": row_X.values,
                    "shap": row_shap,
                })
                contrib_df["abs_shap"] = contrib_df["shap"].abs()
                top_contrib = contrib_df.sort_values("abs_shap", ascending=False).head(7)

                sorted_wf = top_contrib.sort_values("shap")
                colors = ["#EF4444" if v < 0 else "#10B981" for v in sorted_wf["shap"]]

                fig_wf = go.Figure(go.Bar(
                    x=sorted_wf["shap"],
                    y=sorted_wf["label"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.1f}" for v in sorted_wf["shap"]],
                    textposition="auto",
                ))
                fig_wf.update_layout(
                    height=280,
                    xaxis_title="SHAP Impact (Push toward higher / lower sales)",
                    yaxis_title="",
                    margin=dict(l=10, r=10, t=10, b=30),
                )
                st.plotly_chart(fig_wf, use_container_width=True)

                # Causal Narrative Bullets
                st.markdown("**Causal Drivers:**")
                narrative_top = contrib_df.sort_values("abs_shap", ascending=False).head(3)
                for _, r in narrative_top.iterrows():
                    verb = "pushed prediction **up** by" if r["shap"] > 0 else "pulled prediction **down** by"
                    st.markdown(f"- **{r['label']}** ({r['value']:.1f}) {verb} **{abs(r['shap']):.1f}** units.")

                residual = chosen_row["sales"] - predicted
                if abs(residual) > 0.1 * max(abs(predicted), 1):
                    st.caption(
                        f"Residual gap: {residual:+,.0f} units (unexplained variance due to unmodeled store-specific shock)."
                    )
