import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..config import MODEL_FEATURES, FAMILY_NAMES

def render(df, model):
    st.header("Explore a Store / Product Family")

    col1, col2 = st.columns(2)
    with col1:
        store = st.selectbox("Store number", sorted(df["store_nbr"].unique()), key="explore_store")
    with col2:
        family_options = sorted(df["family"].unique())
        family = st.selectbox(
            "Product family",
            family_options,
            format_func=lambda code: FAMILY_NAMES.get(code, f"Family {code}"),
            key="explore_family",
        )

    # Optional: Update URL query params for sharing this specific view
    # In newer Streamlit versions (>=1.30.0), st.query_params is dict-like
    try:
        st.query_params["store"] = store
        st.query_params["family"] = family
    except Exception:
        pass # Fallback for older Streamlit versions

    subset = df[(df["store_nbr"] == store) & (df["family"] == family)].sort_values("date")

    if subset.empty:
        st.warning("No data available for this store / family combination in the current window.")
    else:
        X = subset[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").astype(float)
        if X.isna().any().any():
            bad_cols = X.columns[X.isna().any()].tolist()
            st.warning(f"Dropping rows with non-numeric values in columns: {bad_cols}.")
            valid_mask = ~X.isna().any(axis=1)
            X = X.loc[valid_mask]
            subset = subset.loc[valid_mask]
        if X.empty:
            st.error("No valid rows left after dropping NaN values.")
            return

        preds = model.predict(X.values)
        subset["predicted"] = preds

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=subset["date"], y=subset["sales"], name="Actual",
                                  line=dict(color="#1f77b4", width=2)))
        fig.add_trace(go.Scatter(x=subset["date"], y=preds, name="LightGBM Predicted",
                                  line=dict(color="#ff7f0e", width=2, dash="dash")))
        fig.update_layout(
            title=f"Store {store} — {FAMILY_NAMES.get(family, f'Family {family}')}",
            xaxis_title="Date", yaxis_title="Sales", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        mae = float(np.mean(np.abs(subset["sales"] - preds)))
        rmse = float(np.sqrt(np.mean((subset["sales"] - preds) ** 2)))
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (this slice)", f"{mae:,.2f}")
        m2.metric("RMSE (this slice)", f"{rmse:,.2f}")
        m3.metric("Avg. daily sales", f"{subset['sales'].mean():,.2f}")
        
        # Add CSV export
        csv_data = subset[["date", "store_nbr", "family", "sales", "predicted"]].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast Data (CSV)",
            data=csv_data,
            file_name=f"forecast_store{store}_family{family}.csv",
            mime="text/csv",
        )
