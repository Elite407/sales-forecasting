import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..config import MODEL_FEATURES, FAMILY_NAMES

def render(df, model):
    st.header("Explore a Store / Product Family")
    st.caption("Inspect granular actual vs. predicted sales series across all 1,782 store-family time series.")

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

        # Date range filter slider syncing subset and dynamic metric calculation
        min_date = subset["date"].min().date()
        max_date = subset["date"].max().date()

        if min_date < max_date:
            date_range = st.slider(
                "📅 Filter Forecast Date Window",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="forecast_date_slider",
            )
            filtered_subset = subset[
                (subset["date"].dt.date >= date_range[0]) & 
                (subset["date"].dt.date <= date_range[1])
            ]
        else:
            filtered_subset = subset

        if filtered_subset.empty:
            st.info("No data in the selected date range.")
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_subset["date"], y=filtered_subset["sales"], name="Actual Sales",
            line=dict(color="#2563EB", width=2.5)
        ))
        fig.add_trace(go.Scatter(
            x=filtered_subset["date"], y=filtered_subset["predicted"], name="LightGBM Predicted",
            line=dict(color="#F97316", width=2.5, dash="dash")
        ))
        fig.update_layout(
            title=f"Store {store} — {FAMILY_NAMES.get(family, f'Family {family}')} ({len(filtered_subset)} days)",
            xaxis_title="Date", yaxis_title="Daily Sales", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        mae = float(np.mean(np.abs(filtered_subset["sales"] - filtered_subset["predicted"])))
        rmse = float(np.sqrt(np.mean((filtered_subset["sales"] - filtered_subset["predicted"]) ** 2)))
        avg_sales = float(filtered_subset["sales"].mean())

        m1, m2, m3, m4 = st.columns([1, 1, 1, 1.2])
        with m1:
            with st.container(border=True):
                st.metric("Slice MAE", f"{mae:,.2f}", help="Mean Absolute Error across the filtered date range")
        with m2:
            with st.container(border=True):
                st.metric("Slice RMSE", f"{rmse:,.2f}", help="Root Mean Squared Error across the filtered date range")
        with m3:
            with st.container(border=True):
                st.metric("Avg Daily Sales", f"{avg_sales:,.2f}", help="Mean daily actual sales")
        with m4:
            with st.container(border=True):
                st.markdown("**Export Slice**")
                csv_data = filtered_subset[["date", "store_nbr", "family", "sales", "predicted"]].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"forecast_store{store}_family{family}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
