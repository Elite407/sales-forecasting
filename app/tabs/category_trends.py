import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ..config import FAMILY_NAMES

def render(df, generate_quick_insight_gemini_fn):
    st.header("Sales by Category, Summed Across All Stores")
    st.caption(
        "Every store is aggregated together here — the comparison is purely "
        "across product families over time."
    )

    family_totals = (
        df.groupby("family")["sales"].sum().sort_values(ascending=False)
    )
    default_families = family_totals.head(5).index.tolist()

    selected_families = st.multiselect(
        "Product families to compare",
        options=sorted(df["family"].unique()),
        default=default_families,
        format_func=lambda code: FAMILY_NAMES.get(code, f"Family {code}"),
    )
    show_total = st.checkbox("Also show combined total of selected families", value=False)

    if not selected_families:
        st.info("Pick at least one product family above.")
    else:
        cat_df = (
            df[df["family"].isin(selected_families)]
            .groupby(["date", "family"])["sales"]
            .sum()
            .reset_index()
        )
        cat_df["family_name"] = cat_df["family"].map(lambda c: FAMILY_NAMES.get(c, f"Family {c}"))

        fig = px.line(
            cat_df.sort_values("date"),
            x="date", y="sales", color="family_name", markers=True,
            labels={"sales": "Total sales (all stores)", "date": "Date", "family_name": "Family"},
        )

        if show_total:
            total_series = cat_df.groupby("date")["sales"].sum().reset_index()
            fig.add_trace(go.Scatter(
                x=total_series["date"], y=total_series["sales"],
                name="Selected — Combined Total",
                line=dict(color="black", width=3, dash="dot"),
            ))

        fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Top families by total sales in this data window: "
            + ", ".join(f"{FAMILY_NAMES.get(c, c)}" for c in family_totals.head(8).index)
        )
        
        st.subheader("✨ Quick Insight")
        total_sales_selected = cat_df["sales"].sum()
        family_group = cat_df.groupby("family_name")["sales"].sum()
        leading_fam = family_group.idxmax() if not family_group.empty else "N/A"
        
        trend_series = cat_df.groupby("date")["sales"].sum().sort_index()
        if len(trend_series) > 1:
            mid = len(trend_series) // 2
            first_half = trend_series.iloc[:mid].mean()
            second_half = trend_series.iloc[mid:].mean()
            if second_half > first_half * 1.05:
                trend_dir = "Upward"
            elif second_half < first_half * 0.95:
                trend_dir = "Downward"
            else:
                trend_dir = "Stable"
        else:
            trend_dir = "Insufficient data"
            
        d_min = cat_df["date"].min().strftime('%Y-%m-%d') if not cat_df.empty else "N/A"
        d_max = cat_df["date"].max().strftime('%Y-%m-%d') if not cat_df.empty else "N/A"
        fams_str = ", ".join([FAMILY_NAMES.get(c, str(c)) for c in selected_families])
        
        with st.spinner("Generating insight..."):
            insight = generate_quick_insight_gemini_fn(
                fams_str, d_min, d_max,
                total_sales_selected, leading_fam, trend_dir
            )
        st.info(insight)
