import streamlit as st
from ..data_loader import load_comparison
from ..config import COMPARISON_PATH, FEATURE_IMPORTANCE_IMG, PROPHET_IMG

def render():
    st.header("Model Comparison So Far")
    try:
        comp = load_comparison()
        st.dataframe(comp, use_container_width=True)

        if "MAE" in comp.columns and comp["MAE"].max() / max(comp["MAE"].min(), 1e-9) > 100:
            st.caption(
                "⚠️ MAE values span more than two orders of magnitude across rows — "
                "that's usually a sign some rows were fit on a different granularity "
                "(e.g. national aggregate vs. single store-family). Worth double-checking "
                "before treating this table as a like-for-like comparison."
            )
    except FileNotFoundError:
        st.info(f"`{COMPARISON_PATH}` not found in the repo.")
    except Exception as e:
        st.warning(f"Could not load model comparison table: {e}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("LightGBM Feature Importance")
        try:
            st.image(FEATURE_IMPORTANCE_IMG, use_container_width=True)
        except Exception:
            st.info("Feature importance image not found.")
    with c2:
        st.subheader("Prophet Components")
        try:
            st.image(PROPHET_IMG, use_container_width=True)
        except Exception:
            st.info("Prophet components image not found.")
