import json
import os
import pickle
import pandas as pd
import streamlit as st
from .config import (
    MODEL_PATH, DATA_PATH, COMPARISON_PATH, MODEL_FEATURES
)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)

@st.cache_data
def load_text_artifact(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return f.read()

@st.cache_data
def load_json_artifact(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data
def load_csv_artifact(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["date"] if "date" in pd.read_csv(path, nrows=0).columns else None)

def initialize_data_and_model():
    """Wrapper to load core data with error handling for the main app."""
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model from `{MODEL_PATH}`: {e}")
        st.stop()

    try:
        df = load_data()
    except Exception as e:
        st.error(f"Could not load data from `{DATA_PATH}`: {e}")
        st.stop()

    missing_cols = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing_cols:
        st.error(f"feature_store.csv is missing columns the model needs: {missing_cols}")
        st.stop()
        
    return model, df
