# Technical Decisions

This document outlines the key architectural and engineering decisions made during the development of the Corporación Favorita Sales Forecasting project. These decisions reflect a deliberate balance between performance, maintainability, and user experience.

## 1. Why LightGBM over XGBoost?

For the core machine learning model, LightGBM was chosen over XGBoost (and other tree-based ensembles) for several critical reasons:

*   **Training Speed on Large Datasets:** The Corporación Favorita dataset contains millions of rows. LightGBM's histogram-based algorithm and leaf-wise (best-first) tree growth strategy significantly accelerate training times compared to XGBoost's level-wise growth, without sacrificing accuracy.
*   **Memory Efficiency:** LightGBM requires substantially less memory during training, which is crucial when performing rolling-window cross-validation over a massive time-series dataset.
*   **Native Categorical Feature Support:** The dataset contains numerous high-cardinality categorical variables (e.g., store numbers, product families, city names). LightGBM handles these natively by finding the optimal split over categorical bins, eliminating the need for extensive one-hot encoding. This keeps the feature space compact and reduces overfitting risks.

## 2. Why TF-IDF for RAG instead of Vector Embeddings?

The Generative AI Analyst feature utilizes Retrieval-Augmented Generation (RAG) to ground its responses in a factual knowledge base. I opted for TF-IDF (Term Frequency-Inverse Document Frequency) over dense vector embeddings (e.g., OpenAI embeddings, SentenceTransformers):

*   **Highly Specific, Keyword-Driven Domain:** The knowledge base consists of specific historical facts (e.g., "April 16, 2016 earthquake", "WTI oil prices"). Lexical search (exact keyword matching) via TF-IDF excels at retrieving these specific facts based on user queries, often outperforming semantic search which might retrieve tangentially related but factually useless context.
*   **Zero-Dependency & Lightweight:** Implementing TF-IDF via `scikit-learn` requires no external API calls, no GPU acceleration, and adds virtually zero latency to the chat loop. It avoids the overhead of provisioning a vector database (like Pinecone or Milvus) for a relatively small and static set of domain rules.

## 3. Why Streamlit over Gradio or a Custom Frontend?

The interactive dashboard was built using Streamlit rather than Gradio or a custom React/FastAPI stack:

*   **Complex Layout Control:** Streamlit provides superior layout primitives (tabs, sidebars, multi-column layouts) which are essential for building a comprehensive, multi-view analytical dashboard. Gradio is often better suited for simpler, single-purpose ML model demonstrations.
*   **State Management:** The dashboard requires complex state management, particularly for the Generative AI chat interface and dynamic filtering (e.g., updating URL query parameters based on user selections). Streamlit's `st.session_state` and native query parameter handling make this straightforward.
*   **Rapid Iteration for Data Apps:** Streamlit's execution model (rerunning the script top-to-bottom on interaction) pairs perfectly with `st.cache_data` and `st.cache_resource`, allowing for rapid iteration on the data visualization and modeling logic without managing complex frontend/backend API contracts.

## 4. Why Aggregate LightGBM for Comparison?


Classical statistical models (ARIMA, SARIMA, Prophet) are typically trained on aggregated, top-level time series (e.g., National daily sales) because fitting them at the granular store-item level is computationally prohibitive and often excessively noisy. LightGBM, however, thrives on granular row-level data.

To ensure a fair, apples-to-apples evaluation, LightGBM's row-level predictions must be aggregated to the National level before computing error metrics (MAE, RMSE, MAPE) for head-to-head comparison against the classical baselines. Comparing row-level LightGBM errors directly to National-level ARIMA errors constitutes a granularity mismatch that invalidates the evaluation.
