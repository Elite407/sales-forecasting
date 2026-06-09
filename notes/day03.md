# Day 03: Data Preprocessing for Time Series 

**Project:** Retail Sales Forecasting & Inventory Prediction (Corporación Favorita)  
**Objective:** Clean and format raw data for machine learning models while strictly preventing temporal data leakage. 

In time-series forecasting, tabular data rules do not apply. Careless preprocessing is where 90% of forecasting pipelines fail before training even begins.

---

## 1. Missing Value Imputation (Time-Series Specific)

In standard machine learning, dropping a row with missing values or filling it with the dataset's average is common. **In time series, dropping a row breaks the temporal sequence**, destroying lag features and recurrent neural network (RNN) inputs. Filling with a global average destroys local temporal context.

**Strategies Used:**
* **Forward-Fill (ffill):** Carries the last known value forward.
    * *Example:* Global oil markets are closed on weekends, resulting in `NaN` for Saturday and Sunday. We forward-fill Friday's oil price because that is the last known economic state.
* **Linear Interpolation:** Draws a straight line between two known points. Best for variables that change continuously rather than in step-changes.
* **Business Logic Imputation:** A zero in sales might mean "out of stock" or "store closed," not necessarily a lack of demand. Always investigate the *why* before imputing.

## 2. Categorical Encoding

Machine learning models require numerical inputs, meaning categorical text (like "GROCERY I" or "National Holiday") must be encoded.

* **Label Encoding:** Assigns a unique integer to each category (e.g., Produce = 0, Grocery = 1).
    * *Best for:* Tree-based models (XGBoost, LightGBM, Random Forest) which handle these splits natively.
    * *Drawback:* Implies a false mathematical hierarchy (1 is not "greater" than 0 in this context).
* **One-Hot Encoding (OHE):** Creates a binary column for every category.
    * *Drawback:* Causes the "curse of dimensionality" in retail datasets with thousands of product/store combinations, slowing down neural networks.
* **Entity Embeddings (Future Phase):** Maps categories into dense vector spaces. This allows deep learning models to learn relationships (e.g., "Dairy" and "Eggs" behave similarly).

## 3. Normalization and Scaling

Our target variable (`sales`) ranges from 0 to 50,000+, while exogenous variables like `oil_price` range from 40 to 100. 

* **The Problem:** Feeding unscaled numbers into a neural network causes massive gradient updates for large features and tiny updates for small features, leading to training instability and failure to converge.
* **The Solution:** We use **Min-Max Scaling** to squash all numerical values into a standard range (usually 0 to 1). 

## 4. The Cardinal Sin: Data Leakage

Data leakage occurs when information from the future is accidentally used to train the model. It results in spectacular test scores but catastrophic real-world failure.

**How Leakage Happens in Scaling:**
The most common mistake is applying `MinMaxScaler` to the *entire* dataset before making the train/test split. If the global maximum sales value occurs in the test set, your training data is being scaled using information from the future.

**Best Practice Implementation:**
1.  **Split** data chronologically into `train` and `test` sets. **Never randomly shuffle time-series data.**
2.  **Fit** the scaler ONLY on the `train` data (learning the min/max of the past).
3.  **Transform** both `train` and `test` data using that fitted scaler.
4.  **Save** the scaler object (e.g., using `pickle`) to inverse-transform your predictions later.

---
*Notes adapted from the 38-Day AI/ML Development Track.*
