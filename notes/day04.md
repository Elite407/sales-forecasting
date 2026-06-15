# Day 04: Deep Exploratory Data Analysis (EDA)

**Topic:** Time Series Decomposition, ACF, and PACF  
**Objective:** Transition from treating data as a tabular spreadsheet to analyzing it mathematically as a continuous signal. Extract the hidden structural rules that dictate the ARIMA model parameters.

---

## 1. Time Series Decomposition

**Concept:** Any time series can be mathematically separated into underlying components. This allows us to isolate specific behaviors (like weekend spikes) from long-term trajectories (like yearly growth).

### The Four Components
1. **Trend ($T_t$):** The long-term progression of the series (e.g., sales growing year-over-year).
2. **Seasonality ($S_t$):** Repeating, predictable patterns at fixed intervals (e.g., weekly grocery cycles).
3. **Cyclical ($C_t$):** Repeating patterns without a fixed period (e.g., economic recessions). Often absorbed into the trend or residual for short-term retail forecasting.
4. **Residual/Noise ($R_t$):** The random variance left over after Trend and Seasonality are removed. A perfect decomposition leaves residuals that look like pure white noise.

### Additive vs. Multiplicative Models
The choice between these two dictates how the algorithm assumes the components interact.

* **Additive Model:** $Y(t) = T(t) + S(t) + R(t)$
  * **Assumption:** The amplitude of the seasonal fluctuation is constant, regardless of the trend level.
  * **Example:** A store always sells exactly 500 more units on Saturdays than Mondays, whether the baseline is 1,000 or 10,000.
* **Multiplicative Model:** $Y(t) = T(t) \times S(t) \times R(t)$
  * **Assumption:** The amplitude of the seasonal fluctuation scales proportionally with the trend.
  * **Example:** A store always sells 20% more units on Saturdays. If the baseline is 1,000, the spike is 200. If the baseline grows to 10,000, the spike grows to 2,000. 
  * *Note: Multiplicative is usually the correct choice for growing retail businesses*.

### Mathematical Proof: How Classical Decomposition Extracts Components
Decomposition is an algorithmic procedure. Here is the exact math used to extract the additive components:

**Step 1: Extract the Trend ($T_t$) via Moving Average**
To find the trend, we smooth out the seasonality. For a weekly period ($m=7$), we compute a centered moving average.
$$\hat{T}_t = \frac{1}{m} \sum_{j=-k}^k Y_{t+j}$$
*(Where $k = 3$, meaning we average the current day, 3 days prior, and 3 days ahead).*

**Step 2: Detrend the Series**
We subtract the trend from the original data, leaving behind only seasonality and noise.
$$D_t = Y_t - \hat{T}_t$$

**Step 3: Isolate Seasonality ($S_t$)**
We group the detrended values by their specific season (e.g., all Mondays) and average them to find the pure seasonal effect.
$$\hat{S}_{Monday} = \frac{1}{N_{Mondays}} \sum D_{Monday}$$
We then mean-center these values so the net seasonal effect across a full period sums to zero.

**Step 4: Extract the Residual ($R_t$)**
Subtract the computed trend and seasonality from the original observation.
$$\hat{R}_t = Y_t - \hat{T}_t - \hat{S}_t$$

---

## 2. Autocorrelation Function (ACF)

**Concept:** Autocorrelation measures the linear relationship between a time series and a delayed (lagged) version of itself. 

**The Rule:** The ACF plot tells us the **q** parameter (Moving Average order) for our ARIMA model. It shows how long a "shock" echoes through the timeline.

### Mathematical Derivation of ACF
The ACF is derived directly from the Pearson Correlation Coefficient:
$$r = \frac{\sum (X_i - \bar{X})(Z_i - \bar{Z})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Z_i - \bar{Z})^2}}$$

If we substitute our time series $y$ at time $t$ for $X$, and $y$ at time $t-k$ (the lag) for $Z$:
$$\rho_k = \frac{\sum_{t=k+1}^N (y_t - \bar{y}_t)(y_{t-k} - \bar{y}_{t-k})}{\sqrt{\sum_{t=k+1}^N (y_t - \bar{y}_t)^2 \sum_{t=k+1}^N (y_{t-k} - \bar{y}_{t-k})^2}}$$

**Applying Stationarity:**
Because we assume the series is stationary, the mean is constant ($\bar{y}_t \approx \bar{y}_{t-k} \approx \bar{y}$) and the variance is constant over time. The denominator simplifies to the total variance of the series, yielding the final ACF formula:
$$\rho_k = \frac{\sum_{t=k+1}^N (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^N (y_t - \bar{y})^2}$$

### Reading the ACF Plot
* The shaded blue region is the 95% confidence interval.
* Bars extending beyond the blue region indicate statistically significant correlation at that specific lag.
* **Warning:** If the ACF decays very slowly, the data is non-stationary (it has a trend). You must difference the data before reading the ACF.

---

## 3. Partial Autocorrelation Function (PACF)

**Concept:** The PACF isolates the *direct* correlation between $y_t$ and $y_{t-k}$ by mathematically stripping away the influence of all intermediate lags (1 through $k-1$).

**The Rule:** The PACF plot tells us the **p** parameter (AutoRegressive order) for our ARIMA model.

### The "Grandfather" Intuition
* **ACF (Indirect included):** A grandfather's fame correlates with his grandson's fame. But is this because the grandfather directly influenced the grandson, or because the grandfather made the father famous, who then made the son famous? The ACF cannot tell the difference.
* **PACF (Direct only):** The PACF controls for the father's fame. It measures *only* the direct influence the grandfather had on the grandson. If the PACF at lag 2 drops to zero, it means lag 2 has no direct predictive power once lag 1 is accounted for.

---

## 4. Business Application to Corporacion Favorita

When analyzing the Ecuador sales dataset:
1. **Weekly Seasonality:** You will see significant ACF spikes at lags 7, 14, and 21. This confirms that day-of-the-week is the strongest predictor of sales.
2. **Earthquake Anomaly:** When viewing the $R_t$ (Residuals) plot from your decomposition, look at April 2016. The massive spike there is not bad data; it is the mathematical footprint of the magnitude 7.8 Pedernales earthquake.
3. **Payday Effects:** Spikes on the 15th and last day of the month reflect bi-weekly salary disbursements in the region.

---

## 5. Weekly Review Cheat Sheet (Interview Prep)

Review these points to ensure the concepts remain sharp:

* **Why do we decompose?** To isolate the signal (Trend/Seasonality) from the noise (Residuals).
* **Additive vs Multiplicative?** Additive = constant variance. Multiplicative = variance scales with trend.
* **ACF vs PACF?** ACF includes indirect echoes; PACF isolates direct relationships.
* **ARIMA Mapping:** PACF cutoff determines $p$ (AutoRegressive terms). ACF cutoff determines $q$ (Moving Average terms).
* **The Cardinal Rule:** You cannot accurately read $p$ and $q$ from a non-stationary series. You must difference the data first.
