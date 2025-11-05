# STA4003 - Time Series Group 8 Project

This project involves various time series analysis methods applied to Bitcoin active wallet data. It includes data preprocessing, feature analysis, factor filtering, and model forecasting using ARIMA, SARIMA, and ARMA models.

## Table of Contents
1. [Phase 1 - Decomposition, Feature Analysis & Factor Filtering](#phase-1-decomposition-feature-analysis--factor-filtering)
2. [Phase 2 - Model](#phase-2---model)
   - [AR(2) Forecasting for BTC Active Wallets](#ar2-forecasting-for-btc-active-wallets)
   - [SARIMA Forecasting](#sarima-forecasting)
   - [Bitcoin On-Chain Factor Forecasting with ARIMA](#bitcoin-on-chain-factor-forecasting-with-arima)
   - [ARMA Sensitivity Analysis](#arma-sensitivity-analysis)
3. [Dependencies](#dependencies)
4. [Usage Instructions](#usage-instructions)

## Phase 1 - Decomposition, Feature Analysis & Factor Filtering
In this phase, we perform an analysis of Bitcoin price returns and factor filtering using decomposition methods such as STL, ACF, and PACF.

### Steps:
1. Run `Return_analysis.py` to analyze return data for BTC price.
2. Run `Feature_analysis_and_factor_filtering.py` for factor filtering and plot graphs of STL decomposition, ACF, and PACF.

## Phase 2 - Model

### AR(2) Forecasting for BTC Active Wallets
This section applies an AutoRegressive (AR) model of order 2 (AR(2)) to forecast daily BTC active wallet counts from high-frequency data.

#### Features:
- Aggregates 10-minute BTC active wallet data into daily averages.
- Splits the data into training (80%) and testing (20%) sets.
- Fits an AR(2) model on the training data.
- Performs one-step-ahead rolling forecasts.
- Computes forecast metrics: RMSE, MAE, MAPE, MASE.
- Conducts a Ljung-Box test on residuals for autocorrelation.
- Visualizes training data, actual test data, and AR(2) forecasts.

#### Example Metrics:
| Metric  | Value   |
| ------- | ------- |
| RMSE    | 968,754 |
| MAE     | 746,357 |
| MAPE    | 30.14%  |
| MASE    | 49.46%  |

### SARIMA Forecasting
This section involves preprocessing and modeling using SARIMA (Seasonal ARIMA), which accounts for seasonality in the time series data.

#### 1. SARIMA Data Preprocessing (`SARIMA_data_preprocess.py`):
- **Preprocessing & Resampling**: Filters and resamples the data to daily frequency and fills missing values.
- **Seasonal Detection**: Uses ACF to identify seasonal patterns.
- **Stationarity Check**: Performs ADF test to recommend seasonal differencing `D`.
- **Visualization**: Plots original series, ACF, STL decomposition, and seasonal strength.

#### 2. SARIMA Model Fitting (`SARIMA_model_fit.py`):
- **Preprocessing**: Resamples the data to daily frequency and handles missing values.
- **Parameter Selection**: Performs coarse and fine grid search for ARIMA parameters.
- **Rolling Forecast**: Generates step-by-step forecasts for the test set.
- **Metrics & Diagnostics**: Computes RMSE, MAE, MAPE, MASE, R², AIC, BIC, Ljung-Box, and normality tests.
- **Visualization**: Saves plots for residual diagnostics, model fit, and forecasts.

### Bitcoin On-Chain Factor Forecasting with ARIMA
This part forecasts Bitcoin on-chain factors using the ARIMA model with daily aggregated data.

#### Features:
- **Data**: Hourly raw data resampled to daily frequency (2023 - Mid 2025).
- **Model Selection**: Grid search for ARIMA(p, 0, q) with p, q ∈ {0, 1, 2, 3, 4}, evaluated by AIC, BIC, RMSE, MAE, MAPE, MASE.
- **Best Model**: ARIMA(4, 0, 1) — no differencing needed.

#### Key Finding:
Differencing (e.g., ARIMA(4, 1, 1)) leads to flat, uninformative forecasts, confirming that d=0 is optimal.

### ARMA Sensitivity Analysis
This section explores the sensitivity of the ARMA model's performance based on hyperparameter tuning.

#### Steps:
1. **Data Preprocessing**: Lagged variables are created, and missing values are handled.
2. **Visualization**: Interactive line plots visualize the relationship between active supply factors and their lagged values.
3. **Modeling**: ARMA models are fitted, and their performance is evaluated using AIC and BIC.
4. **Hyperparameter Sensitivity**: Systematically tests different values for AR (`p`) and MA (`q`) orders and evaluates model performance.
5. **Residual Analysis**: Checks for patterns, autocorrelation, and heteroscedasticity in the residuals.

## Dependencies
Ensure you have the following Python libraries installed:
- Python 3.8+
- pandas
- numpy
- matplotlib
- statsmodels
- scikit-learn
- plotly (for visualizations)

Install dependencies via pip:
```bash
pip install pandas numpy matplotlib statsmodels scikit-learn plotly
