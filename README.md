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


This project implements an ARIMA-based forecasting pipeline using a daily time series.

## 1. Workflow

1. **Train–Test Split**  
   The dataset is split 80/20 into an initial training set and a test set.

2. **Model Selection (Grid Search)**  
   Since ARIMA requires stationarity, differencing is fixed at **d = 1**.  
   We search over:
   - p = 0–5  
   - q = 0–5  
   The combination with the lowest AIC on the initial training set is selected as the final model order.

3. **Rolling (Expanding) Forecasting**  
   Using the selected (p,1,q), we perform one-step-ahead forecasting:
   - Fit the model on the current training window  
   - Predict the next data point  
   - Append the true value into the training set  
   - Repeat until all test observations are predicted  

4. **Evaluation Metrics**  
   We compute:
   - RMSE  
   - MAE  
   - MAPE  
   - MASE (using naïve forecast on the training set)
