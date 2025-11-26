# STA4003 - Time Series Group 8 Project

This project involves various time series analysis methods applied to Bitcoin active wallet data. It includes data preprocessing, feature analysis, factor filtering, and model forecasting using ARIMA, SARIMA, and ARMA models.

## Table of Contents
1. [Phase 1 - Decomposition, Feature Analysis & Factor Filtering](#phase-1-decomposition-feature-analysis--factor-filtering)
2. [Phase 2 - Model](#phase-2---model)
   - [AR(2) Forecasting for BTC Active Wallets](#ar2-forecasting-for-btc-active-wallets)
   - [SARIMA Forecasting](#sarima-forecasting)
   - [ARIMA](#ARIMA)

## Phase 1 - Decomposition, Feature Analysis & Factor Filtering
In this phase, we perform an analysis of Bitcoin price returns and factor filtering using decomposition methods such as STL, ACF, and PACF.

### Steps:
1. Run `Return_analysis.py` to analyze return data for BTC price.
2. Run `Feature_analysis_and_factor_filtering.py` for factor filtering and plot graphs of STL decomposition, ACF, and PACF.

## Phase 2 - Model

##Benchmark Models
==============================================

This notebook implements and evaluates three basic time series forecasting methods:
1. Mean Method
2. Naive Method
3. Seasonal Naive Method
4. Drift Method
5. 
Data:
- Bitcoin active wallet data (3-6 months activity)
- Daily aggregated from 10-minute data
- Period: 2021-01-01 onwards
  
Models Evaluation:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

Required Libraries:
pandas
numpy
matplotlib
sklearn

Usage:
Run cells sequentially to see the performance comparison of different benchmark models.

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

### ARIMA
This project implements an ARIMA-based forecasting pipeline using a daily time series.

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
   - MASE
