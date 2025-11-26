# STA4003 - Time Series Group 8 Project

This project involves various time series analysis methods applied to Bitcoin active wallet data. It includes data preprocessing, feature analysis, factor filtering, and model forecasting using simple benchmark, SARIMA, ARMA,LSTM models.

## Table of Contents
1. [Phase 1 - Decomposition, Feature Analysis & Factor Filtering](#phase-1-decomposition-feature-analysis--factor-filtering)
2. [Phase 2 - Model](#phase-2---model)
   - [Benchmark Models](#benchmark-models)
   - [SARIMA Seasonal Analysis](#sarima-seasonal-analysis)
   - [ARIMA](#ARIMA)
   - [LSTM](#LSTM)

## Phase 1 - Decomposition, Feature Analysis & Factor Filtering
In this phase, we perform an analysis of Bitcoin price returns and factor filtering using decomposition methods such as STL, ACF, and PACF.

### Steps:
1. Run `Return_analysis.py` to analyze return data for BTC price.
2. Run `Feature_analysis_and_factor_filtering.py` for factor filtering and plot graphs of STL decomposition, ACF, and PACF.

## Phase 2 - Model

# Benchmark Models
==============================================

This notebook implements and evaluates three basic time series forecasting methods:
1. Mean Method
2. Naive Method
3. Seasonal Naive Method
4. Drift Method

## Data:
- Bitcoin active wallet data (3-6 months activity)
- Daily aggregated from 10-minute data
- Period: 2021-01-01 onwards

## Models Evaluation:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

## Required Libraries:
- pandas
- numpy
- matplotlib
- sklearn

## Usage:
Run cells sequentially to see the performance comparison of different benchmark models.

# SARIMA Seasonal Analysis

## Project Overview
This project uses the SARIMA model to perform time series analysis and seasonal pattern identification on BTC hourly relative profit data.

## Main Features
- Data preprocessing and resampling
- Time series visualization analysis
- Stationarity tests (ADF and KPSS tests)
- Seasonal pattern identification
- SARIMA model fitting

### 1. Visualization Analysis
- Original time series plot
- Data distribution histogram + KDE
- Autocorrelation Function (ACF) plot
- Partial Autocorrelation Function (PACF) plot

### 2. Stationarity Tests
- ADF test (Null hypothesis: Non-stationary)
- KPSS test (Null hypothesis: Stationary)
- Comprehensive stationarity judgment

### 3. Model Building
Using `pmdarima` and `statsmodels` libraries:
- Automatic SARIMA parameter selection
- Seasonal pattern identification
- Model training and validation


# ARIMA
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
  
LSTM Time-Series Forecast 
============================================

1. Files
- Notebook: `LSTM.ipynb`，`LSTM_stop.ipynb`,`LSTM_Hyper.ipynb`
- Data: `./data/BTC factors/addresses/BTC_1h_profit_relative.csv`

Dataset → Supervised
- Sliding window (window_size = k): use k previous days to predict next day.
- LSTM input shape: (batch, seq_len, 1).

2. Model
- Single-layer LSTM + linear output.
- Hidden units: 64.
- Loss: `SmoothL1Loss`.
- Optimizer: Adam (lr=0.01).
- Training: full-batch example (epochs in notebook).

3. Inference & Evaluation
- Inverse-transform predictions to original scale.
- Metrics: RMSE, MAE, MAPE, MASE.
- Visualization: Train True / Train Pred / Test True / Test Pred plotted on same axis.

4. Usage
   1. Execute the `LSTM_Hyper.ipynb` to find the best hyperparameters for the LSTM.
   2. Then open `LSTM.ipynb` or run the `LSTM_stop.ipynb` which we added the early stopping criteria to ensure the model not overfitting.
   3. Ensure data path is available.
   4. Run cells sequentially.

5. Notes
- Scaler is fitted only on training data to avoid data leakage.
- For faster experiments reduce epochs or use mini-batches.

