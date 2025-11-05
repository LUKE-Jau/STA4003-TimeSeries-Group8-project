import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import pmdarima as pmd
import itertools  # For grid search

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, normal_ad
from sklearn.metrics import mean_squared_error, mean_absolute_error
warnings.filterwarnings("ignore")

def compute_metrics(y_true, y_pred, y_train):

    # 1. RMSE, MAE, R2
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    # 2. MAPE (Mean Absolute Percentage Error)
    epsilon = 1e-8 
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    # 3. MASE (Mean Absolute Scaled Error)
    n_train = len(y_train)
    if n_train < 2:
        mase = np.nan
    else:
        scale_factor = np.mean(np.abs(y_train.iloc[1:].values - y_train.iloc[:-1].values))
        
        if scale_factor == 0:
            mase = np.nan 
        else:
            mase = mae / scale_factor

    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MASE': mase, 'R2': r2}

def diagnostic_tests(model):
    """Perform residual diagnostic tests (Ljung-Box and Anderson-Darling normality)"""
    resid = model.resid
    lb = acorr_ljungbox(resid, lags=[12], return_df=True)
    ad_stat, ad_p = normal_ad(resid)
    return {
        'LjungBox_p': lb['lb_pvalue'].iloc[0],
        'Normal_p': ad_p
    }

def plot_residual_diagnostics(results, period_name):
    os.makedirs("./Seasonal_Analysis/SARIMA_Results/Diagnostics", exist_ok=True)
    
    fig = results.plot_diagnostics(figsize=(15, 10))
    fig.suptitle(f'SARIMA Residual Diagnostics - {period_name}', fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.savefig(f"./Seasonal_Analysis/SARIMA_Results/Diagnostics/SARIMA_Residual_Diagnostics_{period_name}.png", bbox_inches='tight')
    plt.close()

def plot_forecast(y_train, y_test, y_pred, y_fit, period_name):

    plt.figure(figsize=(14, 6))
    
    plt.plot(y_train.index, y_train, label='Train (True)', color='C0')
    plt.plot(y_fit.index, y_fit, label='Train (Fitted)', color='C1', linestyle=':')
    plt.plot(y_test.index, y_test, label='Test (True)', color='C2')
    plt.plot(y_test.index, y_pred, label='Test (Forecast)', color='C3', linestyle='--')
    
    plt.title(f"SARIMA Model Fit and Forecast - {period_name}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.6, linestyle='--')
    
    os.makedirs("./Seasonal_Analysis/SARIMA_Results", exist_ok=True)
    plt.savefig(f"./Seasonal_Analysis/SARIMA_Results/SARIMA_Fit_Forecast_{period_name}.png", bbox_inches='tight')
    plt.close()

def plot_test_forecast(y_test, y_pred, period_name):

    plt.figure(figsize=(12, 5))
    
    plt.plot(y_test.index, y_test, label='Test (True)', color='C2')
    plt.plot(y_test.index, y_pred, label='Test (Forecast)', color='C3', linestyle='--')
    
    plt.title(f"SARIMA Test Forecast Only - {period_name}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.6, linestyle='--')
    
    os.makedirs("./Seasonal_Analysis/SARIMA_Results", exist_ok=True)
    plt.savefig(f"./Seasonal_Analysis/SARIMA_Results/SARIMA_Test_Forecast_Only_{period_name}.png", bbox_inches='tight')
    plt.close()

def sarima_grid_search(train, start_p, start_q, start_P, start_Q, seasonal_period, d, D, search_range=1):
    
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    
    p_range = range(max(0, start_p - search_range), start_p + search_range + 1)
    q_range = range(max(0, start_q - search_range), start_q + search_range + 1)
    P_range = range(max(0, start_P - search_range), start_P + search_range + 1)
    Q_range = range(max(0, start_Q - search_range), start_Q + search_range + 1)
    
    param_combinations = itertools.product(p_range, q_range, P_range, Q_range)

    print(f"  ↳ Starting fine search: grid search range (p,q)±{search_range}, (P,Q)±{search_range}...")
    
    for p, q, P, Q in param_combinations:
        try:
            order = (p, d, q)
            seasonal_order = (P, D, Q, seasonal_period)
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order, 
                            enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False, maxiter=200) 
            
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_seasonal_order = seasonal_order
                
        except Exception:
            continue
            
    return best_order, best_seasonal_order

def run_sarima(data, seasonal_period, D, period_name):
    
    split = int(len(data)*0.8)
    train, test = data.iloc[:split], data.iloc[split:]
    y_train = train['y'] 
    y_test = test['y']
    d = 0  

    print(" 	↳ 1. Coarse search (auto_arima): quickly searching best parameters...")
    auto_model = pmd.auto_arima(
        y_train, start_p=0, start_q=0, max_p=5, max_q=5,
        d=d, D=D, seasonal=True, m=seasonal_period,
        start_P=0, start_Q=0, max_P=3, max_Q=3,
        stepwise=True, trace=False, suppress_warnings=True,
        error_action='ignore', maxiter=50
    )
    start_p, start_d, start_q = auto_model.order
    start_P, start_D, start_Q, _ = auto_model.seasonal_order
    
    print(f" 	↳ Auto-ARIMA recommended: order=({start_p},{start_d},{start_q}), seasonal=({start_P},{start_D},{start_Q},{seasonal_period})")
    
    best_order, best_seasonal_order = sarima_grid_search(
        y_train, start_p, start_q, start_P, start_Q, seasonal_period, d, D, search_range=1
    )
    if best_order is None or best_seasonal_order is None:
        best_order = (start_p, start_d, start_q)
        best_seasonal_order = (start_P, start_D, start_Q, seasonal_period)
        print(" 	↳ Fine search failed or no improvement, using Auto-ARIMA recommended parameters.")
    else:
        print(f" 	↳ Fine search optimization result: order={best_order}, seasonal={best_seasonal_order}")

    model = SARIMAX(y_train, order=best_order, seasonal_order=best_seasonal_order)
    results = model.fit(disp=False, maxiter=500) 
    y_fit = results.fittedvalues
    
    print(f" ↳ Initial SARIMA fit completed, starting rolling forecast ({len(y_test)} steps)...")

    y_pred_list = []
    current_train = y_train.copy()

    for i in range(len(y_test)):
        try:
            forecast = results.get_forecast(steps=i+1)
            y_pred_step = forecast.predicted_mean.iloc[-1]
            y_pred_list.append(y_pred_step)

        except Exception as e:
            print(f" ↳ Warning: forecast step {i} failed ({e}), using previous prediction.")
            y_pred_step = y_pred_list[-1] if len(y_pred_list) > 0 else current_train.iloc[-1]
            y_pred_list.append(y_pred_step)

    print(" ↳ Rolling forecast completed.")

    y_pred = pd.Series(y_pred_list, index=y_test.index)

    metrics = compute_metrics(y_test, y_pred, y_train) 
    diag = diagnostic_tests(results) 
    metrics.update(diag)
    
    metrics['Period'] = period_name
    metrics['Seasonal_m'] = seasonal_period
    metrics['d'] = d
    metrics['D'] = D
    metrics['p,d,q'] = f"{best_order}"
    metrics['P,D,Q'] = f"({best_seasonal_order[0]},{best_seasonal_order[1]},{best_seasonal_order[2]})"
    metrics['AIC'] = results.aic
    metrics['BIC'] = results.bic

    plot_residual_diagnostics(results, period_name)
    plot_forecast(y_train, y_test, y_pred, y_fit, period_name) 
    plot_test_forecast(y_test, y_pred, period_name) 

    return metrics

def main():

    FILE_PATH = './BTC_10m_active_3m_6m.csv'
    PERIOD_FILE = './Seasonal_Analysis/seasonal_periods_detection.csv'
    TARGET_COL = 'active_3m_6m'

    try:
        df = pd.read_csv(FILE_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        cutoff_date = pd.to_datetime('2023-01-01')
        df = df[df['datetime'] >= cutoff_date].copy()
        df.set_index('datetime', inplace=True)
        data = df[TARGET_COL].resample('1D').mean().dropna()
        data = data.to_frame(name='y') 

        period_df = pd.read_csv(PERIOD_FILE).head(3)
        
    except FileNotFoundError as e:
        print(f"Error: File not found {e.filename}. Please check the file path.")
        return
    except KeyError:
        print(f"Error: Column '{TARGET_COL}' or 'datetime' not found in data file.")
        return

    print("=" * 60)
    print("📊 Data Resampling and Time Split Results")
    print("=" * 60)
    print(f"Time Range: {data.index.min()} → {data.index.max()}")
    print(f"Number of data points: {len(data)}")
    print("=" * 60)

    print("\n📊 Auto-read top 3 recommended periods:")
    print(period_df[['period_name', 'period_value', 'D']])

    all_results = []

    param_map = {}
    
    for _, row in period_df.iterrows():
        pname = row['period_name']
        pval = int(row['period_value'])
        D = int(row['D'])
        D = max(0, min(1, D))
        
        print(f"\n====== Running SARIMA for {pname} ({pval} days), D={D} ======")

        if pname in param_map:
            params = param_map[pname]
            metrics_list = run_sarima(
                data, 
                seasonal_period=pval, 
                D=D, 
                period_name=pname,
                order=params['order'], 
                seasonal_order=params['seasonal_order']
            )
        else:
            metrics_list = run_sarima(
                data, 
                seasonal_period=pval, 
                D=D, 
                period_name=pname
            )

        all_results.append(metrics_list)

    df_results = pd.DataFrame(all_results)

    os.makedirs("./Seasonal_Analysis", exist_ok=True)
    df_results.to_csv("./Seasonal_Analysis/SARIMA_All_Results.csv", index=False)

    print("\n✅ All periods completed, results saved: ./Seasonal_Analysis/SARIMA_All_Results.csv")


if __name__ == '__main__':
    main()
