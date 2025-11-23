from statsmodels.tsa.stattools import adfuller, kpss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
# from sklearn.metrics import mean_squared_error, mean_absolute_error

dff = pd.read_csv('./BTC_1h_profit_relative.csv', parse_dates=['datetime'])
dff['datetime'] = pd.to_datetime(dff['datetime'])
dff = dff.sort_values('datetime')

dff_adjust = dff.copy()
dff_adjust['date'] = dff_adjust['datetime'].dt.date
dff_daily = dff_adjust.groupby('date')['profit_relative'].mean().reset_index()   # 日度因子

split_point = int(len(dff_daily) * 0.8)  
train_2 = dff_daily.iloc[:split_point].copy()
test_2 = dff_daily.iloc[split_point:].copy()

train_2 = train_2.set_index('date', drop=True)
test_2 = test_2.set_index('date', drop=True)

ts = train_2['profit_relative'].dropna().copy()


result_3 = adfuller(ts)


def run_kpss(ts, regression='c', nlags='auto'):
    # regression: 'c' (level), 'ct' (trend)
    res = kpss(ts.dropna(), regression=regression, nlags=nlags)
    out = {
        'kpss_stat': res[0],
        'pvalue': res[1],
        'nlags': res[2],
        'critical_values': res[3]
    }
    return out

kpss_res = run_kpss(ts, regression='c')




# === 假设数据 ===
y_train = train_2['profit_relative']
y_test = test_2['profit_relative']
t_train = train_2.index
t_test = test_2.index

# ===============================
# 1. Mean Method Forecast
# ===============================
mean_forecast = np.full_like(y_test, y_train.mean(), dtype=float)

# ===============================
# 2. Naive Method Forecast
# ===============================
naive_forecast = np.full_like(y_test, y_train.iloc[-1], dtype=float)

# ===============================
# 3. Drift Method Forecast
# ===============================
n_train = len(y_train)
h_steps = np.arange(1, len(y_test) + 1)

# 漂移量（末值 - 首值）
drift = (y_train.iloc[-1] - y_train.iloc[0]) / (n_train - 1)

drift_forecast = y_train.iloc[-1] + h_steps * drift

# ===============================
# 4. Seasonal Naive Method Forecast
# ===============================
seasonal_period = 1 # 例子：每周季节性，你可改成 12（按月）、24（按小时）等
seasonal_forecast = np.array([
    y_train.iloc[-seasonal_period + (i % seasonal_period)]
    for i in range(len(y_test))
])

# ===============================
# 5. 评价指标函数
# ===============================
def calc_metrics(y_true, y_pred, y_train):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))

    naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
    mase = mae / naive_mae

    return rmse, mae, mape, mase, naive_mae

metrics_mean = calc_metrics(y_test.values, mean_forecast, y_train.values)
metrics_naive = calc_metrics(y_test.values, naive_forecast, y_train.values)
metrics_drift = calc_metrics(y_test.values, drift_forecast, y_train.values)
metrics_seasonal = calc_metrics(y_test.values, seasonal_forecast, y_train.values)

# ===============================
# 6. 打印表格
# ===============================
metrics_df = pd.DataFrame(
    [metrics_mean, metrics_naive, metrics_drift, metrics_seasonal],
    index=['Mean', 'Naive', 'Drift', 'Seasonal Naive (p=1)'],
    columns=['RMSE','MAE','MAPE','MASE', 'naive_mae']
)
print(metrics_df)

# ===============================
# 7. 绘图
# ===============================
plt.figure(figsize=(15, 6))

plt.plot(t_train, y_train, label='Training Data', color='gray', alpha=0.8)
plt.plot(t_test, y_test, label='Actual (Test)', color='black')

plt.plot(t_test, mean_forecast, label='Mean Forecast', color='blue')
plt.plot(t_test, naive_forecast, label='Naive Forecast', color='red')
plt.plot(t_test, drift_forecast, label='Drift Forecast', color='purple')
plt.plot(t_test, seasonal_forecast, label='Seasonal Naive (p=1)', color='orange')

plt.axvline(x=t_test[0], color='green', linestyle='--', label='Train/Test Split')

plt.title('Forecast Comparison: Mean / Naive / Drift / Seasonal Naive')
plt.xlabel('Datetime')
plt.ylabel('profit_relative')
plt.legend()
plt.tight_layout()
plt.show()
