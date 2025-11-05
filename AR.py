import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

dff = pd.read_csv('./BTC_10m_active_3m_6m.csv', parse_dates=['datetime'])
dff['datetime'] = pd.to_datetime(dff['datetime'])
dff = dff.sort_values('datetime')

dff_adjust = dff[dff['datetime']>='2023-01-01'].copy()
dff_adjust['date'] = dff_adjust['datetime'].dt.date
dff_daily = dff_adjust.groupby('date')['active_3m_6m'].mean().reset_index()   # 日度因子

split_point = int(len(dff_daily) * 0.8)  
train_2 = dff_daily.iloc[:split_point].copy()
test_2 = dff_daily.iloc[split_point:].copy()

train_2 = train_2.set_index('date', drop=True)
test_2 = test_2.set_index('date', drop=True)

ts = train_2['active_3m_6m'].dropna().copy()
model = AutoReg(ts, lags=2).fit()
print(model.summary())
resid = model.resid
lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
print(lb_test)


# error
y_train = train_2['active_3m_6m']
y_test = test_2['active_3m_6m']
t_train = train_2.index
t_test = test_2.index

p = 2
predictions = []
train, test = train_2['active_3m_6m'], test_2['active_3m_6m']

history = train.copy() 

for i in range(len(test)):
    model = AutoReg(history, lags=p).fit()
    yhat = model.predict(start=len(history), end=len(history))
    predictions.append(yhat.values[0])
    new_point = pd.Series([yhat.values[0]], index=[test.index[i]])
    history = pd.concat([history, new_point])


mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)  
mae = mean_absolute_error(test, predictions)
mape = (abs(test - predictions) / abs(test)).mean() * 100
naive_error = np.abs(y_train[1:].values - y_train[:-1].values).mean()
mase = mae / naive_error

print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)
print("MASE:", mase)

# === AR(2) ===
plt.figure(figsize=(15,6))
plt.plot(t_train, y_train, label='Training Data', color='gray', alpha=0.8)
plt.plot(t_test, y_test, label='Actual (Test)', color='black')
plt.plot(t_test, predictions, label='AR(2) Forecast', color='orange', alpha=0.9)
plt.axvline(x=t_test[0], color='green', linestyle='--', label='Train/Test Split')
plt.title('Training + Testing Forecast (AR(2))')
plt.xlabel('Datetime')
plt.ylabel('Active Wallets')
plt.legend()
plt.tight_layout()
plt.show()