import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)              # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]                # 取最后一个时间步的输出
        out = self.fc(out)
        return out


df = pd.read_csv('./data/BTC factors/addresses/BTC_1h_profit_relative.csv')
df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
start_time = '2021-01-01 00:00:00'
df = df.sort_values('datetime')
df.set_index(df.columns[0], inplace=True)
df = df.resample('D').mean()
df = df[start_time:]
values = df["profit_relative"].values.astype(float)
values =values.reshape(-1,1)
train_size = int(len(values) * 0.8)
train_data_raw = values[:train_size]
test_data_raw = values[train_size:]

# 2. 只用训练数据拟合scaler
scaler = MinMaxScaler()
train_data= scaler.fit_transform(train_data_raw)

# 3. 用训练数据的scaler转换测试数据
test_data= scaler.transform(test_data_raw)


window_size = 7
X_train, y_train = create_dataset(train_data, window_size)
X_test, y_test = create_dataset(test_data, window_size)

# 转为 Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)





model = LSTMModel()

criterion = nn.SmoothL1Loss()  # 平滑且对异常值不敏感
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
from tqdm import tqdm
epochs = 1000

for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {loss.item():.6f}")



model.eval()
with torch.no_grad():
    train_pred = model(X_train)
    test_pred = model(X_test)

# 反归一化到原始数值
train_pred = scaler.inverse_transform(train_pred.numpy())
y_train_true = scaler.inverse_transform(y_train.numpy())

test_pred = scaler.inverse_transform(test_pred.numpy())
y_test_true = scaler.inverse_transform(y_test.numpy())

MSE = np.mean((test_pred-y_test_true)**2)
MAE = np.mean(np.abs(test_pred-y_test_true))
RMSE = np.sqrt(MSE)
MAPE = np.mean(np.abs((test_pred-y_test_true)/y_test_true))
MASE = np.mean(np.abs(y_test_true-test_pred))/np.mean(np.abs(np.array(y_test_true[1:])-np.array(y_test_true[:-1])))
result= [{'Method':' LSTM','RMSE':RMSE,'MAE':MAE,'MAPE':MAPE,'MASE':MASE}]
result_frame = pd.DataFrame(result)
print(result_frame)



plt.figure(figsize=(10,5))
plt.plot(range(len(y_train_true)), y_train_true, label="Train True")
plt.plot(range(len(train_pred)), train_pred, linestyle="--",label="Train Pred")

plt.plot(range(len(y_train_true), len(y_train_true)+len(y_test_true)), y_test_true, linestyle="--",label="Test True")
plt.plot(range(len(y_train_true), len(y_train_true)+len(y_test_true)), test_pred,label="Test Pred")
plt.legend()
plt.title("LSTM Time Series Forecasting")
plt.show()


