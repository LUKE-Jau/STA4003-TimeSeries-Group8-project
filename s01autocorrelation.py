import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf  # 用于计算ACF
from statsmodels.graphics.tsaplots import plot_acf  # 用于绘制ACF图

# ---------------------- Data Preparation ----------------------
# ---------------------- 1. 目录设置与数据读取 ----------------------
project_dir = Path(r'C:\Users\Charles Lee\PycharmProjects\year4 term1\STA4003')
data_dir = project_dir / "data"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)
file_path = data_dir / "BTCUSDT_10m.csv"

try:
    data = pd.read_csv(file_path)
    data["datetime"] = pd.to_datetime(data["datetime"], utc=True)  # 显式设置为UTC
    data.set_index("datetime", inplace=True)
    data = data[["open", "high", "low", "close", "volume"]].dropna()
    if data.empty:
        raise ValueError("数据为空，请检查CSV文件。")
except Exception as e:
    print(f"错误：读取数据失败 - {e}")
    raise

# ---------------------- 新增：Resample到Daily ----------------------
# Resample到daily级别（'D' 表示日频率）
daily_data = data.resample('D').agg({
    'open': 'first',    # 日开盘价：第一个10分钟开盘
    'high': 'max',      # 日最高价
    'low': 'min',       # 日最低价
    'close': 'last',    # 日收盘价：最后一个10分钟收盘
    'volume': 'sum'     # 日成交量：总和
}).dropna()  # 去除任何空日（e.g., 非完整日）

# 打印resample信息（验证）
print(f"原始数据点数: {len(data)} | Daily数据点数: {len(daily_data)}")
print(f"Daily数据起始日期: {daily_data.index.min()} | 结束日期: {daily_data.index.max()}")

# ---------------------- 计算Autocorrelation Function (ACF) ----------------------
# 选择序列（这里用'close'；如果想用'volume'，改为 daily_data['volume']）
series = daily_data['close']

# 计算ACF（nlags=52，类似于R的lag_max=52；返回从lag 0开始的数组，但我们从lag 1开始显示）
acf_values, confint = acf(series, nlags=52, alpha=0.05, fft=True)  # alpha=0.05 返回置信区间（可选）

# 创建DataFrame，类似于R的tsibble（从lag 1开始，忽略lag 0）
acf_df = pd.DataFrame({
    'lag': range(1, 53),  # lag 1 to 52
    'acf': acf_values[1:]  # ACF值（从lag 1开始）
})

# 打印ACF表格（类似于R输出）
print("ACF Results (lag_max=52) on Daily Data:")
print(acf_df.to_string(index=False))  # 无索引，干净输出

# ---------------------- 绘制ACF图并保存 ----------------------
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)  # 增大宽度以适应更多lag
plot_acf(series, lags=52, ax=ax, title="Autocorrelation Function (ACF) for BTCUSDT Daily Close Price")
ax.set_xlabel("Lag (Days)")
ax.set_ylabel("ACF")
plt.tight_layout()

# 保存ACF图
save_path = results_dir / "s01btc_acf_plot_daily_lag52.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.show()  # 取消注释可直接显示图表

print(f"Daily ACF图表（lag_max=52）已保存到: {save_path}")