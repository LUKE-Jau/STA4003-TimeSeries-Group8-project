import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox

# 加入的packages
from sklearn.linear_model import LinearRegression 
from statsmodels.tsa.stattools import acf

from scipy import stats
import numpy as np
import os
import warnings
import re

warnings.filterwarnings("ignore")

# ========== 配置参数 ==========
START_DATE = "2024-08-01"
DATA_DIR = "BTC factors"

# 频率到周期的映射（日周期）
FREQ_TO_PERIOD = {"10m": 144, "1h": 24, "24h": 7}  # 24 * 6  # 24  # 7 days

# 图像和结果按 period 分开存储
IMAGE_DIR = "./data/image_multi_freq"
os.makedirs(IMAGE_DIR, exist_ok=True)

# ========== 安全前向填充缺失值（不泄露未来）==========
def forward_fill_with_limit(series, max_gap=5):
    """
    用前向填充处理缺失值，但限制最大连续缺失长度
    """
    # 先标记连续缺失段
    is_na = series.isna()
    # 计算连续缺失长度
    na_groups = (~is_na).cumsum()
    na_counts = is_na.groupby(na_groups).transform("sum")

    # 只填充连续缺失 <= max_gap 的段
    series_filled = series.copy()
    mask = (is_na) & (na_counts <= max_gap)
    series_filled[mask] = series_filled[mask].fillna(method="ffill")

    return series_filled


# ========== 主函数：处理单个文件 ==========
def process_single_factor(filepath, category, factor_name, original_frequency):
    period = FREQ_TO_PERIOD.get(original_frequency)
    if period is None:
        print(f"⚠️ Unsupported frequency: {original_frequency} in {factor_name}")
        return None

    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["datetime"])
        df = df[df["date"] > pd.to_datetime(START_DATE)]
        df = df.set_index("date").sort_index()

        if factor_name not in df.columns:
            raise ValueError(f"Column '{factor_name}' not found.")
        series = df[factor_name].copy()
        series = pd.to_numeric(series, errors="coerce")

        # === 处理缺失值：前向填充（不泄露未来）===
        # 先确保时间连续（按频率重采样，但不聚合）
        freq_map = {"10m": "10T", "1h": "1H", "24h": "1D"}
        pandas_freq = freq_map.get(original_frequency)
        if pandas_freq:
            series = series.asfreq(pandas_freq)  # 插入缺失时间点

        # 前向填充短缺口（最多5个连续缺失）
        series = forward_fill_with_limit(series, max_gap=5)
        series = series.dropna()

        if len(series) < 50:
            raise ValueError("Insufficient data after cleaning.")

        # === STL 分解 ===
        stl_model = STL(series, period=period, robust=True)
        result = stl_model.fit()

        # 对齐成分
        observed = result.observed.dropna()
        trend = result.trend.dropna()
        seasonal = result.seasonal.dropna()
        resid = result.resid.dropna()

        common_idx = (
            observed.index.intersection(trend.index)
            .intersection(seasonal.index)
            .intersection(resid.index)
        )
        trend = trend[common_idx]
        seasonal = seasonal[common_idx]
        resid = resid[common_idx]

        if len(resid) < 10:
            raise ValueError("Too few residuals after alignment.")

        # === 强度计算 ===
        # 趋势与季节性强度
        eps = 1e-12
        trend_var = np.var(trend)
        seasonal_var = np.var(seasonal)
        resid_var = np.var(resid)
        trend_strength = trend_var / (trend_var + resid_var + eps)
        seasonal_strength = seasonal_var / (seasonal_var + resid_var + eps)

        # === 剩余特征分析（Feature Analysis）===
        # 趋势斜率（线性拟合）
        t = (trend.index - trend.index[0]).total_seconds().values.reshape(-1, 1) / 3600
        model = LinearRegression().fit(t, trend.values)
        trend_slope = model.coef_[0]

        # ACF 峰值（检查周期性）
        nlags = min(36, len(series)//2)  # 增加到36个滞后点
        acf_values = acf(series, nlags=nlags, fft=True)
        acf_peak = np.max(np.abs(acf_values[1:]))  # 取绝对值最大峰

        # === 保存图像（添加版） ===
        image_path = os.path.join(IMAGE_DIR, f"{factor_name}_{original_frequency}.png")
        fig, axes = plt.subplots(6, 1, figsize=(12, 10))
        
        # STL 分解图（原始分解部分）    
        result.observed.plot(ax=axes[0], title=f"{factor_name} ({original_frequency})")
        result.trend.plot(ax=axes[1], title="Trend")
        result.seasonal.plot(ax=axes[2], title=f"Seasonal (Period={period})")
        result.resid.plot(ax=axes[3], title="Residuals")

        # 趋势斜率图
        axes[4].plot(trend.index, trend.values, label="Trend", color="blue", linewidth=2)
        trend_fit = model.predict(t)
        axes[4].plot(trend.index, model.predict(t), "--", color="red",
                     label=f"Fit (slope={trend_slope:.5f})", linewidth=2)
        axes[4].set_title("Trend Slope (Linear Fit)")
        axes[4].legend()
        axes[4].tick_params(axis='x', rotation=45)

        # ACF 图
        lags = np.arange(len(acf_values))
        axes[5].stem(lags, acf_values)
        axes[5].set_title(f"ACF (Peak={acf_peak:.3f})")
        axes[5].set_xlabel("Lag")
        axes[5].set_ylabel("Autocorrelation")
        
        max_lag = len(acf_values) - 1
        if max_lag > 20:
            axes[5].set_xticks(range(0, max_lag+1, max(1, max_lag//6)))
        else:
            axes[5].set_xticks(range(0, max_lag+1))


        plt.tight_layout()
        plt.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # === 残差检验 ===
        mean_resid = resid.mean()
        std_resid = resid.std()
        skew_resid = stats.skew(resid)
        kurtosis_resid = stats.kurtosis(resid)

        lb_test = acorr_ljungbox(resid, lags=min(20, len(resid) // 2), return_df=True)
        lb_pvalue = lb_test["lb_pvalue"].iloc[-1]
        passed_white_noise = lb_pvalue > 0.05

        _, normal_pvalue = stats.normaltest(resid)
        passed_normality = normal_pvalue > 0.05

        return {
            "factor_name": factor_name,
            "original_frequency": original_frequency,
            "category": category,
            "period_used": period,
            "trend_strength": trend_strength,
            "seasonal_strength": seasonal_strength,
            "mean_resid": mean_resid,
            "std_resid": std_resid,
            "skew_resid": skew_resid,
            "kurtosis_resid": kurtosis_resid,
            "lb_pvalue": lb_pvalue,
            "normal_pvalue": normal_pvalue,
            "passed_white_noise": passed_white_noise,
            "passed_normality": passed_normality,
            "final_length": len(series),
            "image_path": image_path,
        }

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None


# ========== 批量处理 ==========
def batch_process_factors():
    results = []

    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n📂 Processing category: {category}")

        for filename in os.listdir(category_path):
            if not filename.endswith(".csv"):
                continue

            match = re.match(r"BTC_(\d+[mh])_(.+)\.csv", filename)
            if not match:
                print(f"⚠️ Skipping invalid filename: {filename}")
                continue

            freq = match.group(1)
            factor_name = match.group(2)

            if freq not in ["10m", "1h", "24h"]:
                continue

            filepath = os.path.join(category_path, filename)
            print(f"✅ Processing: {factor_name} (freq: {freq})")

            result = process_single_factor(filepath, category, factor_name, freq)
            if result:
                results.append(result)

    summary_df = pd.DataFrame(results)
    summary_csv_path = "./data/analysis_summary_multi_freq.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📊 Summary saved to: {summary_csv_path}")
    print(f"📈 Total valid factors processed: {len(summary_df)}")

    return summary_df


# ========== 运行 ==========
if __name__ == "__main__":
    summary = batch_process_factors()
    print("\n✅ All done!")