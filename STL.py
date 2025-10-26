import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import numpy as np
import os
import warnings
import re

warnings.filterwarnings("ignore")

# ========== 配置参数 ==========
START_DATE = "2024-08-01"
DATA_DIR = "./data/BTC_factors"

# PERIOD = 168  # 统一使用 24*7 = 168
PERIOD = 24

IMAGE_DIR = f"./data/image_{PERIOD}"
os.makedirs(IMAGE_DIR, exist_ok=True)


# ========== 主函数：处理单个文件 ==========
def process_single_factor(filepath, category, factor_name, original_frequency):
    if original_frequency == "24h":
        print(f"⏭️ Skipping 24h data: {factor_name}")
        return None

    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["datetime"])
        df = df[df["date"] > pd.to_datetime(START_DATE)]
        df = df.set_index("date")

        if factor_name not in df.columns:
            raise ValueError(f"Column '{factor_name}' not found.")
        series = df[factor_name].copy()
        series = pd.to_numeric(series, errors="coerce").dropna()

        if len(series) == 0:
            raise ValueError("No valid numeric data.")

        # Resample to 1H
        if original_frequency in ["10m", "1h"]:
            series = series.resample("1H").mean().dropna()
        else:
            raise ValueError(f"Unsupported frequency: {original_frequency}")

        if len(series) < 50:
            raise ValueError("Insufficient data after resampling.")

        # STL 分解
        stl_model = STL(series, period=PERIOD, robust=True)
        result = stl_model.fit()

        # === 对齐 STL 成分（避免边界 NaN 导致长度不一致）===
        observed = result.observed.dropna()
        trend = result.trend.dropna()
        seasonal = result.seasonal.dropna()
        resid = result.resid.dropna()

        # 取共同索引
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

        # === 计算趋势强度 & 季节性强度 ===
        eps = 1e-12  # 防止除零
        trend_var = np.var(trend)
        seasonal_var = np.var(seasonal)
        resid_var = np.var(resid)

        trend_strength = trend_var / (trend_var + resid_var + eps)
        seasonal_strength = seasonal_var / (seasonal_var + resid_var + eps)

        # === 保存图像 ===
        image_path = os.path.join(
            IMAGE_DIR, f"{factor_name}_{original_frequency}_to_1h.png"
        )
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        result.observed.plot(
            ax=axes[0], title=f"{factor_name} ({original_frequency} → 1h)"
        )
        result.trend.plot(ax=axes[1], title="Trend")
        result.seasonal.plot(ax=axes[2], title=f"Seasonal (Period={PERIOD})")
        result.resid.plot(ax=axes[3], title="Residuals")
        plt.tight_layout()
        plt.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # === 残差统计检验 ===
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
            if freq == "24h":
                print(f"⏭️ Skipping 24h file: {filename}")
                continue

            filepath = os.path.join(category_path, filename)
            print(f"✅ Processing: {factor_name} (original: {freq})")

            result = process_single_factor(filepath, category, factor_name, freq)
            if result:
                results.append(result)

    summary_df = pd.DataFrame(results)
    summary_csv_path = f"./data/analysis_summary_{PERIOD}_1h.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📊 Summary saved to: {summary_csv_path}")
    print(f"📈 Total valid factors processed: {len(summary_df)}")

    return summary_df


# ========== 运行 ==========
if __name__ == "__main__":
    summary = batch_process_factors()
    print("\n✅ All done!")
    if not summary.empty:
        print("\nTop results by seasonal strength:")
        print(
            summary[
                [
                    "factor_name",
                    "original_frequency",
                    "trend_strength",
                    "seasonal_strength",
                    "lb_pvalue",
                ]
            ]
            .sort_values("seasonal_strength", ascending=False)
            .head()
        )
