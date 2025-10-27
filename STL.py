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


# ========== STL乘法分解函数 ==========
def stl_multiplicative_decomposition(series, period):
    """
    对时间序列进行乘法STL分解（通过log变换实现）
    """
    # 确保所有值为正数（用于log变换）
    min_val = series.min()
    if min_val <= 0:
        # 如果有负值或零，平移使其为正
        series_shifted = series - min_val + 1e-6
    else:
        series_shifted = series.copy()

    # 对数据取log
    log_series = np.log(series_shifted)

    # 进行STL分解
    stl_model = STL(log_series, period=period, robust=True)
    result = stl_model.fit()

    # 指数变换回原尺度
    observed = np.exp(result.observed)
    trend = np.exp(result.trend)
    seasonal = np.exp(result.seasonal)
    resid = np.exp(result.resid)

    # 重新构建，确保 observed = trend * seasonal * resid
    # （由于数值精度，可能略有偏差，这里重新计算resid）
    resid_corrected = observed / (trend * seasonal)

    return {
        "observed": observed,
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid_corrected,
        "original_result": result,  # 保留原始结果用于其他分析
    }


# ========== 主函数：处理单个文件 ==========
def process_single_factor(
    filepath, category, factor_name, original_frequency, use_multiplicative=False
):
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
        if use_multiplicative:
            # 乘法分解
            decomposition_result = stl_multiplicative_decomposition(series, period)
            observed = decomposition_result["observed"].dropna()
            trend = decomposition_result["trend"].dropna()
            seasonal = decomposition_result["seasonal"].dropna()
            resid = decomposition_result["resid"].dropna()
        else:
            # 加法分解
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
        eps = 1e-12
        if use_multiplicative:
            # 对于乘法分解，使用log尺度的方差来计算强度
            log_trend = np.log(trend)
            log_seasonal = np.log(seasonal)
            log_resid = np.log(resid)

            trend_var = np.var(log_trend)
            seasonal_var = np.var(log_seasonal)
            resid_var = np.var(log_resid)
            trend_strength = trend_var / (trend_var + resid_var + eps)
            seasonal_strength = seasonal_var / (seasonal_var + resid_var + eps)
        else:
            trend_var = np.var(trend)
            seasonal_var = np.var(seasonal)
            resid_var = np.var(resid)
            trend_strength = trend_var / (trend_var + resid_var + eps)
            seasonal_strength = seasonal_var / (seasonal_var + resid_var + eps)

        # === 保存图像 ===
        decomposition_type = "multiplicative" if use_multiplicative else "additive"
        image_path = os.path.join(
            IMAGE_DIR, f"{factor_name}_{original_frequency}_{decomposition_type}.png"
        )
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))

        if use_multiplicative:
            observed.plot(
                ax=axes[0],
                title=f"{factor_name} ({original_frequency}) - Multiplicative Decomposition",
            )
            trend.plot(ax=axes[1], title="Trend")
            seasonal.plot(ax=axes[2], title=f"Seasonal (Period={period})")
            resid.plot(ax=axes[3], title="Residuals")
        else:
            result.observed.plot(
                ax=axes[0],
                title=f"{factor_name} ({original_frequency}) - Additive Decomposition",
            )
            result.trend.plot(ax=axes[1], title="Trend")
            result.seasonal.plot(ax=axes[2], title=f"Seasonal (Period={period})")
            result.resid.plot(ax=axes[3], title="Residuals")

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
            "decomposition_type": decomposition_type,
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
def batch_process_factors(use_multiplicative=False):
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
            print(
                f"✅ Processing: {factor_name} (freq: {freq}) - {['Additive', 'Multiplicative'][use_multiplicative]} decomposition"
            )

            result = process_single_factor(
                filepath,
                category,
                factor_name,
                freq,
                use_multiplicative=use_multiplicative,
            )
            if result:
                results.append(result)

    decomposition_suffix = "_multiplicative" if use_multiplicative else ""
    summary_csv_path = f"./data/analysis_summary_multi_freq{decomposition_suffix}.csv"
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📊 Summary saved to: {summary_csv_path}")
    print(f"📈 Total valid factors processed: {len(summary_df)}")

    return summary_df


# ========== 处理指定因子列表的函数 ==========
def process_selected_factors(factor_list, use_multiplicative=False):
    """
    处理并绘图指定的因子列表

    Parameters:
    factor_list: list of str, 要处理的因子名称列表
    use_multiplicative: bool, 是否使用乘法分解

    Returns:
    summary_df: DataFrame, 包含所选因子的统计指标汇总
    """
    results = []
    processed_factors = set()  # 用于跟踪已处理的因子，避免重复

    for category in os.listdir(DATA_DIR):
        category_path = os.path.join(DATA_DIR, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\n🔍 Searching in category: {category}")

        for filename in os.listdir(category_path):
            if not filename.endswith(".csv"):
                continue

            match = re.match(r"BTC_(\d+[mh])_(.+)\.csv", filename)
            if not match:
                continue

            freq = match.group(1)
            factor_name = match.group(2)

            if freq not in ["10m", "1h", "24h"]:
                continue

            # 检查这个因子是否在我们想要处理的列表中
            if factor_name in factor_list and factor_name not in processed_factors:
                filepath = os.path.join(category_path, filename)
                print(
                    f"🎯 Processing selected factor: {factor_name} (freq: {freq}) - {['Additive', 'Multiplicative'][use_multiplicative]} decomposition"
                )

                result = process_single_factor(
                    filepath,
                    category,
                    factor_name,
                    freq,
                    use_multiplicative=use_multiplicative,
                )
                if result:
                    results.append(result)
                    processed_factors.add(factor_name)

    # 检查是否有因子没有找到
    not_found = set(factor_list) - processed_factors
    if not_found:
        print(f"\n⚠️ The following factors were not found: {list(not_found)}")

    # 保存结果
    decomposition_type = "multiplicative" if use_multiplicative else "additive"
    selected_summary_path = f"./data/selected_factors_analysis_{decomposition_type}.csv"
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(selected_summary_path, index=False)
    print(f"\n📊 Selected factors summary saved to: {selected_summary_path}")
    print(f"📈 Total selected factors processed: {len(summary_df)}")

    return summary_df


# ========== 运行 ==========
if __name__ == "__main__":

    # 批量处理所有因子（加法分解）
    print("=== Processing all factors with additive decomposition ===")
    summary_additive = batch_process_factors(use_multiplicative=False)

    # # 批量处理所有因子（乘法分解）
    # print("\n=== Processing all factors with multiplicative decomposition ===")
    # summary_multiplicative = batch_process_factors(use_multiplicative=True)

    print("\n=== Processing selected factors ===")
    selected_factors = [
        "active_1m_3m",
        "count",
        "supply_balance_less_0001",
        "liquid_sum",
    ]  # 示例因子列表
    selected_summary = process_selected_factors(
        selected_factors, use_multiplicative=True
    )

    print("\n✅ All done!")
