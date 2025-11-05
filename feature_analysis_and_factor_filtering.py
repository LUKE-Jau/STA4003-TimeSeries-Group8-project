import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox

# 加入的packages
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf, adfuller, kpss

# from statsmodels.graphics.tsaplots import plot_acf

from scipy import stats
import numpy as np
import os
import re
import pickle
import warnings

warnings.filterwarnings("ignore")


# ========== 配置参数 ==========
START_DATE = "2023-01-01"
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

        # Step 4: 平稳性检验 (在STL分解前进行)
        # ADF检验
        adf_result = adfuller(series)
        adf_p = adf_result[1]

        # KPSS检验
        kpss_result = kpss(series, regression="c")  # 'c' for constant
        kpss_p = kpss_result[1]

        # 判断平稳（ADF p<0.05 和 KPSS p>0.05）
        is_stationary = (adf_p < 0.05) and (kpss_p > 0.05)

        station_method = "none"
        series_original = series.copy()  # 备份原始序列用于可能的重置

        if not is_stationary:
            # 平稳化处理
            transformed = False

            # 尝试对数变换（如果序列全正）
            if (series > 0).all():
                series_log = np.log(series)
                adf_log = adfuller(series_log)[1]
                kpss_log = kpss(series_log, regression="c")[1]
                if (adf_log < 0.05) and (kpss_log > 0.05):
                    series = series_log.dropna()
                    station_method = "log"
                    transformed = True

            if not transformed:
                # 尝试一阶差分
                series_diff1 = series.diff().dropna()
                adf_diff1 = adfuller(series_diff1)[1]
                kpss_diff1 = kpss(series_diff1, regression="c")[1]
                if (adf_diff1 < 0.05) and (kpss_diff1 > 0.05):
                    series = series_diff1
                    station_method = "diff1"
                    transformed = True
                else:
                    # 尝试二阶差分
                    series_diff2 = series_diff1.diff().dropna()
                    adf_diff2 = adfuller(series_diff2)[1]
                    kpss_diff2 = kpss(series_diff2, regression="c")[1]
                    if (adf_diff2 < 0.05) and (kpss_diff2 > 0.05):
                        series = series_diff2
                        station_method = "diff2"
                        transformed = True

            # 如果仍非平稳，使用原始序列但标记
            if not transformed:
                series = series_original
                station_method = "failed"

        # 验证处理后平稳性
        if station_method != "none" and station_method != "failed":
            adf_after = adfuller(series)[1]
            kpss_after = kpss(series, regression="c")[1]
            is_stationary_after = (adf_after < 0.05) and (kpss_after > 0.05)
        else:
            is_stationary_after = is_stationary
            adf_after = adf_p
            kpss_after = kpss_p

        # === Step 2: STL 分解（使用平稳化后的序列，如果适用） ===
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

        # === Step 3: 强度计算 ===
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
        nlags = min(36, len(series) // 2)  # 增加到36个滞后点
        acf_values = acf(series, nlags=nlags, fft=True)
        acf_peak = np.max(np.abs(acf_values[1:]))  # 取绝对值最大峰

        # === 保存图像（添加版） ===
        image_path = os.path.join(IMAGE_DIR, f"{factor_name}_{original_frequency}.png")
        fig, axes = plt.subplots(6, 1, figsize=(12, 10))

        # STL 分解图（原始分解部分）
        result.observed.plot(
            ax=axes[0],
            title=f"{factor_name} ({original_frequency}) - Stationary: {is_stationary_after}",
        )
        result.trend.plot(ax=axes[1], title="Trend")
        result.seasonal.plot(ax=axes[2], title=f"Seasonal (Period={period})")
        result.resid.plot(ax=axes[3], title="Residuals")

        # 趋势斜率图
        axes[4].plot(
            trend.index, trend.values, label="Trend", color="blue", linewidth=2
        )
        trend_fit = model.predict(t)
        axes[4].plot(
            trend.index,
            trend_fit,
            "--",
            color="red",
            label=f"Fit (slope={trend_slope:.5f})",
            linewidth=2,
        )
        axes[4].set_title("Trend Slope (Linear Fit)")
        axes[4].legend()
        axes[4].tick_params(axis="x", rotation=45)

        # ACF 图
        lags = np.arange(len(acf_values))
        axes[5].stem(lags, acf_values)
        axes[5].set_title(f"ACF (Peak={acf_peak:.3f})")
        axes[5].set_xlabel("Lag")
        axes[5].set_ylabel("Autocorrelation")

        max_lag = len(acf_values) - 1
        if max_lag > 20:
            axes[5].set_xticks(range(0, max_lag + 1, max(1, max_lag // 6)))
        else:
            axes[5].set_xticks(range(0, max_lag + 1))

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
            "trend_slope": trend_slope,
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
            "adf_p": adf_p,
            "kpss_p": kpss_p,
            "is_stationary": is_stationary,
            "station_method": station_method,
            "is_stationary_after": is_stationary_after,
            "series": series,  # 返回平稳化后的序列
            "series_original": series_original,  # 返回原始序列（可选用于比较）
        }

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None


# ========== 批量处理 ==========
def batch_process_factors():
    results = []
    series_dict = {}  # 存储每个因子的序列（平稳化后）

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
                key = f"{factor_name}_{freq}"
                series_dict[key] = result.pop("series")  # 移除series以保存到DF
                result.pop("series_original")  # 移除原始序列（如果不需要保存）
                results.append(result)

    # save results to CSV
    summary_df = pd.DataFrame(results)
    summary_csv_path = "./data/analysis_summary_multi_freq.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\n📊 Summary saved to: {summary_csv_path}")
    print(f"📈 Total valid factors processed: {len(summary_df)}")

    # Save series_dict to a pickle file
    series_dict_path = "./data/series_dict.pkl"
    with open(series_dict_path, "wb") as f:
        pickle.dump(series_dict, f)
    print(f"📊 Series dictionary saved to: {series_dict_path}")

    return summary_df, series_dict


# ========== Step 5: 筛选代表性因子 ==========
def select_representative_factors(summary_df, series_dict):
    freq_map = {"10m": "10T", "1h": "1H"}
    freq_minutes = {"10m": 10, "1h": 60}

    # 查找比特币收盘价因子（假设factor_name包含'close'或'price'，freq='10m'）
    close_rows = summary_df[
        (summary_df["original_frequency"] == "10m")
        & (summary_df["factor_name"].str.contains("close", case=False))
    ]
    if len(close_rows) == 0:
        close_rows = summary_df[
            (summary_df["original_frequency"] == "10m")
            & (summary_df["factor_name"].str.contains("price", case=False))
        ]
    if len(close_rows) == 0:
        print(
            "❌ Bitcoin close price factor not found. Assuming 'close' or 'price' in name."
        )
        return
    close_name = close_rows["factor_name"].iloc[0]
    close_key = f"{close_name}_10m"
    if close_key not in series_dict:
        print("❌ Bitcoin close price series not found.")
        return
    close_series_10m = series_dict[close_key]
    close_price_df = pd.DataFrame({"close": close_series_10m})

    # 分别处理10m和1h频率
    for freq in ["10m", "1h"]:
        print(f"\n🔍 Selecting for frequency: {freq}")

        df_freq = summary_df[summary_df["original_frequency"] == freq]
        if len(df_freq) == 0:
            print(f"No factors for {freq}")
            continue

        # 优先平稳因子（使用after，如果适用）
        df_freq["is_stationary_final"] = (
            df_freq["is_stationary_after"]
            if "is_stationary_after" in df_freq
            else df_freq["is_stationary"]
        )

        ############
        print(
            f"Before filtering, there are {len(df_freq)} factors for frequency {freq}."
        )

        # 设置阈值：季节性或趋势强度 > 0.7，或趋势斜率绝对值 > 90%分位数
        slope_threshold = df_freq["trend_slope"].abs().quantile(0.9)
        candidates = df_freq[
            (df_freq["seasonal_strength"] > 0.7)
            | (df_freq["trend_strength"] > 0.7)
            | (df_freq["trend_slope"].abs() > slope_threshold)
        ]

        if len(candidates) < 3:
            # 如果不足，取前10%
            candidates = df_freq.sort_values(
                by=["seasonal_strength", "trend_strength"], ascending=False
            ).head(max(5, int(len(df_freq) * 0.1)))

        # 排序：优先平稳，然后季节性和趋势强度
        candidates = candidates.sort_values(
            by=["is_stationary_final", "seasonal_strength", "trend_strength"],
            ascending=[False, False, False],
        )
        ####################
        print(
            f"Candidates for {freq} after setting threshold on stationarity, seasonal strength and trend strength: {len(candidates)} factors"
        )
        # display(pd.DataFrame(candidates))

        # 计算与比特币未来回报的相关性
        corrs = []

        pandas_freq = freq_map[freq]
        close_freq = close_price_df.resample(pandas_freq).last().dropna()
        return_series = close_freq["close"].pct_change().shift(-1).dropna()

        for idx, row in candidates.iterrows():
            factor_key = f"{row['factor_name']}_{freq}"
            factor_series = series_dict.get(factor_key)
            if factor_series is None:
                print(
                    f"⚠️ Factor series not found for key: {factor_key}, appending 0 correlation"
                )
                corrs.append(0)
                continue
            common_index = factor_series.index.intersection(return_series.index)
            if len(common_index) < 10:
                print(
                    f"⚠️ Insufficient common data points for factor {factor_key} and price series, appending 0 correlation"
                )
                corrs.append(0)
                continue
            corr = factor_series[common_index].corr(return_series[common_index])
            corrs.append(abs(corr))

        candidates["corr_with_price"] = corrs
        # Only chooes the corr with price that are greater than 0.005
        candidates = candidates[candidates["corr_with_price"] > 0.005]
        # show the corrs for all the candidates
        print(corrs)

        candidates = candidates.sort_values(
            by=["seasonal_strength", "trend_strength", "corr_with_price"],
            ascending=False,
        )

        if len(candidates) < 3:
            print(f"⚠️ Insufficient candidates with high correlation for {freq}")
            print("Length of candidates:", len(candidates))
            continue

        #################
        print(
            f"Candidates after correlation with BTC price filter for {freq}: {len(candidates)} factors"
        )

        ##################
        print(candidates)

        # 为每个候选因子计算不同return周期的corr并绘制半衰期图
        print("\n📈 Generating half-life plots for each candidate factor...")

        horizons = {
            "0.5h": 0.5,
            "1h": 1,
            "2h": 2,
            "3h": 3,
            "4h": 4,
            "6h": 6,
            "8h": 8,
            "12h": 12,
            "16h": 16,
            "20h": 20,
            "24h": 24,
            "30h": 30,
            "36h": 36,
            "42h": 42,
            "48h": 48,
        }  # 以小时为单位的周期数

        for idx, row in candidates.iterrows():
            factor_key = f"{row['factor_name']}_{freq}"
            factor_series = series_dict.get(factor_key)
            if factor_series is None:
                print(f"⚠️ Skipping plot for {factor_key}: Factor series not found")
                continue

            corr_dict = {}
            freq_min = freq_minutes[freq]
            for label, hor_h in horizons.items():
                # 计算 periods
                periods = max(1, round(hor_h * 60 / freq_min))
                # 计算未来periods期的return：从t到t+periods的pct_change
                future_return = (
                    close_freq["close"]
                    .pct_change(periods=periods)
                    .shift(-periods)
                    .dropna()
                )

                common_index = factor_series.index.intersection(future_return.index)
                if len(common_index) < 10:
                    print(
                        f"⚠️ Skipping {label} for {factor_key}: Insufficient common data points"
                    )
                    corr_dict[label] = 0
                    continue

                corr = factor_series.loc[common_index].corr(
                    future_return.loc[common_index]
                )
                corr_dict[label] = abs(corr)  # 使用绝对值来观察衰减

            if not corr_dict:
                print(f"⚠️ No correlations computed for {factor_key}")
                continue

            # 绘制图表
            plt.figure(figsize=(8, 5))
            x = list(corr_dict.keys())
            y = [corr_dict.get(label, 0) for label in x]
            plt.plot(x, y, marker="o")
            plt.title(
                f"Half-Life Plot for Factor: {row['factor_name']} (Abs Corr vs Return Period)"
            )
            plt.xlabel("Return Period")
            plt.ylabel("Absolute Correlation")
            plt.grid(True)
            plt.show()

            # 可选：粗略估计半衰期（找到corr降到初始值一半的第一个周期）
            if y[0] > 0:
                half_value = y[0] / 2
                half_life_period = next(
                    (label for label, val in zip(x, y) if val <= half_value),
                    "Beyond 48h",
                )
                print(
                    f"Estimated half-life for {row['factor_name']}: {half_life_period}"
                )
            else:
                print(
                    f"No meaningful half-life for {row['factor_name']} (initial corr <= 0)"
                )

        selected_keys = []
        selected_rows = []

        for idx, row in candidates.iterrows():
            factor_key = f"{row['factor_name']}_{freq}"
            factor_series = series_dict[factor_key]
            high_corr = False
            for sel_key in selected_keys:
                sel_series = series_dict[sel_key]
                common = factor_series.index.intersection(sel_series.index)
                if len(common) < 10:
                    continue
                corr = factor_series[common].corr(sel_series[common])
                if abs(corr) > 0.8:
                    high_corr = True
                    break
            if not high_corr:
                selected_keys.append(factor_key)
                selected_rows.append(row)
            if len(selected_keys) >= 5:
                break

        selected_df = pd.DataFrame(selected_rows)

        # 输出选定因子并解释
        print(f"Selected {len(selected_keys)} factors for {freq}:")
        for i, row in enumerate(selected_rows):
            print(f"- {row['factor_name']} (Category: {row['category']})")
            print(
                f"  Reasons: Seasonal Strength={row['seasonal_strength']:.3f}, Trend Strength={row['trend_strength']:.3f}, "
            )
            # print(f"  Trend Slope={row['trend_slope']:.5f}, Corr with Price={row['corr_with_price']:.3f}, ")
            print(
                f"  Stationary: {row['is_stationary_final']}, Method: {row['station_method']}"
            )
            # print(f"  Low correlation with other selected factors.")

        # 保存选定结果（可选）
        selected_csv = f"./data/selected_factors_{freq}.csv"
        selected_df.to_csv(selected_csv, index=False)
        print(f"📊 Selected factors saved to: {selected_csv}")


# ========== 运行 ==========
if __name__ == "__main__":
    summary, series_dict = batch_process_factors()
    select_representative_factors(summary, series_dict)
    print("\n✅ All done!")
