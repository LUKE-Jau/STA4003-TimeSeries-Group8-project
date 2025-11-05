import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import numpy as np
import os
import warnings
import re

warnings.filterwarnings("ignore")

START_DATE = "2023-01-01"
DATA_DIR = "./data/BTC_factors"

IMAGE_DIR = "./data/selected_factor_imaged"
os.makedirs(IMAGE_DIR, exist_ok=True)

TRAIN_RATIO = 0.8  # 使用前80%作为训练集


def forward_fill_with_limit(series, max_gap=5):
    is_na = series.isna()
    na_groups = (~is_na).cumsum()
    na_counts = is_na.groupby(na_groups).transform("sum")

    series_filled = series.copy()
    mask = (is_na) & (na_counts <= max_gap)
    series_filled[mask] = series_filled[mask].fillna(method="ffill")

    return series_filled


def stl_multiplicative_decomposition(series, period):
    min_val = series.min()
    if min_val <= 0:
        series_shifted = series - min_val + 1e-6
    else:
        series_shifted = series.copy()

    log_series = np.log(series_shifted)
    stl_model = STL(log_series, period=period, robust=True)
    result = stl_model.fit()

    observed = np.exp(result.observed)
    trend = np.exp(result.trend)
    seasonal = np.exp(result.seasonal)
    # resid = np.exp(result.resid)
    resid_corrected = observed / (trend * seasonal)

    return {
        "observed": observed,
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid_corrected,
        "original_result": result,
    }


def process_single_factor(
    filepath, category, factor_name, original_frequency, use_multiplicative=False
):
    if original_frequency not in ["10m", "1h", "24h"]:
        print(f"⚠️ Unsupported frequency: {original_frequency} in {factor_name}")
        return None

    period = 7
    try:
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["datetime"])
        df = df[df["date"] > pd.to_datetime(START_DATE)]
        df = df.set_index("date").sort_index()

        if factor_name not in df.columns:
            raise ValueError(f"Column '{factor_name}' not found.")
        series = df[factor_name].copy()
        series = pd.to_numeric(series, errors="coerce")

        freq_map = {"10m": "10T", "1h": "1H", "24h": "1D"}
        pandas_freq = freq_map.get(original_frequency)
        if pandas_freq:
            series = series.asfreq(pandas_freq)

        series = forward_fill_with_limit(series, max_gap=5)
        series = series.dropna()

        if len(series) < 50:
            raise ValueError("Insufficient data after cleaning.")

        if original_frequency in ["10m", "1h"]:
            daily_series = series.groupby(series.index.date).mean()
            daily_series.index = pd.to_datetime(daily_series.index)
            series = daily_series
            display_freq = "24h (aggregated)"
        else:
            display_freq = "24h"

        if len(series) < 50:
            raise ValueError("Insufficient data after daily aggregation.")

        n_total = len(series)
        n_train = int(n_total * TRAIN_RATIO)
        if n_train < 30:
            raise ValueError("Too few samples in training set (<30).")
        train_series = series.iloc[:n_train]

        if use_multiplicative:
            decomposition_result = stl_multiplicative_decomposition(
                train_series, period
            )
            observed = decomposition_result["observed"].dropna()
            trend = decomposition_result["trend"].dropna()
            seasonal = decomposition_result["seasonal"].dropna()
            resid = decomposition_result["resid"].dropna()
        else:
            stl_model = STL(train_series, period=period, robust=True)
            result = stl_model.fit()
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

        eps = 1e-12
        if use_multiplicative:
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

        decomposition_type = "multiplicative" if use_multiplicative else "additive"
        image_path = os.path.join(
            IMAGE_DIR, f"{factor_name}_{display_freq}_{decomposition_type}.png"
        )

        fig, axes = plt.subplots(6, 1, figsize=(12, 14))

        # 1. Observed (decomposed)
        observed.plot(
            ax=axes[0],
            title=f"{factor_name} ({display_freq}) - {decomposition_type.capitalize()} Decomposition (Train: {TRAIN_RATIO*100:.0f}%)",
        )
        # 2. Trend
        trend.plot(ax=axes[1], title="Trend")
        # 3. Seasonal
        seasonal.plot(ax=axes[2], title=f"Seasonal (Period={period})")
        # 4. Residuals
        resid.plot(ax=axes[3], title="Residuals (Training Set)")

        # 5 & 6. ACF/PACF of ORIGINAL TRAINING SERIES (not residuals)
        original_series_for_acf = train_series
        max_lag = min(350, len(original_series_for_acf) // 2 - 1)
        if max_lag > 0:
            plot_acf(
                original_series_for_acf,
                lags=max_lag,
                ax=axes[4],
                title="ACF of Original Series (Training Set)",
            )
            plot_pacf(
                original_series_for_acf,
                lags=max_lag,
                ax=axes[5],
                title="PACF of Original Series (Training Set)",
            )
        else:
            for i, name in enumerate(["ACF", "PACF"], start=4):
                axes[i].text(
                    0.5,
                    0.5,
                    f"Insufficient data for {name}",
                    transform=axes[i].transAxes,
                    ha="center",
                )
                axes[i].set_title(f"{name} of Original Series (Training Set)")

        plt.tight_layout()
        plt.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        mean_resid = resid.mean()
        std_resid = resid.std()
        skew_resid = stats.skew(resid)
        kurtosis_resid = stats.kurtosis(resid)

        lb_test = acorr_ljungbox(resid, lags=min(20, len(resid) // 2), return_df=True)
        lb_pvalue = lb_test["lb_pvalue"].iloc[-1] if not lb_test.empty else np.nan
        passed_white_noise = lb_pvalue > 0.05 if not np.isnan(lb_pvalue) else False

        _, normal_pvalue = stats.normaltest(resid)
        passed_normality = normal_pvalue > 0.05

        return {
            "factor_name": factor_name,
            "original_frequency": original_frequency,
            "display_frequency": display_freq,
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
            "final_length_total": n_total,
            "train_length": len(train_series),
            "image_path": image_path,
        }

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None


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


def process_selected_factors(factor_list, use_multiplicative=False):
    results = []
    processed_factors = set()

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

    not_found = set(factor_list) - processed_factors
    if not_found:
        print(f"\n⚠️ The following factors were not found: {list(not_found)}")

    decomposition_type = "multiplicative" if use_multiplicative else "additive"
    selected_summary_path = f"./data/selected_factors_analysis_{decomposition_type}.csv"
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(selected_summary_path, index=False)
    print(f"\n📊 Selected factors summary saved to: {selected_summary_path}")
    print(f"📈 Total selected factors processed: {len(summary_df)}")

    return summary_df


# ========== run ==========
if __name__ == "__main__":

    # 批量处理所有因子（加法分解）
    # print("=== Processing all factors with additive decomposition ===")
    # summary_additive = batch_process_factors(use_multiplicative=False)

    print("\n=== Processing selected factors ===")
    selected_factors = ["active_3m_6m"]  # 示例因子列表
    selected_summary = process_selected_factors(
        selected_factors, use_multiplicative=False
    )

    print("\n✅ All done!")
