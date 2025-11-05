import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

warnings.filterwarnings("ignore")

# ========== 配置参数 ==========
START_DATE = "2023-01-01"  # 过滤起始日期
PERIOD = 7  # 日频率的周期（假设周季节性，7 天）
NLAGS_ACF = 20  # 计算残差 ACF 的最大滞后阶数

# 复用指定的目录结构
project_dir = Path(r"C:\Users\Charles Lee\PycharmProjects\year4 term1\STA4003")
data_dir = project_dir / "data"
results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)
file_path = data_dir / "BTCUSDT_10m.csv"


# ========== 安全前向填充缺失值（不泄露未来）==========
def forward_fill_with_limit(series: pd.Series, max_gap: int = 5) -> pd.Series:
    """
    对时间序列进行前向填充，但仅填充连续缺失长度 <= max_gap 的缺口。
    任何更长的缺口仍保持 NaN，避免泄露未来信息。
    """
    is_na = series.isna()
    groups = (~is_na).cumsum()
    na_counts = is_na.groupby(groups).transform("sum")

    filled = series.ffill(limit=max_gap)
    filled[is_na & (na_counts > max_gap)] = np.nan
    return filled


# ========== 主函数：处理 'return' 序列 ==========
def process_return_stl(filepath: Path):
    try:
        df = pd.read_csv(filepath)

        # 明确解析格式，强制转换为 UTC
        df["date"] = pd.to_datetime(
            df["datetime"],
            format="%Y-%m-%d %H:%M:%S",
            utc=True,
            errors="coerce",
        )
        df = df.dropna(subset=["date"])
        df = df[df["date"] >= pd.to_datetime(START_DATE, utc=True)]
        df = df.set_index("date").sort_index()

        # ---------------------- Resample 到 Daily ----------------------
        daily_data = (
            df.resample("D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        if "close" not in daily_data.columns:
            raise ValueError("Column 'close' not found.")

        daily_data = daily_data.tz_convert(None)  # 去掉时区方便绘图

        # === 处理收盘价缺失值 ===
        close_series = (
            pd.to_numeric(daily_data["close"], errors="coerce")
            .asfreq("D")
        )
        close_series = forward_fill_with_limit(close_series, max_gap=5).dropna()

        if len(close_series) < 51:
            raise ValueError("Insufficient close data after cleaning to compute returns.")

        # === 计算简单日收益率 ===
        series = close_series.pct_change().dropna()

        if len(series) < 50:
            raise ValueError("Insufficient return data after differencing.")

        # === STL 分解 ===
        stl_model = STL(series, period=PERIOD, robust=True)
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
        observed = observed[common_idx]
        trend = trend[common_idx]
        seasonal = seasonal[common_idx]
        resid = resid[common_idx]

        if len(resid) < 10:
            raise ValueError("Too few residuals after alignment.")

        # === 指标 1 & 2：Trend Strength & Seasonal Strength ===
        eps = 1e-12
        trend_var = np.var(trend, ddof=1)
        seasonal_var = np.var(seasonal, ddof=1)
        resid_var = np.var(resid, ddof=1)
        trend_strength = trend_var / (trend_var + resid_var + eps)
        seasonal_strength = seasonal_var / (seasonal_var + resid_var + eps)

        # === 指标 3：Trend Slope (线性回归) ===
        t = np.arange(len(trend))
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, trend.values)
        trend_fit = intercept + slope * t

        # === 指标 4：ACF（对残差）的数值分析 ===
        acf_values = acf(resid, nlags=min(NLAGS_ACF, len(resid) - 1), fft=True)
        acf_lags = list(range(len(acf_values)))

        # === 残差检验 ===
        mean_resid = resid.mean()
        std_resid = resid.std(ddof=1)
        skew_resid = stats.skew(resid)
        kurtosis_resid = stats.kurtosis(resid)

        lb_test = acorr_ljungbox(resid, lags=min(20, len(resid) // 2), return_df=True)
        lb_pvalue = lb_test["lb_pvalue"].iloc[-1]
        passed_white_noise = lb_pvalue > 0.05

        _, normal_pvalue = stats.normaltest(resid)
        passed_normality = normal_pvalue > 0.05

        adf_stat, adf_pvalue, *_ = adfuller(resid)
        passed_adf = adf_pvalue < 0.05
        is_stationary = passed_white_noise and passed_adf

        # === 绘图 ===
        image_path = results_dir / "s02btc_return_stl_daily.png"
        fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=False)

        ax_obs, ax_trend, ax_season, ax_resid, ax_slope, ax_acf = axes

        ax_obs.plot(observed.index, observed.values)
        ax_obs.set_title("Observed (Daily Return)")

        ax_trend.plot(trend.index, trend.values)
        ax_trend.set_title("Trend")

        ax_season.plot(seasonal.index, seasonal.values)
        ax_season.set_title(f"Seasonal (Period={PERIOD})")

        ax_resid.plot(resid.index, resid.values)
        ax_resid.set_title("Residuals")

        ax_slope.plot(trend.index, trend.values, label="Trend")
        ax_slope.plot(
            trend.index,
            trend_fit,
            label=f"Linear Fit (Slope: {slope:.6f})",
            color="red",
            linestyle="--",
        )
        ax_slope.set_title("Trend Slope (Linear Fit)")
        ax_slope.legend()

        plot_acf(resid, ax=ax_acf, lags=889)
        ax_acf.set_title("ACF of Residuals")
        ax_acf.set_xlabel("Lag")

        for ax in [ax_obs, ax_trend, ax_season, ax_resid, ax_slope]:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.grid(True, linestyle="--", alpha=0.3)

        ax_acf.grid(True, linestyle="--", alpha=0.3)

        fig.autofmt_xdate(rotation=45)
        fig.suptitle(
            (
                "return (daily from 10m) | "
                f"Trend Strength: {trend_strength:.4f} | "
                f"Seasonal Strength: {seasonal_strength:.4f} | "
                f"Stationary: {is_stationary}"
            ),
            fontsize=14,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        result_dict = {
            "factor_name": "return",
            "original_frequency": "10m (resampled to daily)",
            "period_used": PERIOD,
            # === 指标 ===
            "trend_strength": trend_strength,          # 指标 1
            "seasonal_strength": seasonal_strength,    # 指标 2
            "trend_slope": slope,                      # 指标 3
            "trend_intercept": intercept,
            "trend_r_value": r_value,
            "trend_p_value": p_value,
            "trend_std_err": std_err,
            "acf_lags": acf_lags,                      # 指标 4
            "acf_values": acf_values.tolist(),
            # === 额外统计量 ===
            "mean_resid": mean_resid,
            "std_resid": std_resid,
            "skew_resid": skew_resid,
            "kurtosis_resid": kurtosis_resid,
            "lb_pvalue": lb_pvalue,
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "normal_pvalue": normal_pvalue,
            "passed_white_noise": passed_white_noise,
            "passed_adf": passed_adf,
            "passed_normality": passed_normality,
            "is_stationary": is_stationary,
            "final_length": len(series),
            "image_path": str(image_path),
        }

        print("STL 分解与指标计算结果:")
        for k, v in result_dict.items():
            print(f"  {k}: {v}")
        return result_dict

    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None


# ========== 运行 ==========
if __name__ == "__main__":
    result = process_return_stl(file_path)
    print("\n✅ All done!")
