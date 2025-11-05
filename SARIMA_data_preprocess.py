import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf
from collections import Counter
from statsmodels.tsa.seasonal import STL


def data_presummary(data, test_size=0.2):
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        cutoff_date = pd.to_datetime('2023-01-01')
        data = data[data['datetime'] >= cutoff_date].copy()
        data = data.reset_index(drop=True)
        split_index = int(len(data) * (1 - test_size))
        data = data.iloc[:split_index]

    print("=" * 50)
    print("Overall Data Summary")
    print("=" * 50)

    print("\n1. Basic Information:")
    print(f"Data Shape: {data.shape}")
    print(f"Time Range: {data['datetime'].min()} to {data['datetime'].max()}")

    print("\n2. Data Completeness Check:")
    if 'active_3m_6m' in data.columns:
        data['active_3m_6m'] = pd.to_numeric(data['active_3m_6m'], errors='coerce')
        null_count = data['active_3m_6m'].isnull().sum()
        print(f"Number of missing values in active_3m_6m: {null_count}")
        print(data['active_3m_6m'].describe())

    return data


def resample_data(data, freq='D'):
    print(f"\n{'='*50}")
    print(f"Data Resampling: {freq}")
    print(f"{'='*50}")

    data_resampled = data.copy()
    data_resampled['datetime'] = pd.to_datetime(data_resampled['datetime'])
    data_resampled = data_resampled.set_index('datetime')

    original_count = len(data_resampled)
    data_resampled = data_resampled.resample(freq).mean()
    data_resampled = data_resampled.ffill().bfill()

    print(f"Original number of data points: {original_count}")
    print(f"Number of data points after resampling: {len(data_resampled)}")
    print(f"Time Range: {data_resampled.index.min()} to {data_resampled.index.max()}")

    return data_resampled.reset_index()


def detect_seasonal_periods_acf(ts_data, max_lags=365):
    print("\n--- ACF Seasonal Period Detection ---")
    nlags = min(max_lags, len(ts_data) // 2)
    acf_values = acf(ts_data, nlags=nlags, fft=True)

    seasonal_periods = {}
    threshold = 0.2  # ACF significance threshold

    common_periods = {
        'WEEKLY': 7,
        'BIWEEKLY': 14,
        'MONTHLY': 30,
        'QUARTERLY': 90,
        'HALF_YEARLY': 180,
        "Strange_year": 296,
        'YEARLY': 365
    }

    for name, period in common_periods.items():
        if period < len(acf_values):
            acf_val = acf_values[period]
            if abs(acf_val) > threshold:
                seasonal_periods[name] = period
                print(f"✓ Detected {name} seasonal period (Period={period}, ACF={acf_val:.3f})")

    significant_lags = [lag for lag in range(2, len(acf_values)) if abs(acf_values[lag]) > threshold]
    if len(significant_lags) > 3:
        intervals = [
            significant_lags[i] - significant_lags[i - 1]
            for i in range(1, len(significant_lags))
            if 2 <= significant_lags[i] - significant_lags[i - 1] <= 180
        ]
        if intervals:
            interval_counts = Counter(intervals)
            for interval, count in interval_counts.most_common(3):
                if count >= 2 and interval not in seasonal_periods.values():
                    seasonal_periods[f'AUTO_{interval}'] = interval
                    print(f"✓ Automatically detected period: {interval} (appeared {count} times)")

    return seasonal_periods


def adf_stationarity_test(ts_data, period):
    ts_diff = ts_data.diff(periods=period).dropna()
    result = adfuller(ts_diff)
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value < 0.05
    D = 0 if is_stationary else 1
    return {
        'period': period,
        'adf_stat': adf_stat,
        'p_value': p_value,
        'is_stationary': is_stationary,
        'recommended_D': D
    }


def plot_seasonal_analysis(ts_data, adf_df, save_dir):
    import matplotlib.dates as mdates
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ts_data.plot(ax=axes[0], color='steelblue')
    axes[0].set_title("Original Time Series (after resample)", fontsize=14)
    axes[0].grid(alpha=0.3)

    plot_acf(ts_data, ax=axes[1], lags=min(365, len(ts_data)//2))
    axes[1].set_title("Autocorrelation Function (ACF)", fontsize=13)

    top3 = adf_df.head(3)
    for i, row in top3.iterrows():
        period = row["period_value"]
        label = f"{row['period_name']} ({period}d)"
        axes[1].axvline(period, color=f"C{i}", linestyle='--', alpha=0.7, label=label)
    axes[1].legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/seasonal_overview.png", dpi=300)
    plt.close()

    main_period = int(top3.iloc[0]['period_value'])
    try:
        stl = STL(ts_data, period=main_period, robust=True)
        res = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        res.observed.plot(ax=axes[0], color='steelblue')
        axes[0].set_ylabel('Observed')
        res.trend.plot(ax=axes[1], color='darkorange')
        axes[1].set_ylabel('Trend')
        res.seasonal.plot(ax=axes[2], color='forestgreen')
        axes[2].set_ylabel('Seasonal')
        res.resid.plot(ax=axes[3], color='crimson')
        axes[3].set_ylabel('Residual')
        axes[3].set_xlabel('Date')

        plt.suptitle(f"STL Decomposition (Period = {main_period} days)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{save_dir}/stl_decomposition.png", dpi=300)
        plt.close()

    except Exception as e:
        print(f"⚠ STL decomposition failed: {e}")


def plot_strength_vs_D(adf_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    df = adf_df.copy().sort_values(by='strength', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='period_name', y='strength', hue='D',
                palette={0: 'steelblue', 1: 'darkorange'})

    for i, row in enumerate(df.itertuples()):
        plt.text(i, row.strength + 0.01,
                 f"{int(row.period_value)}d\np={row.p_value:.3f}",
                 ha='center', va='bottom', fontsize=8)

    plt.title('Seasonal Strength Ranking (with D value)', fontsize=14)
    plt.xlabel('Detected Period')
    plt.ylabel('Strength (ACF)')
    plt.legend(title='Recommended D', loc='upper right', frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/seasonal_strength_analysis.png", dpi=300)
    plt.close()


def comprehensive_seasonal_detection(data, target_column='active_3m_6m'):
    print("=" * 60)
    print("Comprehensive Seasonal Period Detection (ACF + ADF + STL)")
    print("=" * 60)

    data_daily = resample_data(data, freq='D')
    ts_data = data_daily.set_index('datetime')[target_column].dropna()

    acf_values = acf(ts_data, nlags=min(365, len(ts_data)//2), fft=True)
    acf_periods = detect_seasonal_periods_acf(ts_data)

    all_periods = []
    for name, period in acf_periods.items():
        acf_val = acf_values[period] if period < len(acf_values) else np.nan
        all_periods.append({
            'method': 'ACF',
            'period_name': name,
            'period_value': period,
            'strength': abs(acf_val)
        })

    df_periods = pd.DataFrame(all_periods).sort_values(by='strength', ascending=False)
    print("\nDetected Seasonal Periods (sorted by strength):")
    for i, row in df_periods.iterrows():
        print(f"{i+1:02d}. {row['period_name']} Period={row['period_value']} Strength={row['strength']:.3f}")

    print("\nADF Stationarity Test:")
    adf_results = []
    for _, row in df_periods.iterrows():
        res = adf_stationarity_test(ts_data, row['period_value'])
        adf_results.append({
            'period_name': row['period_name'],
            'period_value': row['period_value'],
            'strength': row['strength'],
            'ADF_Stat': res['adf_stat'],
            'p_value': res['p_value'],
            'is_stationary': res['is_stationary'],
            'D': res['recommended_D']
        })
        print(f"  {row['period_name']} ({row['period_value']}d): ADF={res['adf_stat']:.3f}, p={res['p_value']:.3f} → {'Stationary' if res['is_stationary'] else 'Non-stationary'}")

    adf_df = pd.DataFrame(adf_results).sort_values(by='strength', ascending=False).reset_index(drop=True)
    os.makedirs('./Seasonal_Analysis', exist_ok=True)
    adf_df.to_csv('./Seasonal_Analysis/seasonal_periods_detection.csv', index=False)

    plot_seasonal_analysis(ts_data, adf_df, './Seasonal_Analysis')
    plot_strength_vs_D(adf_df, './Seasonal_Analysis')

    return adf_df, ts_data


def main():
    FILE_PATH = './BTC_10m_active_3m_6m.csv'
    raw_data = pd.read_csv(FILE_PATH)
    data_checked = data_presummary(raw_data, test_size=0.2)

    adf_df, ts_data = comprehensive_seasonal_detection(data_checked, 'active_3m_6m')

    print("\n🌟 Final Seasonal Strength + D Value Report 🌟")
    top3 = adf_df.head(3)
    for _, row in top3.iterrows():
        print(f"  Period: {row['period_name']} ({row['period_value']} days)")
        print(f"    → D={row['D']}, Strength={row['strength']:.3f}, Stationarity: {'Stationary' if row['is_stationary'] else 'Non-stationary'}")
        print(f"    → ADF={row['ADF_Stat']:.3f}, p={row['p_value']:.4f}\n")

    print("*" * 80)
    print(f"Top Recommended Period: {top3.iloc[0]['period_name']} ({int(top3.iloc[0]['period_value'])} days)")
    print(f"Corresponding Difference Parameter: D={top3.iloc[0]['D']}")
    print("*" * 80)


if __name__ == "__main__":
    main()
