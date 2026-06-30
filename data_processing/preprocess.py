"""
数据预处理流水线

将原始分钟级 .txt 数据转换为日级多变量时间序列，构建滑动窗口。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from utils.config import AGGREGATION_RULES, INPUT_WINDOW, TRAIN_RATIO


def load_raw_data(data_path: str) -> pd.DataFrame:
    """加载原始 txt 文件，合并 Date 和 Time 列为 datetime index"""
    df = pd.read_csv(
        data_path,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    # 合并日期和时间
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"]).set_index("datetime")
    # 列名统一转小写
    df.columns = [col.lower() for col in df.columns]
    # 转换为数值类型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """处理缺失值：前向填充后线性插值"""
    print(f"  缺失值统计: {df.isna().sum().sum()} 个")
    df = df.ffill().bfill()  # 前后向填充
    df = df.interpolate(method="linear")  # 线性插值剩余缺失
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """按天汇总数据

    根据 AGGREGATION_RULES:
      - sum: 功率/能耗类变量
      - mean: 电压/电流类变量
    """
    daily = pd.DataFrame(index=df.resample("D").mean().index)

    for col, rule in AGGREGATION_RULES.items():
        if col not in df.columns:
            print(f"  警告: 列 {col} 不存在，跳过")
            continue
        if rule == "sum":
            daily[col] = df[col].resample("D").sum()
        elif rule == "mean":
            daily[col] = df[col].resample("D").mean()

    # 计算 sub_metering_remainder（剩余能耗）
    # formula: (global_active_power * 1000 / 60) - sum of sub_meterings
    # 每分钟: global_active_power (kW) → 转换为 Wh/min → kW * 1000 / 60
    # 每天: 对每分钟求和
    minutely_sub_remainder = (
        df["global_active_power"] * 1000 / 60
        - df["sub_metering_1"]
        - df["sub_metering_2"]
        - df["sub_metering_3"]
    )
    daily["sub_metering_remainder"] = minutely_sub_remainder.resample("D").sum()

    daily = daily.dropna()
    print(f"  日级数据形状: {daily.shape}, 日期范围: {daily.index[0]} ~ {daily.index[-1]}")
    return daily


def add_weather_features(
    daily: pd.DataFrame, weather_path: Optional[str] = None
) -> pd.DataFrame:
    """融合天气数据 + 时间周期性特征

    Args:
        daily: 日级电力数据
        weather_path: 月度天气数据 CSV 路径

    Returns:
        融合后的 DataFrame
    """
    # 始终添加 sin/cos 时间特征
    daily["day_of_year"] = daily.index.dayofyear
    daily["month"] = daily.index.month
    daily["day_of_week"] = daily.index.dayofweek
    daily["sin_doy"] = np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["cos_doy"] = np.cos(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["sin_dow"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["cos_dow"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    daily["sin_month"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["cos_month"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily = daily.drop(columns=["day_of_year", "month", "day_of_week"])

    # 加载月度天气数据并广播到每天
    if weather_path is not None and Path(weather_path).exists():
        weather = pd.read_csv(weather_path, parse_dates=["date"])
        weather = weather.set_index("date")
        print(f"  天气数据已加载: {weather.shape}, 列: {list(weather.columns)}")

        # 将月度数据广播到日级: 每天分配所属月份的值
        for col in weather.columns:
            daily[col] = np.nan
            for month_start in weather.index:
                # 该月所有天
                month_end = month_start + pd.offsets.MonthEnd(1)
                mask = (daily.index >= month_start) & (daily.index <= month_end)
                daily.loc[mask, col] = weather.loc[month_start, col]

        # 前向填充缺失的天气数据
        weather_cols = list(weather.columns)
        daily[weather_cols] = daily[weather_cols].ffill().bfill()
        missing_count = daily[weather_cols].isna().sum().sum()
        if missing_count > 0:
            print(f"  天气数据缺失: {missing_count} 个 (已前向+后向填充)")
        print(f"  融合后特征数: {daily.shape[1]}")
    else:
        print("  未找到天气数据，仅使用日期周期性特征")

    return daily


def normalize_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """使用训练集的统计量对全体数据做 Z-score 标准化"""
    stats = {}
    normalized_train = train_df.copy()
    normalized_test = test_df.copy()

    for col in train_df.columns:
        mean = train_df[col].mean()
        std = train_df[col].std()
        if std < 1e-8:  # 避免除零
            std = 1.0
        stats[col] = {"mean": mean, "std": std}
        normalized_train[col] = (train_df[col] - mean) / std
        normalized_test[col] = (test_df[col] - mean) / std

    return normalized_train, normalized_test, stats


def create_sliding_windows(
    data: pd.DataFrame,
    input_window: int,
    output_window: int,
    target_col: str = "global_active_power",
) -> Tuple[np.ndarray, np.ndarray]:
    """构建滑动窗口

    Args:
        data: shape (T, D) 标准化后的 DataFrame
        input_window: 输入长度
        output_window: 输出长度
        target_col: 目标列名

    Returns:
        X: shape (N, input_window, D) 输入序列
        y: shape (N, output_window) 目标序列
    """
    values = data.values  # (T, D)
    X, y = [], []

    for i in range(len(data) - input_window - output_window + 1):
        X.append(values[i : i + input_window])  # 所有特征
        # 目标: 未来 output_window 天的有功功率
        y_seq = data[target_col].iloc[
            i + input_window : i + input_window + output_window
        ]
        y.append(y_seq.values)

    return np.array(X), np.array(y)


def build_dataset(
    data_path: str,
    output_window: int,
    weather_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, list]:
    """完整的数据预处理流水线

    Returns:
        X_train, y_train, X_test, y_test, normalization_stats, feature_names
    """
    print("=" * 50)
    print(f"构建数据集: output_window={output_window} 天")
    print("=" * 50)

    # 1. 加载原始数据
    print("[1/5] 加载原始数据...")
    df = load_raw_data(data_path)

    # 2. 处理缺失值
    print("[2/5] 处理缺失值...")
    df = handle_missing(df)

    # 3. 按天汇总
    print("[3/5] 按天汇总...")
    daily = aggregate_daily(df)
    feature_names = list(daily.columns)

    # 4. 添加天气/日期特征
    print("[4/5] 添加特征...")
    daily = add_weather_features(daily, weather_path)
    feature_names = list(daily.columns)

    # 5. 划分训练/测试集
    # 对于长期预测, 自动调整比例以保证测试集至少有足够天数构造样本
    min_test_samples = 5
    sample_span = INPUT_WINDOW + output_window  # 每个样本需要的天数
    total_days = len(daily)
    min_test_days = min_test_samples + sample_span - 1

    if total_days - int(total_days * TRAIN_RATIO) < min_test_days:
        # 保证测试集至少有 min_test_samples 个样本
        effective_ratio = (total_days - min_test_days) / total_days
        split_idx = int(total_days * effective_ratio)
        print(f"  TRAIN_RATIO 自动调整: {TRAIN_RATIO} → {effective_ratio:.2f} "
              f"(确保测试集 ≥{min_test_samples}个样本)")
    else:
        split_idx = int(total_days * TRAIN_RATIO)

    train_df = daily.iloc[:split_idx]
    test_df = daily.iloc[split_idx:]
    print(f"  训练集: {train_df.shape}, 测试集: {test_df.shape}")

    # 6. 标准化
    train_df, test_df, stats = normalize_data(train_df, test_df)

    # 7. 构建滑动窗口
    print("[5/5] 构建滑动窗口...")
    X_train, y_train = create_sliding_windows(train_df, INPUT_WINDOW, output_window)
    X_test, y_test = create_sliding_windows(test_df, INPUT_WINDOW, output_window)

    print(f"  训练样本: {X_train.shape}, 测试样本: {X_test.shape}")
    return X_train, y_train, X_test, y_test, stats, feature_names
