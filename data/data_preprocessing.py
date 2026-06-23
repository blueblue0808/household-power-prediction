import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


def process_and_merge_data(power_file, weather_file):

    print("1. 读取分钟级用电数据")

    power_df = pd.read_csv(
        power_file,
        sep=';',
        na_values=['?'],
        low_memory=False
    )

    power_df["Datetime"] = pd.to_datetime(
        power_df["Date"] + " " + power_df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    power_df = power_df.set_index("Datetime")
    power_df = power_df.drop(["Date", "Time"], axis=1)
    power_df = power_df.apply(pd.to_numeric, errors="coerce")

    print("2. 分钟级 -> 日级聚合")

    sum_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
    mean_cols = [
        "Voltage",
        "Global_intensity"
    ]

    daily_power = pd.DataFrame()
    daily_power[sum_cols] = power_df[sum_cols].resample("D").sum()
    daily_power[mean_cols] = power_df[mean_cols].resample("D").mean()

    # 剔除首尾不完整数据
    daily_power = daily_power.loc["2006-12-17":"2010-11-25"]

    daily_power["AAAAMM"] = daily_power.index.strftime("%Y%m").astype(int)

    print("3. 读取气象数据")

    weather_df = pd.read_csv(
        weather_file,
        sep=";",
        low_memory=False
    )

    # 只保留指定站点
    weather_df = weather_df[weather_df["NUM_POSTE"] == 92048001].copy()

    weather_cols = [
        "AAAAMM",
        "RR",
        "NBJRR1",
        "NBJRR5",
        "NBJRR10",
        "NBJBROU"
    ]
    weather_df = weather_df[weather_cols]

    # RR 单位转换（十分之一毫米 -> 毫米）
    weather_df["RR"] = pd.to_numeric(weather_df["RR"], errors="coerce") / 10.0

    for col in weather_cols[1:]:
        weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

    weather_df = (
        weather_df
        .drop_duplicates(subset=["AAAAMM"])
        .sort_values("AAAAMM")
    )

    print("4. 数据融合")

    merged_df = pd.merge(
        daily_power.reset_index(),
        weather_df,
        on="AAAAMM",
        how="left"
    )
    merged_df = merged_df.set_index("Datetime")
    merged_df = merged_df.drop("AAAAMM", axis=1)

    print("5. 构造剩余能耗特征")

    merged_df["Sub_metering_remainder"] = (
        merged_df["Global_active_power"] * 1000 / 60
        - merged_df["Sub_metering_1"]
        - merged_df["Sub_metering_2"]
        - merged_df["Sub_metering_3"]
    )

    print("6. 缺失值处理")

    merged_df = merged_df.ffill().bfill()

    print(f"最终天数: {len(merged_df)}")

    return merged_df


def create_window_np_data(feature_array, target_array, input_len, pred_len):

    X_list = []
    Y_list = []
    total_len = len(feature_array)

    for i in range(total_len - input_len - pred_len + 1):
        X = feature_array[i:i + input_len]
        Y = target_array[i + input_len:i + input_len + pred_len]

        X_list.append(X)
        Y_list.append(Y)

    return np.array(X_list), np.array(Y_list)


def main():

    POWER_FILE = "household_power_consumption.txt"
    WEATHER_FILE = "MENSQ_92_previous-1950-2024.csv"
    SAVE_DIR = "saved_dataset"

    os.makedirs(SAVE_DIR, exist_ok=True)

    df = process_and_merge_data(POWER_FILE, WEATHER_FILE)

    target_name = "Global_active_power"

    print("\n总天数:", len(df))
    print("总特征数:", len(df.columns))

    # 60% / 40% 时间顺序划分

    split_point = int(len(df) * 0.6)
    train_df = df.iloc[:split_point].copy()
    test_df = df.iloc[split_point:].copy()

    print("训练集天数:", len(train_df))
    print("测试集天数:", len(test_df))

    df.to_csv(os.path.join(SAVE_DIR, "processed_daily_data.csv"))
    train_df.to_csv(os.path.join(SAVE_DIR, "train.csv"))
    test_df.to_csv(os.path.join(SAVE_DIR, "test.csv"))

    # 唯一 Scaler（只拟合训练集）

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    scaler_X.fit(train_df.values)
    scaler_Y.fit(train_df[[target_name]].values)

    train_X_scaled = scaler_X.transform(train_df.values)
    train_Y_scaled = scaler_Y.transform(train_df[[target_name]].values).flatten()

    test_X_scaled = scaler_X.transform(test_df.values)
    test_Y_scaled = scaler_Y.transform(test_df[[target_name]].values).flatten()

    # 保存 Scaler 参数
    np.savez_compressed(
        os.path.join(SAVE_DIR, "scaler_info.npz"),
        X_mean=scaler_X.mean_,
        X_std=scaler_X.scale_,
        Y_mean=scaler_Y.mean_,
        Y_std=scaler_Y.scale_
    )

    INPUT_LEN = 90
    SHORT_PRED = 90
    LONG_PRED = 365


    X_train_short, Y_train_short = create_window_np_data(
        train_X_scaled,
        train_Y_scaled,
        INPUT_LEN,
        SHORT_PRED
    )
    X_test_short, Y_test_short = create_window_np_data(
        test_X_scaled,
        test_Y_scaled,
        INPUT_LEN,
        SHORT_PRED
    )

    np.savez_compressed(
        os.path.join(SAVE_DIR, "dataset_short.npz"),
        X_train=X_train_short,
        Y_train=Y_train_short,
        X_test=X_test_short,
        Y_test=Y_test_short
    )

    X_train_long, Y_train_long = create_window_np_data(
        train_X_scaled,
        train_Y_scaled,
        INPUT_LEN,
        LONG_PRED
    )
    X_test_long, Y_test_long = create_window_np_data(
        test_X_scaled,
        test_Y_scaled,
        INPUT_LEN,
        LONG_PRED
    )

    np.savez_compressed(
        os.path.join(SAVE_DIR, "dataset_long.npz"),
        X_train=X_train_long,
        Y_train=Y_train_long,
        X_test=X_test_long,
        Y_test=Y_test_long
    )

    print("\n===== 数据集构建完成 =====")
    print(f"短期训练集 X:{X_train_short.shape}, Y:{Y_train_short.shape}")
    print(f"短期测试集 X:{X_test_short.shape}, Y:{Y_test_short.shape}")
    print(f"长期训练集 X:{X_train_long.shape}, Y:{Y_train_long.shape}")
    print(f"长期测试集 X:{X_test_long.shape}, Y:{Y_test_long.shape}")


if __name__ == "__main__":
    main()