# helper.py --> preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


 

def load_data(path):
    df = pd.read_csv(path)
    df.replace([-1, "?"], np.nan, inplace=True)
    return df

def convert_watch_time_period(df):
    def map_period(time_str):
        hour = int(time_str.strip().split(":")[0])
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    df["watch_time_period"] = df["Watch Time"].astype(str).apply(map_period)
    return df.drop(columns=["Watch Time"])

def bin_labels(df, config):
    for label, bins in config["label_bins"].items():
        col = label  # e.g. AddictionLevel
        label_col = f"{label}_label"
        if col not in df.columns:
            print(f" Label '{col}' not found in DataFrame columns. Skipping binning.")
            continue

        def assign_bin(value):
            if pd.isna(value):
                return np.nan
            for idx, (bin_name, (low, high)) in enumerate(bins.items()):
                if low <= float(value) < high:
                    return idx
            return len(bins) - 1

        df[label_col] = df[col].apply(assign_bin)

    return df

def encode_categorical(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return pd.get_dummies(df, columns=cat_cols)

def scale_features(df):
    scaler = StandardScaler()

    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

def preprocess_data(path, config=None):
    df = load_data(path)
    df = convert_watch_time_period(df)
    df = bin_labels(df, config)

    target_cols = ["AddictionLevel", "ProductivityLoss"]
    drop_cols = ["UserID", "Video ID", "Video Category", "Location", "Demographics", "Frequency"]


    # Drop unneeded columns
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    print("🧾 Available columns after loading:")
    print(df.columns.tolist())

    # Encode categoricals
    df = encode_categorical(df)

    # Fill missing
    #df.fillna(df.mean(numeric_only=True), inplace=True)

    return df
