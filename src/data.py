import pandas as pd


def load_dataset(csv_path):
    return pd.read_csv(csv_path)


def clean_dataset(df):
    df_clean = df.drop_duplicates()
    df_clean["transfer_steps"] = pd.to_numeric(df_clean["transfer_steps"])
    df_clean = df_clean.dropna()

    valid_connection = df_clean["connection_quality"].between(0, 1)
    df_clean = df_clean[valid_connection]

    valid_size = (df_clean["message_size_kb"] > 0) & (df_clean["transfer_steps"] > 0)
    df_clean = df_clean[valid_size]

    return df_clean.reset_index(drop=True)


def train_val_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train, y_train = X.iloc[:split_index], y.iloc[:split_index]
    X_val, y_val = X.iloc[split_index:], y.iloc[split_index:]
    return X_train, X_val, y_train, y_val
