import argparse
import os

from .data import clean_dataset, load_dataset, train_val_split
from .decision_tree import DecisionTreeClassifierScratch
from .metrics import evaluate_binary

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "daneshchat_logs.csv")


def run(data_path=DEFAULT_DATA_PATH, max_depth=5, min_samples_split=20, min_samples_leaf=10):
    df = load_dataset(data_path)
    df = clean_dataset(df)

    y = df["is_spam"]
    X = df.drop(columns=["is_spam", "is_lost"])

    X_train, X_val, y_train, y_val = train_val_split(X, y)

    model = DecisionTreeClassifierScratch(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    return evaluate_binary(y_val, predictions, "DecisionTreeClassifierScratch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a from-scratch decision tree spam classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to the CSV dataset.")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-samples-split", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    args = parser.parse_args()

    run(
        data_path=args.data,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )
