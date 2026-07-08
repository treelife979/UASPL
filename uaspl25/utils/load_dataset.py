from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_dataset(dataset_name):
    file_path = ROOT_DIR / "data" / dataset_name
    if not file_path.exists():
        print(f"Dataset file not found: {file_path}")
        return None, None

    data = pd.read_csv(file_path)
    data = data.replace("?", np.nan).dropna()

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return X, y


def load_data_feature(dataset_name):
    train_path = ROOT_DIR / "data_features" / dataset_name / "train.csv"
    test_path = ROOT_DIR / "data_features" / dataset_name / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print(f"Feature files not found under: {train_path.parent}")
        return None, None

    data_train = pd.read_csv(train_path).replace("?", np.nan).dropna()
    data_test = pd.read_csv(test_path).replace("?", np.nan).dropna()

    X_train = data_train.iloc[:, 1:].values
    y_train = data_train.iloc[:, 0].values
    X_test = data_test.iloc[:, 1:].values
    y_test = data_test.iloc[:, 0].values
    return X_train, y_train, X_test, y_test
