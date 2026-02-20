"""Tests for data preprocessing pipeline."""

import numpy as np
import pytest
from src.data.loader import load_raw
from src.data.preprocessor import engineer_features, preprocess, preprocess_single, ALL_FEATURES


@pytest.fixture(scope="module")
def df():
    return load_raw()  # uses synthetic data if real data absent


def test_engineer_features_adds_columns(df):
    result = engineer_features(df)
    assert "temp_diff" in result.columns
    assert "power" in result.columns
    assert "wear_rate" in result.columns
    assert "Type_encoded" in result.columns


def test_preprocess_returns_correct_shapes(df):
    X_train, X_test, y_train, y_test, scaler = preprocess(df, test_size=0.2)
    total = len(df)
    assert len(X_train) + len(X_test) == total
    assert X_train.shape[1] == len(ALL_FEATURES)
    assert X_test.shape[1] == len(ALL_FEATURES)


def test_preprocess_stratified_split(df):
    X_train, X_test, y_train, y_test, scaler = preprocess(df, test_size=0.2)
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    # Failure rates should be close (stratified)
    assert abs(train_rate - test_rate) < 0.02


def test_preprocess_single(df):
    _, _, _, _, scaler = preprocess(df)
    record = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }
    X = preprocess_single(record, scaler)
    assert X.shape == (1, len(ALL_FEATURES))
    assert not np.any(np.isnan(X))
