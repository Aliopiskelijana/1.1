"""
Feature engineering and preprocessing for the AI4I 2020 dataset.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)

# Numeric features (will be scaled)
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    # Engineered features
    "temp_diff",
    "power",
    "wear_rate",
]

CATEGORICAL_FEATURES = ["Type_encoded"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-knowledge features for predictive maintenance.

    Features added:
    - temp_diff: Process - Air temperature (heat buildup indicator)
    - power: Rotational speed × Torque (mechanical power)
    - wear_rate: Tool wear / Rotational speed (wear per revolution proxy)
    """
    df = df.copy()

    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["power"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"]
    df["wear_rate"] = df["Tool wear [min]"] / (df["Rotational speed [rpm]"] + 1e-6)

    # Encode machine type: L=0, M=1, H=2
    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type_encoded"] = df["Type"].map(type_map).fillna(0).astype(int)

    logger.debug("Engineered features added: temp_diff, power, wear_rate, Type_encoded")
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Full preprocessing pipeline:
    1. Feature engineering
    2. Train/test split (stratified)
    3. Standard scaling on numeric features

    Returns:
        X_train, X_test, y_train, y_test, fitted_scaler
    """
    df = engineer_features(df)

    X = df[ALL_FEATURES].values
    y = df["Machine failure"].values

    failure_rate = y.mean() * 100
    logger.info("Dataset shape: %s | Failure rate: %.2f%%", X.shape, failure_rate)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Scale numeric columns only (last column is categorical)
    n_numeric = len(NUMERIC_FEATURES)

    if fit_scaler:
        scaler = StandardScaler()
        X_train[:, :n_numeric] = scaler.fit_transform(X_train[:, :n_numeric])
    else:
        assert scaler is not None, "Must provide a fitted scaler when fit_scaler=False"

    X_test[:, :n_numeric] = scaler.transform(X_test[:, :n_numeric])

    logger.info(
        "Train size: %d | Test size: %d | Positive in train: %d (%.2f%%)",
        len(X_train), len(X_test),
        y_train.sum(), y_train.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, scaler


def preprocess_single(
    record: dict,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Preprocess a single inference record into a feature vector.

    Args:
        record: Dict with raw feature keys matching AI4I schema
        scaler: Fitted StandardScaler from training

    Returns:
        2-D numpy array of shape (1, n_features)
    """
    df = pd.DataFrame([record])
    df = engineer_features(df)
    X = df[ALL_FEATURES].values.astype(float)

    n_numeric = len(NUMERIC_FEATURES)
    X[:, :n_numeric] = scaler.transform(X[:, :n_numeric])
    return X
