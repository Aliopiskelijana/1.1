"""
AI4I 2020 Predictive Maintenance Dataset Loader.

Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
"""

import logging
import os
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "ai4i2020.csv"

# AI4I 2020 column schema
FEATURE_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COL = "Machine failure"

FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Downloads a synthetic sample if file not found."""
    path = Path(path)
    if not path.exists():
        logger.warning("Dataset not found at %s — generating synthetic sample.", path)
        return _generate_synthetic_sample()
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def _generate_synthetic_sample(n: int = 10_000) -> pd.DataFrame:
    """
    Generate a synthetic dataset that mirrors AI4I 2020 statistics.
    Useful for CI/CD and local development without the real dataset.
    """
    rng = np.random.default_rng(42)

    machine_types = rng.choice(["L", "M", "H"], size=n, p=[0.60, 0.30, 0.10])
    air_temp = rng.normal(300.0, 2.0, n)
    process_temp = air_temp + rng.normal(10.0, 1.0, n)
    rot_speed = rng.normal(1538, 179, n).astype(int).clip(1168, 2886)
    torque = rng.normal(40.0, 10.0, n).clip(3.8, 76.6)
    tool_wear = rng.integers(0, 254, n)

    # Approximate 3.4% failure rate
    failure = rng.binomial(1, 0.034, n)

    df = pd.DataFrame(
        {
            "UDI": range(1, n + 1),
            "Product ID": [f"{t}{i}" for t, i in zip(machine_types, range(n))],
            "Type": machine_types,
            "Air temperature [K]": air_temp.round(1),
            "Process temperature [K]": process_temp.round(1),
            "Rotational speed [rpm]": rot_speed,
            "Torque [Nm]": torque.round(1),
            "Tool wear [min]": tool_wear,
            "Machine failure": failure,
            "TWF": (tool_wear > 200) & (failure == 1),
            "HDF": (process_temp - air_temp > 11.5) & (failure == 1),
            "PWF": ((rot_speed * torque) < 3500) & (failure == 1),
            "OSF": (tool_wear > 200) & (torque > 50) & (failure == 1),
            "RNF": rng.binomial(1, 0.001, n),
        }
    )
    df[FAILURE_MODE_COLS] = df[FAILURE_MODE_COLS].astype(int)
    logger.info("Generated synthetic dataset with %d rows (failure rate=%.1f%%)", n, failure.mean() * 100)
    return df
