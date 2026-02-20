"""
Entry point: Train the predictive maintenance model.

Usage:
    python train.py
    python train.py --data data/raw/ai4i2020.csv
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("train")


def main(data_path: str | None = None):
    from src.data.loader import load_raw
    from src.data.preprocessor import preprocess
    from src.models.trainer import train

    logger.info("Loading dataset...")
    df = load_raw(data_path) if data_path else load_raw()

    logger.info("Preprocessing...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    logger.info("Training models...")
    best_model, threshold, metrics = train(X_train, y_train, X_test, y_test, scaler)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Model       : %s", metrics["model_name"])
    logger.info("ROC-AUC     : %.4f", metrics["roc_auc"])
    logger.info("Avg Precision: %.4f", metrics["avg_precision"])
    logger.info("Recall      : %.4f", metrics["recall"])
    logger.info("Threshold   : %.3f", threshold)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train predictive maintenance model")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    args = parser.parse_args()
    main(args.data)
