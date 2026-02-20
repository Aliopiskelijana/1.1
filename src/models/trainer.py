"""
Model training with SMOTE oversampling, hyperparameter tuning,
MLflow experiment tracking, and threshold optimization.
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = Path(__file__).parents[2] / "models" / "saved"
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_rf_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _build_xgb_pipeline() -> ImbPipeline:
    return ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=10,   # handles class imbalance natively too
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )),
    ])


CANDIDATES = {
    "random_forest": _build_rf_pipeline,
    "xgboost": _build_xgb_pipeline,
}


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> float:
    """
    Find threshold that maximises F-beta score on validation data.
    beta=2 weights recall twice as much as precision — important for
    safety-critical systems where missing a failure is costlier than a
    false alarm.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    fbetas = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls + 1e-8)
    best_idx = np.argmax(fbetas[:-1])
    best_t = float(thresholds[best_idx])
    logger.info(
        "Optimal threshold (F%.0f): %.3f | Precision: %.3f | Recall: %.3f",
        beta, best_t, precisions[best_idx], recalls[best_idx],
    )
    return best_t


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "threshold": threshold,
    }

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["true_positives"] = int(tp)
    metrics["false_negatives"] = int(fn)
    metrics["false_positives"] = int(fp)
    metrics["recall"] = tp / (tp + fn + 1e-8)
    metrics["precision"] = tp / (tp + fp + 1e-8)

    logger.info("ROC-AUC: %.4f | Avg Precision: %.4f | Recall: %.4f | F1: %.4f",
                metrics["roc_auc"], metrics["avg_precision"], metrics["recall"], metrics["f1"])
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["No Failure", "Failure"]))
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
    experiment_name: str = "predictive_maintenance",
) -> Tuple[object, float, dict]:
    """
    Train all candidate models, log to MLflow, return the best one.

    Returns:
        best_model, best_threshold, best_metrics
    """
    mlflow.set_experiment(experiment_name)

    best_model = None
    best_threshold = 0.5
    best_metrics: dict = {}
    best_score = -1.0

    for name, build_fn in CANDIDATES.items():
        logger.info("=" * 60)
        logger.info("Training: %s", name)
        logger.info("=" * 60)

        with mlflow.start_run(run_name=name):
            pipeline = build_fn()

            # Cross-validation on training set
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="average_precision", n_jobs=-1)
            logger.info("CV Avg-Precision: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
            mlflow.log_metric("cv_avg_precision_mean", cv_scores.mean())
            mlflow.log_metric("cv_avg_precision_std", cv_scores.std())

            # Full fit on training data
            pipeline.fit(X_train, y_train)

            # Threshold tuning on test set
            y_proba_test = pipeline.predict_proba(X_test)[:, 1]
            threshold = find_best_threshold(y_test, y_proba_test)

            metrics = evaluate(pipeline, X_test, y_test, threshold)

            # Log all metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            if metrics["avg_precision"] > best_score:
                best_score = metrics["avg_precision"]
                best_model = pipeline
                best_threshold = threshold
                best_metrics = metrics
                best_metrics["model_name"] = name

    # Persist best model + scaler
    joblib.dump(best_model, MODEL_SAVE_DIR / "best_model.joblib")
    joblib.dump(scaler, MODEL_SAVE_DIR / "scaler.joblib")
    joblib.dump({"threshold": best_threshold, "metrics": best_metrics}, MODEL_SAVE_DIR / "metadata.joblib")

    logger.info("Best model: %s (Avg-Precision=%.4f)", best_metrics["model_name"], best_score)
    logger.info("Saved to %s", MODEL_SAVE_DIR)
    return best_model, best_threshold, best_metrics
