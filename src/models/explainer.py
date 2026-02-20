"""
SHAP-based model explainability for Responsible AI (EU AI Act alignment).
"""

import logging
from pathlib import Path

import numpy as np
import shap

from src.data.preprocessor import ALL_FEATURES

logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parents[2] / "reports" / "shap"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def get_explainer(model, X_background: np.ndarray) -> shap.Explainer:
    """
    Build a SHAP TreeExplainer for tree-based models (RF, XGBoost).
    Falls back to KernelExplainer for other model types.
    """
    clf = model.named_steps.get("clf", model)

    try:
        explainer = shap.TreeExplainer(clf)
        logger.info("Using TreeExplainer")
    except Exception:
        explainer = shap.KernelExplainer(
            lambda x: clf.predict_proba(x)[:, 1],
            shap.sample(X_background, 100),
        )
        logger.info("Using KernelExplainer (slower, model-agnostic)")

    return explainer


def explain_prediction(
    model,
    X_background: np.ndarray,
    X_instance: np.ndarray,
) -> dict:
    """
    Return SHAP values for a single prediction as a feature → contribution dict.

    Args:
        model: Trained sklearn/imblearn pipeline
        X_background: Background dataset (training set) for SHAP
        X_instance: Single row (1, n_features) to explain

    Returns:
        dict mapping feature name → SHAP value
    """
    explainer = get_explainer(model, X_background)
    shap_values = explainer.shap_values(X_instance)

    # For classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    explanation = {feat: round(float(val), 6) for feat, val in zip(ALL_FEATURES, values)}
    top_features = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    logger.info("Top SHAP contributors: %s", top_features)

    return {
        "shap_values": explanation,
        "top_contributors": [{"feature": f, "shap_value": v} for f, v in top_features],
        "base_value": float(explainer.expected_value[1]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value),
    }
