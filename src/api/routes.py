"""
API route handlers.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.auth import get_api_key
from src.api.model_store import ModelStore
from src.api.schemas import (
    ExplainResponse,
    HealthResponse,
    MachineReading,
    PredictionResponse,
)
from src.data.preprocessor import preprocess_single

logger = logging.getLogger(__name__)
router = APIRouter()

APP_VERSION = "1.0.0"


def _risk_level(prob: float, threshold: float) -> str:
    if prob < threshold * 0.5:
        return "LOW"
    elif prob < threshold:
        return "MEDIUM"
    return "HIGH"


# ---------------------------------------------------------------------------
# Health check — no auth required
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    store = ModelStore.get()
    return HealthResponse(
        status="ok" if store.is_ready else "degraded",
        model_loaded=store.is_ready,
        model_name=store.model_name,
        version=APP_VERSION,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    reading: MachineReading,
    _: str = Depends(get_api_key),
):
    """
    Predict machine failure probability from sensor readings.

    Returns failure probability, risk level, and the decision threshold used.
    """
    store = ModelStore.get()
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run train.py first.",
        )

    X = preprocess_single(reading.to_raw_dict(), store.scaler)
    prob = store.predict_proba(X)
    failure = prob >= store.threshold

    logger.info(
        "Prediction | prob=%.4f | threshold=%.3f | failure=%s",
        prob, store.threshold, failure,
    )

    return PredictionResponse(
        failure_predicted=failure,
        failure_probability=round(prob, 4),
        risk_level=_risk_level(prob, store.threshold),
        threshold_used=store.threshold,
        model_version=store.model_name or "unknown",
    )


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain(
    reading: MachineReading,
    _: str = Depends(get_api_key),
):
    """
    Return prediction + SHAP feature contributions for a single reading.

    Use this endpoint for audit trails and EU AI Act compliance.
    """
    store = ModelStore.get()
    if not store.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run train.py first.",
        )

    from src.models.explainer import explain_prediction

    X = preprocess_single(reading.to_raw_dict(), store.scaler)
    prob = store.predict_proba(X)
    failure = prob >= store.threshold

    # Use a small background (model_store can cache this in production)
    explanation = explain_prediction(
        model=store.model,
        X_background=X,   # In production: pass actual training X sample
        X_instance=X,
    )

    return ExplainResponse(
        failure_predicted=failure,
        failure_probability=round(prob, 4),
        risk_level=_risk_level(prob, store.threshold),
        threshold_used=store.threshold,
        model_version=store.model_name or "unknown",
        explanation=explanation,
    )
