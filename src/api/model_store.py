"""
Singleton model store — loads model artifacts once at startup.
"""

import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parents[2] / "models" / "saved"


class ModelStore:
    _instance: Optional["ModelStore"] = None

    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold: float = 0.5
        self.metadata: dict = {}
        self.X_background: Optional[np.ndarray] = None

    @classmethod
    def get(cls) -> "ModelStore":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        model_path = MODEL_DIR / "best_model.joblib"
        scaler_path = MODEL_DIR / "scaler.joblib"
        meta_path = MODEL_DIR / "metadata.joblib"

        if not model_path.exists():
            logger.warning(
                "Model file not found at %s — predictions will be unavailable until training runs.",
                model_path,
            )
            return

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            meta = joblib.load(meta_path)
            self.threshold = meta["threshold"]
            self.metadata = meta.get("metrics", {})
            logger.info(
                "Model loaded: %s | Threshold: %.3f",
                self.metadata.get("model_name", "unknown"),
                self.threshold,
            )
        except Exception as e:
            logger.error("Failed to load model: %s", e)

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    @property
    def model_name(self) -> str | None:
        return self.metadata.get("model_name")

    def predict_proba(self, X: np.ndarray) -> float:
        """Return failure probability for preprocessed feature vector."""
        assert self.is_ready, "Model not loaded"
        return float(self.model.predict_proba(X)[0, 1])
