"""Carga del modelo exportado y lógica de predicción (la usan la API y los tests).

El pipeline completo (imputación + encoding + escalado + modelo) viaja en un único archivo
joblib, por lo que inferencia aplica exactamente las mismas transformaciones que el entrenamiento.
"""
import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.preprocessing import FEATURES, NUMERIC

MODEL_FILE = "churn_pipeline.joblib"
META_FILE = "model_meta.json"


def risk_level(probability: float, low_max: float, high_min: float) -> str:
    """LOW: p < low_max · MEDIUM: low_max <= p < high_min · HIGH: p >= high_min."""
    if probability < low_max:
        return "LOW"
    if probability < high_min:
        return "MEDIUM"
    return "HIGH"


class ChurnPredictor:
    def __init__(self, pipeline, meta: dict):
        self.pipeline = pipeline
        self.meta = meta
        self.threshold = float(meta["threshold"])
        self.low_max = float(meta["risk_levels"]["low_max"])
        self.high_min = float(meta["risk_levels"]["high_min"])

    @property
    def model_version(self) -> str:
        return str(self.meta["model_version"])

    @classmethod
    def load(cls, model_dir: str | Path) -> "ChurnPredictor":
        model_dir = Path(model_dir)
        with open(model_dir / META_FILE, encoding="utf-8") as f:
            meta = json.load(f)
        return cls(joblib.load(model_dir / MODEL_FILE), meta)

    def predict(self, features: dict) -> dict:
        """features: dict con las columnas de FEATURES (TotalCharges puede ser None)."""
        row = pd.DataFrame([features])[FEATURES]
        row[NUMERIC] = row[NUMERIC].astype(float)  # None -> NaN, lo imputa el pipeline
        probability = float(self.pipeline.predict_proba(row)[0, 1])
        return {
            "prediction": "CHURN" if probability >= self.threshold else "NO_CHURN",
            "probability": round(probability, 4),
            "risk_level": risk_level(probability, self.low_max, self.high_min),
            "model_version": self.model_version,
        }
