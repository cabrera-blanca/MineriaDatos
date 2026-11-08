"""Prueba con el modelo real exportado (se omite si todavía no está en models/)."""
from pathlib import Path

import pandas as pd
import pytest

from src.features.preprocessing import FEATURES
from src.inference.predictor import MODEL_FILE, ChurnPredictor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SCORING = ROOT / "data" / "scoring" / "scoring_batch.csv"

pytestmark = pytest.mark.skipif(not (MODEL_DIR / MODEL_FILE).exists(), reason="modelo real no disponible en models/")


def test_real_model_loads_and_has_lineage():
    predictor = ChurnPredictor.load(MODEL_DIR)
    assert predictor.model_version
    for key in ("source_run_id", "data_md5", "git_commit", "threshold"):
        assert key in predictor.meta


@pytest.mark.skipif(not SCORING.exists(), reason="scoring_batch.csv no disponible (dvc pull)")
def test_real_model_scores_the_scoring_batch():
    predictor = ChurnPredictor.load(MODEL_DIR)
    batch = pd.read_csv(SCORING)
    results = [predictor.predict(row[FEATURES].to_dict()) for _, row in batch.head(20).iterrows()]
    assert all(0 <= r["probability"] <= 1 for r in results)
    assert {r["risk_level"] for r in results} <= {"LOW", "MEDIUM", "HIGH"}
