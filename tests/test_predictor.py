import pytest

from src.inference.predictor import ChurnPredictor, risk_level
from tests.conftest import VALID_PAYLOAD


def test_load_model(model_dir):
    predictor = ChurnPredictor.load(model_dir)
    assert predictor.model_version == "test-1"
    assert predictor.threshold == 0.3


def test_prediction_is_valid(model_dir):
    result = ChurnPredictor.load(model_dir).predict(VALID_PAYLOAD)
    assert set(result) == {"prediction", "probability", "risk_level", "model_version"}
    assert result["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(result["probability"], float) and 0 <= result["probability"] <= 1
    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_missing_total_charges_is_imputed(model_dir):
    result = ChurnPredictor.load(model_dir).predict({**VALID_PAYLOAD, "TotalCharges": None})
    assert 0 <= result["probability"] <= 1


@pytest.mark.parametrize(
    "p, expected",
    [(0.0, "LOW"), (0.299, "LOW"), (0.30, "MEDIUM"), (0.599, "MEDIUM"), (0.60, "HIGH"), (1.0, "HIGH")],
)
def test_risk_level_boundaries(p, expected):
    assert risk_level(p, low_max=0.3, high_min=0.6) == expected


def test_prediction_is_consistent_with_risk(model_dir):
    """CHURN <=> el riesgo no es LOW (el threshold del modelo es el límite inferior de MEDIUM)."""
    predictor = ChurnPredictor.load(model_dir)
    for tenure in (0, 12, 36, 72):
        for contract in ("Month-to-month", "One year", "Two year"):
            r = predictor.predict({**VALID_PAYLOAD, "tenure": tenure, "Contract": contract})
            assert (r["prediction"] == "CHURN") == (r["risk_level"] != "LOW")
