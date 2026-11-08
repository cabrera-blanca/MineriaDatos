import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == "test-1"


def test_health_without_model_returns_503(client_without_model):
    r = client_without_model.get("/health")
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "model_loaded": False}


def test_predict_valid_request(client, payload):
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "probability", "risk_level", "model_version", "request_id"}
    assert body["prediction"] in {"CHURN", "NO_CHURN"}
    assert isinstance(body["probability"], float) and 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert body["model_version"] == "test-1"
    uuid.UUID(body["request_id"])  # es un UUID válido


def test_predict_valid_example_file(client):
    valid = json.loads((EXAMPLES / "predict_request_valid.json").read_text(encoding="utf-8"))
    assert client.post("/predict", json=valid).status_code == 200


def test_predict_invalid_example_is_rejected_with_details(client):
    invalid = json.loads((EXAMPLES / "predict_request_invalid.json").read_text(encoding="utf-8"))
    r = client.post("/predict", json=invalid)
    assert r.status_code == 422
    failed = {e["loc"][-1] for e in r.json()["detail"]}
    assert failed == {"tenure", "Contract", "MonthlyCharges"}


def test_request_ids_are_unique(client, payload):
    ids = {client.post("/predict", json=payload).json()["request_id"] for _ in range(3)}
    assert len(ids) == 3


def test_total_charges_is_optional(client, payload):
    payload.pop("TotalCharges")
    assert client.post("/predict", json=payload).status_code == 200
    payload["TotalCharges"] = None
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("tenure", 73), ("tenure", -1), ("tenure", 12.5), ("tenure", "doce"),
        ("Contract", "Weekly"), ("gender", "Other"), ("SeniorCitizen", 2),
        ("InternetService", "Cable"), ("MonthlyCharges", -5), ("MonthlyCharges", "caro"),
        ("TotalCharges", -1),
    ],
)
def test_predict_rejects_invalid_values(client, payload, field, value):
    payload[field] = value
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert field in {e["loc"][-1] for e in r.json()["detail"]}


def test_predict_rejects_missing_required_field(client, payload):
    payload.pop("Contract")
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_unknown_field(client, payload):
    payload["customerID"] = "1234-ABCD"
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_malformed_body(client):
    bad = client.post("/predict", content="no es json", headers={"Content-Type": "application/json"})
    assert bad.status_code == 422
    assert client.post("/predict", json=[]).status_code == 422


def test_predict_without_model_returns_503(client_without_model, payload):
    assert client_without_model.post("/predict", json=payload).status_code == 503


def test_internal_error_is_generic_and_has_request_id(model_dir, payload, monkeypatch):
    from app.main import app

    def boom(self, features):
        raise RuntimeError("detalle interno secreto")

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setattr("src.inference.predictor.ChurnPredictor.predict", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/predict", json=payload)
    assert r.status_code == 500
    assert "secreto" not in r.text
    uuid.UUID(r.json()["request_id"])


def test_swagger_and_openapi_are_available(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/predict" in spec["paths"] and "/health" in spec["paths"]
    assert "CustomerFeatures" in spec["components"]["schemas"]


def test_unknown_route_returns_404(client):
    assert client.get("/no-existe").status_code == 404
