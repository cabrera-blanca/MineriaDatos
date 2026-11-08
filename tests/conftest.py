"""Fixtures: un modelo mínimo entrenado con datos sintéticos, para que los tests no dependan
de DVC, DagsHub ni del modelo real."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import FEATURES, build_preprocessor
from src.inference.predictor import META_FILE, MODEL_FILE

DOMAINS = {
    "gender": ["Female", "Male"], "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"], "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"], "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"], "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"], "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"], "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}

VALID_PAYLOAD = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 32,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
    "MonthlyCharges": 101.35, "TotalCharges": 2865.88,
}


def synthetic_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in DOMAINS.items()})
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    df["TotalCharges"] = (df.tenure * df.MonthlyCharges).round(2)
    df.loc[df.tenure == 0, "TotalCharges"] = np.nan
    logit = 1.5 * (df.Contract == "Month-to-month") - 0.04 * df.tenure - 0.8
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df[FEATURES], y


def save_test_model(directory, version: str = "test-1") -> None:
    X, y = synthetic_dataset()
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]).fit(X, y)
    joblib.dump(pipe, directory / MODEL_FILE)
    meta = {"model_name": "test-model", "model_version": version, "threshold": 0.3,
            "risk_levels": {"low_max": 0.3, "high_min": 0.6}}
    (directory / META_FILE).write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    save_test_model(d)
    return d


@pytest.fixture
def client(model_dir, monkeypatch):
    from app.main import app

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    with TestClient(app) as c:  # el `with` ejecuta el arranque (carga del modelo)
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    from app.main import app

    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # carpeta vacía: no hay modelo
    with TestClient(app) as c:
        yield c


@pytest.fixture
def payload():
    return dict(VALID_PAYLOAD)
