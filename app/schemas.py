"""Contratos de entrada y salida de la API (Pydantic)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    """Características de un cliente. Mismos campos y dominios que el diccionario de datos
    (sin customerID, que es solo un identificador y no se usa como predictor)."""

    model_config = ConfigDict(
        extra="forbid",  # un campo mal escrito se rechaza en lugar de ignorarse en silencio
        json_schema_extra={
            "example": {
                "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 32,
                "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
                "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
                "MonthlyCharges": 101.35, "TotalCharges": 2865.88,
            }
        },
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72, description="Meses de permanencia (0 a 72)")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=500, description="Cargo mensual actual")
    TotalCharges: float | None = Field(
        default=None, ge=0, description="Cargos acumulados. Puede omitirse (el pipeline lo imputa)"
    )


class PredictionResponse(BaseModel):
    prediction: Literal["CHURN", "NO_CHURN"]
    probability: float = Field(ge=0, le=1, description="Probabilidad estimada de abandono")
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
