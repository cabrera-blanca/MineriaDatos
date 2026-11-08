"""API de predicción de churn.

Variables de entorno:
  MODEL_DIR   carpeta con churn_pipeline.joblib y model_meta.json (default: models)
  LOG_LEVEL   nivel de logging (default: INFO)
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import CustomerFeatures, HealthResponse, PredictionResponse
from src.inference.predictor import ChurnPredictor

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("churn-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", "models")
    try:
        app.state.predictor = ChurnPredictor.load(model_dir)
        logger.info("Modelo cargado: %s v%s desde %s", app.state.predictor.meta["model_name"],
                    app.state.predictor.model_version, model_dir)
    except Exception:  # el servicio arranca igual, pero /health y /predict responden 503
        logger.exception("No se pudo cargar el modelo desde %s", model_dir)
        app.state.predictor = None
    yield


app = FastAPI(
    title="Customer Churn API",
    version="0.2.0",
    description="Estima la probabilidad de abandono de un cliente de telecomunicaciones y devuelve un nivel de riesgo.",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.exception("Error no controlado request_id=%s path=%s", request_id, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Error interno del servidor", "request_id": request_id})


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request):
    predictor: ChurnPredictor | None = request.app.state.predictor
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "model_loaded": False})
    return HealthResponse(status="ok", model_loaded=True, model_name=predictor.meta["model_name"],
                          model_version=predictor.model_version)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures, request: Request):
    predictor: ChurnPredictor | None = request.app.state.predictor
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = predictor.predict(customer.model_dump())
    logger.info("predict request_id=%s prediction=%s probability=%.4f risk=%s model_version=%s ms=%.1f",
                request_id, result["prediction"], result["probability"], result["risk_level"],
                result["model_version"], (time.perf_counter() - start) * 1000)
    return PredictionResponse(**result, request_id=request_id)
