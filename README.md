# Customer Churn ML — Entregas 1 y 2

Laboratorio de Minería de Datos · ISTEA · 2° cuatrimestre 2026 · Prof. Diego Mosquera

Predice la probabilidad de abandono (`Churn`) de clientes de una telco. Clasificación binaria supervisada.

- **Entrega 1** — proyecto Python reproducible, versionado y trazable:
  `dataset → proyecto Python → Git → DVC → MLflow → Model Registry`.
- **Entrega 2** — servicio de inferencia:
  `modelo registrado → capa de inferencia → FastAPI/Pydantic → pytest → Docker → Docker Compose`.

## Estructura

```
├── data/{raw,production,scoring}   # CSV versionados con DVC (solo los .dvc están en Git)
├── data/processed/                 # train.csv / test.csv generados por `dvc repro`
├── notebooks/01_eda.ipynb          # EDA (ejecutado, con salidas)
├── src/data/split.py               # partición estratificada reproducible (seed en params.yaml)
├── src/features/preprocessing.py   # ColumnTransformer: imputación + escalado + One-Hot
├── src/training/train.py           # entrena/compara modelos, loguea en MLflow y registra el candidato
├── src/evaluation/metrics.py       # métricas, threshold, gráficos
├── src/inference/predictor.py      # carga del modelo + predicción + nivel de riesgo (usado por la API)
├── src/inference/export_model.py   # exporta el modelo del Model Registry a models/
├── app/                            # API FastAPI: main.py (endpoints) y schemas.py (contratos Pydantic)
├── models/                         # churn_pipeline.joblib + model_meta.json (versionado con DVC)
├── tests/                          # pytest (split, features, pipeline, predictor, API)
├── Dockerfile / .dockerignore      # imagen de la API
├── compose.yaml                    # `docker compose up`
├── requirements.txt                # entorno de desarrollo completo
├── requirements-api.txt            # solo lo necesario para ejecutar la API (lo usa Docker)
├── params.yaml                     # TODA la configuración (seed, CV, modelos, umbrales)
├── dvc.yaml / dvc.lock             # pipeline de datos
├── reports/                        # comparación de modelos y figuras del EDA
├── docs/                           # informe_entrega1.md e informe_entrega2.md
└── scripts/setup_remotes.sh        # configura DagsHub (DVC remote)
```
(`monitoring/` y `.github/` quedan para la entrega final.)

## Instalación

```bash
git clone <url-del-repo> customer-churn-ml && cd customer-churn-ml
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Recuperar los datos (DVC)

```bash
# Credenciales de DagsHub (una vez; se guardan en .dvc/config.local, fuera de Git)
dvc remote modify --local dagshub user <usuario>
dvc remote modify --local dagshub password <token>
dvc pull
```
Quien crea el remote por primera vez usa `DAGSHUB_USER=... DAGSHUB_TOKEN=... bash scripts/setup_remotes.sh`.
Los CSV **no** están en Git.

## Reproducir el entrenamiento

```bash
dvc repro                       # data/raw -> data/processed/{train,test}.csv (seed y test_size de params.yaml)
python -m src.training.train    # 9 runs en MLflow + registro del candidato
pytest -q                       # tests
```

Por defecto MLflow usa `sqlite:///mlflow.db` local (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
Para loguear en DagsHub: copiar `.env.example` a `.env`, completarlo y exportar esas variables
(`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`). **Nunca commitear credenciales.**

## Entrega 2 — API de inferencia

### 1. Obtener el modelo (`models/`)

El modelo viaja como `models/churn_pipeline.joblib` (pipeline completo: imputación + encoding + escalado + modelo)
y `models/model_meta.json` (versión, run de origen, threshold, niveles de riesgo, `data_md5` y `git_commit`).
Está versionado con DVC, así que después de clonar:

```bash
dvc pull                        # trae datos y models/
```

Para regenerarlo a partir del Model Registry (alias `candidate`), con las variables de DagsHub exportadas:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/MineriaDatos.mlflow
export MLFLOW_TRACKING_USERNAME=<usuario>
export MLFLOW_TRACKING_PASSWORD=<token>
python -m src.inference.export_model
dvc add models && dvc push      # solo si cambia el modelo; luego commitear models.dvc
```

### 2. Ejecutar la API localmente

```bash
uvicorn app.main:app --port 8000
```

- Swagger / OpenAPI: http://localhost:8000/docs
- `GET /health` → `{"status":"ok","model_loaded":true,"model_name":"customer-churn-classifier","model_version":"1"}`
  (responde **503** si el modelo no se pudo cargar).
- `POST /predict` → recibe las 19 características del cliente (sin `customerID`) y responde:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @examples/predict_request_valid.json
```
```json
{"prediction": "CHURN", "probability": 0.7694, "risk_level": "HIGH", "model_version": "1", "request_id": "..."}
```

| Código | Cuándo |
|---|---|
| 200 | Predicción correcta |
| 422 | Entrada inválida (campo faltante o desconocido, tipo incorrecto, valor fuera de dominio). El detalle indica qué campo falló. Probar con `examples/predict_request_invalid.json` |
| 503 | El modelo no está cargado |
| 500 | Error interno: mensaje genérico + `request_id` (el detalle queda solo en el log) |

**Niveles de riesgo** (`params.yaml` + threshold del modelo): `LOW` si p < 0,18 (el modelo no predice churn),
`MEDIUM` si 0,18 ≤ p < 0,50 y `HIGH` si p ≥ 0,50. `prediction` es `CHURN` cuando el riesgo no es `LOW`.
Justificación con datos en [docs/informe_entrega2.md](docs/informe_entrega2.md).

Variables de entorno: `MODEL_DIR` (carpeta del modelo, default `models`) y `LOG_LEVEL` (default `INFO`).

### 3. Tests

```bash
pytest -q
```
Los tests de la API y del predictor usan un modelo mínimo entrenado con datos sintéticos, por lo que no necesitan
`dvc pull` ni acceso a DagsHub. Los tests del modelo real se omiten automáticamente si `models/` no existe.

### 4. Docker y Docker Compose

```bash
dvc pull                        # el build copia models/ dentro de la imagen
docker build -t customer-churn-api:entrega-2 .
docker compose up --build       # API en http://localhost:8000 (cambiar con API_PORT en .env)
docker compose logs -f api      # logs
docker compose down
```

La imagen se construye con `requirements-api.txt` (sin MLflow ni DVC), corre con un usuario sin privilegios y
trae un `HEALTHCHECK` sobre `/health`. No contiene secretos: `.env`, `.dvc/config.local` y los datos están en `.dockerignore`.

### Flujo end-to-end de la entrega

```
git clone → pip install -r requirements.txt → dvc pull → pytest → docker build → docker compose up → POST /predict → respuesta
```

## Decisiones de diseño

- **Partición**: 80/20 estratificada por `Churn`, `random_state=42`. El test se evalúa una sola vez por modelo y **no interviene en la selección**.
- **Preprocesamiento dentro del `Pipeline`** (mismo código en train e inferencia): `TotalCharges` nulo → 0 (todos los nulos son `tenure = 0`, ver EDA); escalado solo en modelos lineales; `OneHotEncoder(handle_unknown="ignore")`; `customerID` excluido.
- **Modelos**: baseline `DummyClassifier`, `LogisticRegression` (4 configs), `RandomForest` (3), `GradientBoosting` (1) → 9 runs.
- **Selección**: mayor ROC-AUC en CV de 5 folds sobre train (desempate por F2 en CV).
- **Threshold**: se ajusta con las predicciones out-of-fold maximizando F1.5 (favorece recall; ver informe).
- **Trazabilidad**: cada run guarda `git_commit`, `data_md5` (hash DVC del dataset), `params.yaml`, parámetros, métricas, gráficos y modelo. El Model Registry (`customer-churn-classifier`, alias `candidate`) guarda `source_run_id`, `data_md5` y `git_commit` como tags.

## Resultados y modelo candidato

Ver [docs/informe_entrega1.md](docs/informe_entrega1.md).
