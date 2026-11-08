# Informe — Entrega 2

## 1. Resumen
El modelo candidato de la Entrega 1 (`customer-churn-classifier`, alias `candidate`) se convirtió en un servicio de
inferencia: una API REST con FastAPI y contratos Pydantic, una suite de tests con pytest y un entorno reproducible con
Docker y Docker Compose.

```
Model Registry → export_model → models/ → ChurnPredictor → FastAPI (/health, /predict) → pytest → Docker → Compose
```

## 2. Integración del modelo
- `src/inference/export_model.py` toma del Model Registry la versión con alias `candidate` y genera dos archivos en `models/`:
  - `churn_pipeline.joblib`: el **pipeline completo** (imputación, One-Hot, escalado y regresión logística). La API no
    reimplementa ninguna transformación, por lo que inferencia aplica exactamente las mismas que el entrenamiento.
  - `model_meta.json`: versión del modelo, `source_run_id`, threshold, niveles de riesgo, `data_md5` y `git_commit`.
- `GET /health` y cada respuesta de `/predict` informan la `model_version` cargada. Con el `source_run_id` de la ficha se
  llega al run de MLflow, al commit de Git y al hash del dataset en DVC: la trazabilidad de la Entrega 1 se mantiene.
- La imagen usa `requirements-api.txt`, sin MLflow ni DVC: solo lo necesario para ejecutar.
- `models/` está versionado con DVC (no en Git).

## 3. Diseño de la API

| Endpoint | Función |
|---|---|
| `GET /health` | Estado del servicio y versión del modelo cargado. **503** si el modelo no se pudo cargar. |
| `POST /predict` | Valida el contrato, ejecuta el pipeline y devuelve `prediction`, `probability`, `risk_level`, `model_version` y `request_id`. |
| `GET /docs`, `/openapi.json` | Documentación Swagger/OpenAPI generada por FastAPI. |

**Validación con Pydantic** (`app/schemas.py`): categorías como `Literal` (por ejemplo `Contract` solo acepta
`Month-to-month`, `One year` o `Two year`), `tenure` entero entre 0 y 72, `MonthlyCharges` numérico entre 0 y 500,
`TotalCharges` opcional (el pipeline lo imputa con 0, como en el entrenamiento) y `extra="forbid"` para rechazar campos
desconocidos en lugar de ignorarlos. `customerID` no forma parte del contrato porque no es un predictor.

**Manejo de errores:** entrada inválida → **422** con el campo que falló; modelo no cargado → **503**; error inesperado →
**500** con mensaje genérico y `request_id` (el detalle queda solo en el log, no se expone al cliente).

**Logs:** una línea por predicción con `request_id`, resultado, probabilidad, nivel de riesgo, versión del modelo y
latencia. El nivel se configura con `LOG_LEVEL`.

## 4. Niveles de riesgo (LOW / MEDIUM / HIGH)
- **LOW**: `p < 0,18`. Es el threshold del modelo candidato (ajustado en la Entrega 1 para priorizar recall): por debajo, el
  modelo no predice churn.
- **MEDIUM**: `0,18 ≤ p < 0,50`. Zona de incertidumbre.
- **HIGH**: `p ≥ 0,50`.
- `prediction = CHURN` si y solo si el riesgo no es `LOW`, de modo que ambos campos nunca se contradicen (hay un test para esto).

**Justificación con datos.** Se calculó la tasa real de abandono dentro de cada nivel, usando probabilidades de validación
cruzada sobre **train** (el test no se usa para decidir) y, solo como reporte, sobre el test aislado:

| Nivel | Clientes (train, CV) | Abandono real (train, CV) | Clientes (test) | Abandono real (test) |
|---|---|---|---|---|
| LOW | 2.733 | 7,4 % | 695 | 8,6 % |
| MEDIUM | 1.868 | 30,9 % | 462 | 31,8 % |
| HIGH | 1.033 | 68,2 % | 252 | 65,5 % |

Los niveles ordenan bien el riesgo y el patrón se mantiene en el test. Para elegir el límite de HIGH se compararon varios
valores en train (validación cruzada):

| Si HIGH empieza en | Clientes en HIGH | Abandono real en HIGH | % de todos los que abandonan |
|---|---|---|---|
| 0,40 | 1.491 | 60 % | 60 % |
| **0,50** | **1.033** | **68 %** | **47 %** |
| 0,60 | 664 | 73 % | 33 % |

Con 0,50 la alerta HIGH acierta unas 2 de cada 3 veces y concentra casi la mitad de los abandonos. Subirlo a 0,60 mejora
poco la precisión pero deja fuera a muchos clientes que sí se van, algo que el caso de negocio penaliza (un falso negativo
es un cliente perdido). El valor se parametriza en `params.yaml` (`risk_levels.high_min`) y debería validarse con el área
comercial según el costo de cada acción de retención.

## 5. Testing
`pytest -q` ejecuta 40 tests (de los cuales 3 son de la Entrega 1):

| Archivo | Qué verifica |
|---|---|
| `tests/test_predictor.py` | Carga del modelo, predicción válida, `TotalCharges` ausente, límites exactos de los niveles de riesgo, coherencia entre `prediction` y `risk_level`. |
| `tests/test_api.py` | `/health` (200 y 503), `/predict` válido (estructura y tipos de la respuesta, `request_id` UUID único), los ejemplos del profesor, 11 valores inválidos, campo faltante, campo desconocido, cuerpo mal formado, modelo no disponible (503), error interno genérico (500), Swagger/OpenAPI y 404. |
| `tests/test_real_model.py` | Carga del modelo real con su ficha de trazabilidad y predicción sobre `scoring_batch.csv`. Se omite si `models/` no existe. |

Los tests de API y predictor entrenan un modelo mínimo con datos sintéticos (`tests/conftest.py`): no dependen de DVC,
DagsHub ni del modelo real, así que se ejecutan en cualquier máquina.

## 6. Docker y Compose
- **Dockerfile**: parte de `python:3.12-slim`, instala `requirements-api.txt` en una capa propia (cache), copia `src/`, `app/` y
  `models/`, corre con un usuario sin privilegios (uid 10001) e incluye un `HEALTHCHECK` sobre `/health`.
- **.dockerignore**: excluye datos, notebooks, tests, historial de Git, `.env` y `.dvc/config.local`. No hay secretos en la imagen.
- **compose.yaml**: un servicio `api` que se construye desde el `Dockerfile`, publica el puerto 8000 (configurable con `API_PORT`),
  recibe `MODEL_DIR` y `LOG_LEVEL` por variables de entorno y se reinicia salvo detención manual.
- Para probar otro modelo sin reconstruir la imagen se puede montar una carpeta como volumen en `/app/models`.

## 7. Evidencia de ejecución
Flujo ejecutado: `export_model` (desde DagsHub) → `dvc add/push` → `pytest` (40 tests OK) → `docker build` → `docker compose up -d` → `POST /predict`.

- Imagen `customer-churn-api:entrega-2`: 659 MB en disco (la mayor parte son scikit-learn, scipy y pandas). Construida con `requirements-api.txt` (24 paquetes, sin MLflow ni DVC).
- Contenedor `churn-api`: estado `healthy`, puerto `8000`, usuario `appuser`.
- `GET /health` → 200 `{"status":"ok","model_loaded":true,"model_name":"customer-churn-classifier","model_version":"1"}`.
- `POST /predict` con `examples/predict_request_valid.json` → 200 `{"prediction":"CHURN","probability":0.7694,"risk_level":"HIGH","model_version":"1",...}`.
- `POST /predict` con `examples/predict_request_invalid.json` → 422 (`tenure`, `Contract`, `MonthlyCharges`).
- Modelo cargado: versión 1, run `f8b894e8...` del Model Registry de DagsHub, commit `672d838` (tag `entrega-1`).
- Logs del contenedor: una línea por predicción con `request_id`, resultado, riesgo, versión y latencia.

## 8. Limitaciones y próximos pasos
- Una imagen contiene un modelo: cambiar de versión implica reconstruirla (o montar el volumen). En la entrega final se publicará
  en GHCR con tags de versión.
- No hay autenticación ni límite de uso en la API: queda fuera del alcance académico.
- El monitoreo de drift con `data/production` (Evidently) corresponde a la entrega final.
