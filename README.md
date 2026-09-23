# Customer Churn ML — Entrega 1 (Primer Parcial)

Laboratorio de Minería de Datos · ISTEA · 2° cuatrimestre 2026 · Prof. Diego Mosquera

Proyecto que pasa de *dataset + notebook* a un **proyecto Python reproducible, versionado y trazable**:
`dataset → proyecto Python → Git → DVC → MLflow → Model Registry`.

Predice la probabilidad de abandono (`Churn`) de clientes de una telco. Clasificación binaria supervisada.

## Estructura

```
├── data/{raw,production,scoring}   # CSV versionados con DVC (solo los .dvc están en Git)
├── data/processed/                 # train.csv / test.csv generados por `dvc repro`
├── notebooks/01_eda.ipynb          # EDA (ejecutado, con salidas)
├── src/data/split.py               # partición estratificada reproducible (seed en params.yaml)
├── src/features/preprocessing.py   # ColumnTransformer: imputación + escalado + One-Hot
├── src/training/train.py           # entrena/compara modelos, loguea en MLflow y registra el candidato
├── src/evaluation/metrics.py       # métricas, threshold, gráficos
├── tests/                          # pytest (split, features, pipeline)
├── params.yaml                     # TODA la configuración (seed, CV, modelos, umbrales)
├── dvc.yaml / dvc.lock             # pipeline de datos
├── reports/                        # comparación de modelos y figuras del EDA
├── docs/informe_entrega1.md        # resultados y justificación de la selección
└── scripts/setup_remotes.sh        # configura DagsHub (DVC remote)
```
(`app/`, `monitoring/`, `models/`, `Dockerfile`, `compose.yaml` y `.github/` quedan para las entregas 2 y 3.)

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

## Decisiones de diseño

- **Partición**: 80/20 estratificada por `Churn`, `random_state=42`. El test se evalúa una sola vez por modelo y **no interviene en la selección**.
- **Preprocesamiento dentro del `Pipeline`** (mismo código en train e inferencia): `TotalCharges` nulo → 0 (todos los nulos son `tenure = 0`, ver EDA); escalado solo en modelos lineales; `OneHotEncoder(handle_unknown="ignore")`; `customerID` excluido.
- **Modelos**: baseline `DummyClassifier`, `LogisticRegression` (4 configs), `RandomForest` (3), `GradientBoosting` (1) → 9 runs.
- **Selección**: mayor ROC-AUC en CV de 5 folds sobre train (desempate por F2 en CV).
- **Threshold**: se ajusta con las predicciones out-of-fold maximizando F1.5 (favorece recall; ver informe).
- **Trazabilidad**: cada run guarda `git_commit`, `data_md5` (hash DVC del dataset), `params.yaml`, parámetros, métricas, gráficos y modelo. El Model Registry (`customer-churn-classifier`, alias `candidate`) guarda `source_run_id`, `data_md5` y `git_commit` como tags.

## Resultados y modelo candidato

Ver [docs/informe_entrega1.md](docs/informe_entrega1.md).
