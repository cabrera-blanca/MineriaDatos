#!/usr/bin/env bash
# Configura remotes de DagsHub (DVC + MLflow). Requiere haber creado el repo en DagsHub.
# Uso:  DAGSHUB_USER=<usuario> DAGSHUB_TOKEN=<token> bash scripts/setup_remotes.sh
# El token NUNCA se guarda en el repo: dvc lo escribe en .dvc/config.local (ignorado por git).
set -euo pipefail
: "${DAGSHUB_USER:?definir DAGSHUB_USER}" ; : "${DAGSHUB_TOKEN:?definir DAGSHUB_TOKEN}"
REPO="customer-churn-ml"
dvc remote add -f -d dagshub "https://dagshub.com/${DAGSHUB_USER}/${REPO}.dvc"
dvc remote modify dagshub auth basic
dvc remote modify --local dagshub user "${DAGSHUB_USER}"
dvc remote modify --local dagshub password "${DAGSHUB_TOKEN}"
dvc push
echo "MLflow: exportar MLFLOW_TRACKING_URI=https://dagshub.com/${DAGSHUB_USER}/${REPO}.mlflow (ver .env.example)"
