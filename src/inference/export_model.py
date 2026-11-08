"""Exporta el modelo registrado (alias `candidate`) del Model Registry a `models/`.

Uso:  python -m src.inference.export_model [--alias candidate]

Usa MLFLOW_TRACKING_URI / MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (DagsHub) o, por
defecto, el MLflow local. Genera:
  models/churn_pipeline.joblib   pipeline completo de scikit-learn
  models/model_meta.json         versión, run de origen, threshold, niveles de riesgo y lineage
"""
import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import sklearn
from mlflow import MlflowClient

from src.data.split import load_params
from src.inference.predictor import META_FILE, MODEL_FILE
from src.training.train import DEFAULT_URI


def load_pipeline(client: MlflowClient, mv):
    """Prueba varias fuentes: algunos servidores (p. ej. DagsHub) no resuelven todos los formatos de URI."""
    sources = [f"models:/{mv.name}/{mv.version}", mv.source, f"runs:/{mv.run_id}/model"]
    for src in sources:
        try:
            pipe = mlflow.sklearn.load_model(src)
            print(f"Modelo cargado desde: {src}")
            return pipe
        except Exception as e:  # noqa: BLE001
            print(f"No se pudo cargar desde {src}: {type(e).__name__}")
    with tempfile.TemporaryDirectory() as tmp:  # último recurso: bajar los artefactos del run
        local = client.download_artifacts(mv.run_id, "model", tmp)
        print(f"Modelo cargado desde artefactos del run {mv.run_id}")
        return mlflow.sklearn.load_model(local)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="candidate")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    cfg = load_params()
    name = cfg["train"]["registered_model_name"]
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI))
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, args.alias)
    if "threshold" not in mv.tags:
        raise SystemExit(f"La versión {mv.version} de {name} no tiene el tag 'threshold'; reentrenar y registrar.")

    pipe = load_pipeline(client, mv)
    threshold = float(mv.tags["threshold"])
    meta = {
        "model_name": name,
        "model_version": str(mv.version),
        "alias": args.alias,
        "source_run_id": mv.run_id,
        "threshold": threshold,
        # LOW = por debajo del threshold (no se predice churn); HIGH desde high_min (params.yaml)
        "risk_levels": {"low_max": threshold, "high_min": float(cfg["risk_levels"]["high_min"])},
        "data_md5": mv.tags.get("data_md5", "unknown"),
        "git_commit": mv.tags.get("git_commit", "unknown"),
        "sklearn_version": sklearn.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / MODEL_FILE)
    (out / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exportado {name} v{mv.version} (run {mv.run_id}) -> {out}/")


if __name__ == "__main__":
    main()
