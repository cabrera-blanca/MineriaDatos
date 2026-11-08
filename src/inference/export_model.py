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

# Si el servidor no soporta una consulta, MLflow reintenta durante minutos y parece colgado.
# Se acotan los reintentos para que cada fuente falle rápido y se pase a la siguiente.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import sklearn  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from src.data.split import load_params  # noqa: E402
from src.inference.predictor import META_FILE, MODEL_FILE  # noqa: E402
from src.training.train import DEFAULT_URI  # noqa: E402


def _download_logged_model(client: MlflowClient, mv, tmp: str):
    """MLflow 3 guarda el modelo logueado en <experimento>/models/<model_id>/artifacts."""
    model_id = mv.source.split("models:/")[-1] if mv.source.startswith("models:/m-") else None
    if model_id is None:
        raise ValueError("la versión no apunta a un logged model (m-...)")
    run_root = client.get_run(mv.run_id).info.artifact_uri  # .../<run_id>/artifacts
    uri = f"{run_root.rsplit('/', 2)[0]}/models/{model_id}/artifacts"
    return mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp)


def load_pipeline(client: MlflowClient, mv):
    """Prueba varias fuentes: algunos servidores (p. ej. DagsHub) no resuelven todos los formatos de URI."""
    sources = {
        f"registro (models:/{mv.name}/{mv.version})": lambda: mlflow.sklearn.load_model(f"models:/{mv.name}/{mv.version}"),
        f"origen de la versión ({mv.source})": lambda: mlflow.sklearn.load_model(mv.source),
        f"run (runs:/{mv.run_id[:8]}.../model)": lambda: mlflow.sklearn.load_model(f"runs:/{mv.run_id}/model"),
    }
    for label, load in sources.items():
        print(f"Intentando cargar desde {label} ...", flush=True)
        try:
            pipe = load()
            print("  OK")
            return pipe
        except Exception as e:  # noqa: BLE001
            print(f"  falló: {type(e).__name__}", flush=True)
    for label, download in {
        "artefactos del logged model": lambda tmp: _download_logged_model(client, mv, tmp),
        "artefactos del run": lambda tmp: client.download_artifacts(mv.run_id, "model", tmp),
    }.items():
        print(f"Intentando bajar {label} ...", flush=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                local = download(tmp)
                pipe = mlflow.sklearn.load_model(local)
            print("  OK")
            return pipe
        except Exception as e:  # noqa: BLE001
            print(f"  falló: {type(e).__name__}: {str(e)[:150]}", flush=True)
    raise SystemExit("No se pudo cargar el modelo desde ninguna fuente. Pegá los mensajes de arriba (sin credenciales).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", default="candidate")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    cfg = load_params()
    name = cfg["train"]["registered_model_name"]
    uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI)
    print(f"Conectando a {uri.split('@')[-1]} ...", flush=True)
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    print(f"Buscando {name} con alias '{args.alias}' ...", flush=True)
    mv = client.get_model_version_by_alias(name, args.alias)
    print(f"Encontrado: versión {mv.version}, run {mv.run_id}", flush=True)
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
