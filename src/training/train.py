"""Entrenamiento y comparación de modelos con tracking en MLflow y registro del candidato.

Uso:  python -m src.training.train

- Tracking: MLFLOW_TRACKING_URI (DagsHub) o, por defecto, sqlite local.
- La selección del candidato se hace con ROC-AUC de validación cruzada sobre TRAIN.
  El test se evalúa una sola vez por modelo y solo se reporta (no participa en la selección).
"""
import os
import subprocess
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data.split import load_params
from src.evaluation.metrics import best_threshold, classification_metrics, save_plots
from src.features.preprocessing import FEATURES, build_preprocessor

DEFAULT_URI = "sqlite:///mlflow.db"
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")


def make_estimator(mtype: str, params: dict, seed: int):
    if mtype == "dummy":
        return DummyClassifier(**params), False
    if mtype == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed, **params), True
    if mtype == "rf":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params), False
    if mtype == "gb":
        return GradientBoostingClassifier(random_state=seed, **params), False
    raise ValueError(f"Modelo desconocido: {mtype}")


def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(a, stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return "unknown"
    return {"git_commit": run("git", "rev-parse", "HEAD"),
            "git_dirty": str(bool(run("git", "status", "--porcelain")))}


def data_version(raw_path: str) -> str:
    """Hash MD5 del dataset según el archivo .dvc (lineage datos -> run)."""
    dvc_file = raw_path + ".dvc"
    if os.path.exists(dvc_file):
        for line in open(dvc_file, encoding="utf-8"):
            if line.strip().startswith("- md5:") or line.strip().startswith("md5:"):
                return line.split("md5:")[1].strip()
    return "unknown"


def main() -> None:
    cfg = load_params()
    d, t = cfg["data"], cfg["train"]
    seed = d["seed"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_URI))
    mlflow.set_experiment(t["experiment_name"])

    train = pd.read_csv(d["train_path"])
    test = pd.read_csv(d["test_path"])
    X_tr, y_tr = train[FEATURES], (train[d["target"]] == "Yes").astype(int)
    X_te, y_te = test[FEATURES], (test[d["target"]] == "Yes").astype(int)
    cv = StratifiedKFold(n_splits=t["cv_folds"], shuffle=True, random_state=seed)

    ginfo, dver = git_info(), data_version(d["raw_path"])
    results = []

    for m in cfg["models"]:
        est, scale = make_estimator(m["type"], m["params"], seed)
        pipe = Pipeline([("preprocess", build_preprocessor(scale=scale)), ("model", est)])

        with mlflow.start_run(run_name=m["name"]) as run:
            mlflow.set_tags({"model_type": m["type"], "stage": "entrega-1", **ginfo, "data_md5": dver})
            mlflow.log_params({"model_type": m["type"], "seed": seed, "test_size": d["test_size"],
                               "cv_folds": t["cv_folds"], "scaled": scale, "n_features": len(FEATURES),
                               **{f"model__{k}": v for k, v in m["params"].items()}})
            mlflow.log_artifact("params.yaml")

            # CV sobre train: probabilidades out-of-fold -> métricas de validación y threshold
            oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
            thr = 0.5 if m["type"] == "dummy" else best_threshold(y_tr, oof, t["beta_threshold"])
            mlflow.log_metrics(classification_metrics(y_tr, oof, 0.5, "cv_"))
            mlflow.log_metrics(classification_metrics(y_tr, oof, thr, "cv_tuned_"))
            mlflow.log_param("threshold", round(thr, 2))

            # Ajuste final en train completo y evaluación única en test aislado
            pipe.fit(X_tr, y_tr)
            proba_te = pipe.predict_proba(X_te)[:, 1]
            mlflow.log_metrics(classification_metrics(y_te, proba_te, 0.5, "test_"))
            mlflow.log_metrics(classification_metrics(y_te, proba_te, thr, "test_tuned_"))

            with tempfile.TemporaryDirectory() as tmp:
                for p in save_plots(y_te, proba_te, thr, tmp):
                    mlflow.log_artifact(p, "plots")

            sig = infer_signature(X_tr.head(50), pipe.predict_proba(X_tr.head(50)))
            mlflow.sklearn.log_model(pipe, name="model", signature=sig, input_example=X_tr.head(3),
                                     serialization_format="cloudpickle")

            results.append({"run_id": run.info.run_id, "name": m["name"], "type": m["type"], "threshold": thr,
                            "cv_roc_auc": (roc := classification_metrics(y_tr, oof, 0.5)["roc_auc"]),
                            "cv_f2_tuned": classification_metrics(y_tr, oof, thr)["f2"],
                            **{k: v for k, v in classification_metrics(y_te, proba_te, thr, "test_").items()
                               if k in ("test_roc_auc", "test_precision", "test_recall", "test_f1", "test_f2")}})
            print(f"{m['name']:<22} cv_auc={roc:.4f} thr={thr:.2f} test_auc={results[-1]['test_roc_auc']:.4f} "
                  f"test_recall={results[-1]['test_recall']:.3f}")

    res = pd.DataFrame(results).sort_values(["cv_roc_auc", "cv_f2_tuned"], ascending=False)
    os.makedirs("reports", exist_ok=True)
    res.to_csv("reports/model_comparison.csv", index=False)

    # Selección del candidato: mejor ROC-AUC de CV (nunca test)
    best = res.iloc[0]
    print(f"\nCandidato: {best['name']} (run {best['run_id']})")
    model_uri = f"runs:/{best['run_id']}/model"
    mv = mlflow.register_model(model_uri, t["registered_model_name"],
                               tags={"source_run_id": best["run_id"], "threshold": f"{best['threshold']:.2f}",
                                     "data_md5": dver, "git_commit": ginfo["git_commit"]})
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(t["registered_model_name"], "candidate", mv.version)
    client.set_model_version_tag(t["registered_model_name"], mv.version, "selection_criterion",
                                 f"max cv_roc_auc ({t['cv_folds']}-fold CV en train)")
    client.set_tag(best["run_id"], "registered_as", f"{t['registered_model_name']} v{mv.version}")
    print(f"Registrado: {t['registered_model_name']} v{mv.version} alias=candidate <- run {best['run_id']}")


if __name__ == "__main__":
    main()
