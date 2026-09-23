import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, proba, threshold: float, prefix: str = "") -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    m = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "f2": fbeta_score(y_true, pred, beta=2),
        "f1_5": fbeta_score(y_true, pred, beta=1.5),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return {f"{prefix}{k}": float(v) for k, v in m.items()}


def best_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Threshold que maximiza F-beta (beta>1 privilegia recall: un falso negativo es un cliente perdido)."""
    grid = np.arange(0.05, 0.95, 0.01)
    scores = [fbeta_score(y_true, (proba >= t).astype(int), beta=beta) for t in grid]
    return float(grid[int(np.argmax(scores))])


def save_plots(y_true, proba, threshold: float, out_dir: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, (proba >= threshold).astype(int),
                                            display_labels=["No churn", "Churn"], ax=ax, colorbar=False)
    ax.set_title(f"Matriz de confusión (test, thr={threshold:.2f})")
    cm_path = f"{out_dir}/confusion_matrix.png"
    fig.tight_layout(); fig.savefig(cm_path, dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("Curva ROC (test)")
    roc_path = f"{out_dir}/roc_curve.png"
    fig.tight_layout(); fig.savefig(roc_path, dpi=120); plt.close(fig)
    return [cm_path, roc_path]
