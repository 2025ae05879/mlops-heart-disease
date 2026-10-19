"""Metric computation and diagnostic plots (pure functions; easy to log to MLflow)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

SCORING = {
    "accuracy": "accuracy",
    # zero_division=0: very strong regularisation can predict a single class in a fold
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def test_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def cv_metrics(cv_results: dict, best_index: int) -> dict[str, float]:
    """Mean and std of every scorer for the selected grid point."""
    out = {}
    for m in SCORING:
        out[f"cv_{m}_mean"] = float(cv_results[f"mean_test_{m}"][best_index])
        out[f"cv_{m}_std"] = float(cv_results[f"std_test_{m}"][best_index])
    return out


def plot_roc_overlay(models: dict, X_test, y_test, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax.set_title("ROC on held-out test set")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_single(model, X_test, y_test, name: str, path: Path) -> Path:
    return plot_roc_overlay({name: model}, X_test, y_test, path)


def plot_confusion(model, X_test, y_test, name: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["no disease", "disease"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"Confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_importance(model, path: Path, top: int = 15) -> Path | None:
    """Permutation-free importance view: coefficients or impurity importances."""
    clf = model.named_steps["clf"]
    names = model.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "coef_"):
        vals = clf.coef_.ravel()
        label = "coefficient (standardised)"
    elif hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        label = "impurity importance"
    else:
        return None
    order = np.argsort(np.abs(vals))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], vals[order],
            color=["#c0392b" if v > 0 else "#2471a3" for v in vals[order]])
    ax.set_xlabel(label)
    ax.set_title("Top features of selected model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
