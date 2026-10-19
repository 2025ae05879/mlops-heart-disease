"""Train, tune and compare classifiers.  Usage:  python -m heart.train [--quick]

Flow
----
1. load cleaned data -> stratified 80/20 split (test set locked away)
2. for each candidate: Pipeline(prep -> clf) tuned by GridSearchCV with
   StratifiedKFold(5, shuffle, seed); refit on ROC-AUC
3. score the refitted model once on the test set
4. pick the final model with a screening-oriented rule (see `select_model`)
5. write reports/model_comparison.{csv,md}, figures, models/model.joblib
   (+ models/mlflow_model) and log every candidate to MLflow (./mlruns)

Every per-model result goes through a `RunTracker`; `heart.tracking.MlflowTracker`
is used by default (disable with --no-mlflow, e.g. in unit tests).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from heart import __version__, config
from heart.data import load_processed, train_test
from heart.evaluate import (
    SCORING,
    cv_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_roc_overlay,
    plot_roc_single,
    test_metrics,
)
from heart.features import build_preprocessor
from heart.models import get_model_specs

AUC_TOLERANCE = 0.02  # models within this CV-AUC of the leader are "equally good"


class RunTracker:
    """Interface for experiment tracking. Default implementation does nothing."""

    def start_run(self, name: str) -> None: ...
    def log_params(self, params: dict) -> None: ...
    def log_metrics(self, metrics: dict) -> None: ...
    def log_artifact(self, path: Path) -> None: ...
    def log_model(self, model, name: str, X_example) -> None: ...
    def set_tag(self, key: str, value) -> None: ...
    def mark_selected(self, name: str, rationale: str) -> None: ...
    def end_run(self) -> None: ...


@dataclass
class ModelResult:
    name: str
    model: Pipeline
    best_params: dict
    metrics: dict
    fit_seconds: float


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clone(estimator))])


def tune_model(spec, X_train, y_train, seed: int = config.SEED, quick: bool = False):
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    grid = spec.param_grid
    if quick:  # first value of each param - used by CI/tests for speed
        grid = {k: v[:1] for k, v in grid.items()}
    search = GridSearchCV(
        build_pipeline(spec.estimator), grid, scoring=SCORING, refit="roc_auc",
        cv=cv, n_jobs=-1, return_train_score=False,
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    return search, time.perf_counter() - t0


def select_model(results: list[ModelResult]) -> tuple[ModelResult, str]:
    """Screening rule, decided on CV only (the test set never drives the choice).

    A missed patient (false negative) costs far more than an extra follow-up
    test, so among models whose CV ROC-AUC is within AUC_TOLERANCE of the best,
    take the one with the highest CV recall; ties -> higher CV ROC-AUC.
    """
    best_auc = max(r.metrics["cv_roc_auc_mean"] for r in results)
    shortlist = [r for r in results if r.metrics["cv_roc_auc_mean"] >= best_auc - AUC_TOLERANCE]
    chosen = max(shortlist, key=lambda r: (round(r.metrics["cv_recall_mean"], 3),
                                           r.metrics["cv_roc_auc_mean"]))
    why = (
        f"Shortlist (CV ROC-AUC within {AUC_TOLERANCE} of best {best_auc:.3f}): "
        f"{', '.join(r.name for r in shortlist)}. Chose '{chosen.name}' for the highest "
        f"CV recall ({chosen.metrics['cv_recall_mean']:.3f}) - false negatives are the "
        f"costly error for a screening tool."
    )
    return chosen, why


def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r.name}
        for m in SCORING:
            row[f"cv_{m}"] = (f"{r.metrics[f'cv_{m}_mean']:.3f} "
                              f"+/- {r.metrics[f'cv_{m}_std']:.3f}")
        for m in SCORING:
            row[f"test_{m}"] = round(r.metrics[f"test_{m}"], 3)
        row["fit_s"] = round(r.fit_seconds, 1)
        row["best_params"] = json.dumps({k.replace("clf__", ""): v
                                         for k, v in r.best_params.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def run(quick: bool = False, tracker: RunTracker | None = None,
        out_dir: Path = config.PROJECT_ROOT, save_mlflow_model: bool = True) -> dict:
    tracker = tracker or RunTracker()
    reports = out_dir / "reports"
    figures = reports / "figures"
    models_dir = out_dir / "models"
    for d in (reports, figures, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = load_processed()
    X_train, X_test, y_train, y_test = train_test(df)
    print(f"train={len(X_train)}  test={len(X_test)}  "
          f"positive rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    results: list[ModelResult] = []
    for spec in get_model_specs():
        search, secs = tune_model(spec, X_train, y_train, quick=quick)
        model = search.best_estimator_
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = cv_metrics(search.cv_results_, search.best_index_)
        metrics.update({f"test_{k}": v for k, v in test_metrics(y_test, pred, proba).items()})

        cm_path = plot_confusion(model, X_test, y_test, spec.name,
                                 figures / f"confusion_{spec.name}.png")
        roc_path = plot_roc_single(model, X_test, y_test, spec.name,
                                   figures / f"roc_{spec.name}.png")
        tracker.start_run(spec.name)
        tracker.log_params({"model": spec.name, "seed": config.SEED,
                            "cv_folds": config.CV_FOLDS, **search.best_params_})
        tracker.log_metrics(metrics)
        tracker.log_artifact(cm_path)
        tracker.log_artifact(roc_path)
        tracker.log_model(model, spec.name, X_train.head(5))
        tracker.set_tag("quick_mode", quick)
        tracker.end_run()

        results.append(ModelResult(spec.name, model, search.best_params_, metrics, secs))
        print(f"[{spec.name:>13}] cv_auc={metrics['cv_roc_auc_mean']:.3f} "
              f"cv_recall={metrics['cv_recall_mean']:.3f} "
              f"test_auc={metrics['test_roc_auc']:.3f} test_recall={metrics['test_recall']:.3f} "
              f"({secs:.1f}s) {search.best_params_}")

    chosen, rationale = select_model(results)
    tracker.mark_selected(chosen.name, rationale)
    print("\n" + rationale)

    table = comparison_table(results)
    table.to_csv(reports / "model_comparison.csv", index=False, encoding="utf-8")
    md = ["# Model comparison", "",
          f"CV = StratifiedKFold({config.CV_FOLDS}, shuffle, seed={config.SEED}) on the "
          f"training split (n={len(X_train)}); test = held-out stratified "
          f"{int(config.TEST_SIZE*100)}% (n={len(X_test)}), threshold 0.5.", "",
          table.to_markdown(index=False), "", "## Selected model", "", rationale, ""]
    (reports / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    plot_roc_overlay({r.name: r.model for r in results}, X_test, y_test,
                     figures / "roc_test.png")
    plot_feature_importance(chosen.model, figures / "feature_importance.png")

    model_path = models_dir / "model.joblib"
    joblib.dump(chosen.model, model_path)
    if save_mlflow_model:  # self-contained MLflow model dir (MLmodel + conda/requirements)
        import shutil

        import mlflow.sklearn

        from heart.tracking import skops_trusted_types

        mlflow_dir = models_dir / "mlflow_model"
        shutil.rmtree(mlflow_dir, ignore_errors=True)
        mlflow.sklearn.save_model(chosen.model, str(mlflow_dir), input_example=X_train.head(5),
                                  skops_trusted_types=skops_trusted_types(chosen.model))
    meta = {
        "model_name": chosen.name, "package_version": __version__,
        "best_params": chosen.best_params, "metrics": chosen.metrics,
        "features": config.FEATURE_COLS, "selection_rationale": rationale,
        "quick_mode": quick,
    }
    (models_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, default=str),
                                                encoding="utf-8")
    print(f"saved {model_path}")
    return {"chosen": chosen.name, "results": results, "table": table}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="use only the first grid value per param (fast smoke run)")
    ap.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    args = ap.parse_args(argv)
    tracker = None
    if not args.no_mlflow:
        from heart.tracking import MlflowTracker

        tracker = MlflowTracker()
    run(quick=args.quick, tracker=tracker, save_mlflow_model=not args.no_mlflow)


if __name__ == "__main__":
    main()
