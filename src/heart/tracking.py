"""MLflow implementation of the RunTracker interface used by heart.train.

Everything stays local under ./mlruns:
  * run metadata (params/metrics/tags) -> ./mlruns/mlflow.db  (SQLite)
  * artifacts (plots, logged models)   -> ./mlruns/artifacts/
MLflow 3.x refuses the plain file backend by default ("maintenance mode"), so a
SQLite file is used instead - still a single local file, no server needed.

View:  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")  # no git on the dev laptop
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

from heart import config  # noqa: E402
from heart.train import RunTracker  # noqa: E402

EXPERIMENT = "heart-disease-classifier"
# MLflow 3 serialises sklearn models with skops, which only loads allow-listed types.
# We trust exactly the types skops flags that come from these packages (our own
# FunctionTransformer callables, sklearn tree internals, numpy dtypes); anything else
# still fails loudly.
TRUSTED_PREFIXES = ("heart.", "sklearn.", "numpy.")


def skops_trusted_types(model) -> list[str]:
    import skops.io as sio

    flagged = sio.get_untrusted_types(data=sio.dumps(model))
    return [t for t in flagged if t.startswith(TRUSTED_PREFIXES)]


class MlflowTracker(RunTracker):
    def __init__(self, tracking_dir: Path = config.PROJECT_ROOT / "mlruns",
                 experiment: str = EXPERIMENT) -> None:
        root = Path(tracking_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment,
                                     artifact_location=(root / "artifacts").as_uri())
        mlflow.set_experiment(experiment)
        self.run_ids: dict[str, str] = {}

    def start_run(self, name: str) -> None:
        run = mlflow.start_run(run_name=name)
        self.run_ids[name] = run.info.run_id

    def log_params(self, params: dict) -> None:
        mlflow.log_params({k.replace("clf__", ""): v for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, path: Path) -> None:
        mlflow.log_artifact(str(path), artifact_path="plots")

    def log_model(self, model, name: str, X_example) -> None:
        mlflow.sklearn.log_model(model, name="model", input_example=X_example,
                                 skops_trusted_types=skops_trusted_types(model))

    def set_tag(self, key: str, value) -> None:
        mlflow.set_tag(key, value)

    def mark_selected(self, name: str, rationale: str) -> None:
        client = mlflow.MlflowClient()
        for model_name, run_id in self.run_ids.items():
            client.set_tag(run_id, "selected", str(model_name == name).lower())
        client.set_tag(self.run_ids[name], "selection_rationale", rationale)

    def end_run(self) -> None:
        mlflow.end_run()
