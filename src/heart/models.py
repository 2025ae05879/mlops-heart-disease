"""Candidate estimators and their hyper-parameter grids.

Grids are deliberately small: the dataset has ~300 rows, so wide grids mostly
over-fit the CV folds. Each grid targets the knob that matters most for that
model family on small tabular data.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from heart import config


@dataclass
class ModelSpec:
    name: str
    estimator: object
    param_grid: dict = field(default_factory=dict)
    note: str = ""


def get_model_specs(seed: int = config.SEED) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logreg",
            estimator=LogisticRegression(max_iter=10000, solver="saga", random_state=seed),
            param_grid={
                "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                # 0 = ridge, 1 = lasso, 0.5 = elastic-net (sklearn>=1.8 API)
                "clf__l1_ratio": [0.0, 0.5, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
            note="Interpretable baseline; the L1 share shows which one-hot levels survive.",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=seed, n_jobs=-1),
            param_grid={
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__max_features": ["sqrt", 0.5],
            },
            note="Captures interactions (e.g. sex x chest-pain) without manual crosses.",
        ),
        ModelSpec(
            name="hist_gb",
            estimator=HistGradientBoostingClassifier(random_state=seed),
            param_grid={
                "clf__learning_rate": [0.03, 0.1],
                "clf__max_depth": [2, 3],
                "clf__max_iter": [100, 300],
                "clf__l2_regularization": [0.0, 1.0],
            },
            note="Boosting with shallow trees; strong regularisation for n~300.",
        ),
    ]
