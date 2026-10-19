"""Load the packaged pipeline and score raw patient records.

The pipeline contains imputation, feature derivation, scaling and encoding, so
callers pass the 13 raw UCI fields exactly as they appear in the dataset.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart import config

DEFAULT_MODEL_PATH = config.MODELS_DIR / "model.joblib"


def model_path() -> Path:
    return Path(os.environ.get("HEART_MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    return joblib.load(path or model_path())


def to_frame(records: dict | list[dict]) -> pd.DataFrame:
    rows = [records] if isinstance(records, dict) else list(records)
    df = pd.DataFrame(rows)
    missing = [c for c in config.FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    return df[config.FEATURE_COLS].astype(float)


def predict(records: dict | list[dict], model=None) -> list[dict]:
    """Return one {prediction, label, probability, confidence} dict per record."""
    model = model or load_model()
    proba = model.predict_proba(to_frame(records))[:, 1]
    out = []
    for p in proba:
        pred = int(p >= 0.5)
        out.append({
            "prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(max(p, 1 - p)), 4),
        })
    return out
