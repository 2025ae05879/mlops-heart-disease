"""Loading, cleaning and splitting the Cleveland heart-disease data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from heart import config

# Physiologically valid codes; anything else is treated as missing.
VALID_CODES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
    "ca": {0, 1, 2, 3},
}


def load_raw(path: Path | str = config.RAW_FILE) -> pd.DataFrame:
    """Read the raw CSV written by scripts/download_data.py.

    '?' is the UCI missing-value marker (only present in `ca` and `thal`).
    """
    return pd.read_csv(path, na_values=["?"], encoding="utf-8")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed, validated frame with a binary `target` column.

    Decisions (see README / report):
    * `?` in `ca` (4 rows) and `thal` (2 rows) are kept as NaN here and imputed
      *inside* the model pipeline (most-frequent for `thal`, median for `ca`).
      Dropping would lose 2% of an already small dataset, and imputing in the
      pipeline means (a) the imputer is fitted on training folds only - no
      leakage into CV/test - and (b) the deployed API tolerates a missing field.
    * Codes outside the documented UCI domain are set to NaN for the same path.
    * `num` (0-4 severity) is binarised: 0 -> 0 (no disease), 1-4 -> 1.
    * Exact duplicate rows are dropped.
    """
    out = df.copy()
    missing = [c for c in config.RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")

    out = out[config.RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    for col, valid in VALID_CODES.items():
        out.loc[~out[col].isin(valid) & out[col].notna(), col] = np.nan

    out[config.TARGET] = (out["num"] > 0).astype(int)
    out = out.drop(columns="num").drop_duplicates().reset_index(drop=True)

    # Store nominal codes as nullable integers so the CSV stays readable (3 not 3.0)
    for col in config.CATEGORICAL_COLS + config.BINARY_COLS + ["ca"]:
        out[col] = out[col].astype("Int64")
    return out


def save_processed(df: pd.DataFrame, path: Path | str = config.PROCESSED_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def load_processed(path: Path | str = config.PROCESSED_FILE) -> pd.DataFrame:
    """Load the cleaned CSV; regenerates it from raw if it does not exist yet."""
    path = Path(path)
    if not path.exists():
        save_processed(clean(load_raw()), path)
    return pd.read_csv(path, encoding="utf-8")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[config.FEATURE_COLS].astype(float)  # NaN-safe float matrix for sklearn
    y = df[config.TARGET].astype(int)
    return X, y


def train_test(df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.SEED):
    """Stratified hold-out split; the test set is never seen during tuning."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
