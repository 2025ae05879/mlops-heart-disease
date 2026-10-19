import numpy as np
import pandas as pd
import pytest

from heart import config

SAMPLE_PATIENT = {
    "age": 57, "sex": 1, "cp": 4, "trestbps": 140, "chol": 241, "fbs": 0, "restecg": 0,
    "thalach": 123, "exang": 1, "oldpeak": 0.2, "slope": 2, "ca": 0, "thal": 7,
}


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic frame shaped like processed.cleveland.data after load_raw()."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.integers(30, 75, n).astype(float),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.integers(95, 180, n).astype(float),
        "chol": rng.integers(150, 350, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.integers(90, 200, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": rng.uniform(0, 4, n).round(1),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "num": rng.integers(0, 5, n),
    })
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    return df[config.RAW_COLUMNS]


@pytest.fixture
def sample_patient() -> dict:
    return dict(SAMPLE_PATIENT)
