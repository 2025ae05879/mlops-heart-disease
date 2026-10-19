"""Single source of truth for paths, column roles and the random seed.

Every other module imports from here so that the training script, the tests,
the notebook and (later) the API agree on what each column means.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILE = RAW_DIR / "cleveland_raw.csv"
PROCESSED_FILE = PROCESSED_DIR / "heart_clean.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
UCI_REPO_ID = 45

SEED = 523  # course code, used everywhere for reproducibility
TEST_SIZE = 0.2
CV_FOLDS = 5

# The 14 attributes of processed.cleveland.data, in file order.
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
TARGET = "target"

# Column roles -----------------------------------------------------------------
# Continuous measurements -> impute (median) + StandardScaler.
# `ca` (number of major vessels coloured by fluoroscopy, 0-3) is a count with a
# natural order and a roughly monotone link to disease, so it is kept numeric
# rather than one-hot encoded.
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Nominal codes with no meaningful distance between levels -> one-hot.
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
# Already 0/1 flags -> passed through unchanged.
BINARY_COLS = ["sex", "fbs", "exang"]
# Derived in heart.features (inside the pipeline, so it is reproduced at serving).
DERIVED_COLS = ["pct_max_hr", "st_per_mmhg"]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS
