"""Feature engineering + preprocessing, packaged as one sklearn transformer.

Two clinically-motivated features are derived *inside* the pipeline so the same
code runs during training, CV, and serving:

* pct_max_hr  = thalach / (220 - age)
  Peak heart rate as a fraction of the age-predicted maximum. Raw `thalach`
  falls with age anyway, so this separates "low because old" from "low because
  the patient could not reach their target heart rate" (chronotropic
  incompetence), which is the clinically interesting signal.
* st_per_mmhg = oldpeak / trestbps * 100
  ST depression scaled by resting blood pressure; a rough proxy for ischaemic
  response relative to baseline load. Kept because EDA showed oldpeak is one of
  the strongest single predictors and this lets linear models see an interaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from heart import config


def add_derived_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    age_max = (220 - X["age"]).replace(0, np.nan)
    X["pct_max_hr"] = X["thalach"] / age_max
    X["st_per_mmhg"] = X["oldpeak"] / X["trestbps"].replace(0, np.nan) * 100
    return X


def _derived_names(transformer, input_features):
    return np.asarray(list(input_features) + config.DERIVED_COLS, dtype=object)


def build_preprocessor() -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary = SimpleImputer(strategy="most_frequent")

    columns = ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_COLS + config.DERIVED_COLS),
            ("cat", categorical, config.CATEGORICAL_COLS),
            ("bin", binary, config.BINARY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline([
        ("derive", FunctionTransformer(add_derived_features,
                                       feature_names_out=_derived_names, validate=False)),
        ("columns", columns),
    ])
