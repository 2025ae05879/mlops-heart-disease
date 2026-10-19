import numpy as np

from heart import config
from heart.data import clean, split_xy
from heart.features import add_derived_features, build_preprocessor


def test_derived_features_values():
    import pandas as pd

    X = pd.DataFrame({"age": [60.0], "thalach": [120.0], "oldpeak": [2.0], "trestbps": [100.0]})
    out = add_derived_features(X)
    assert out.loc[0, "pct_max_hr"] == 120 / 160
    assert out.loc[0, "st_per_mmhg"] == 2.0
    assert "pct_max_hr" not in X.columns  # input not mutated


def test_preprocessor_output_is_finite_and_shaped(raw_df):
    X, _ = split_xy(clean(raw_df))
    prep = build_preprocessor().fit(X)
    Z = prep.transform(X)
    n_cat = sum(X[c].nunique() for c in config.CATEGORICAL_COLS)
    expected = len(config.NUMERIC_COLS) + len(config.DERIVED_COLS) + n_cat + len(config.BINARY_COLS)
    assert Z.shape == (len(X), expected)
    assert np.isfinite(Z).all()  # NaNs in ca/thal were imputed
    assert len(prep.get_feature_names_out()) == expected


def test_unseen_category_is_ignored(raw_df):
    X, _ = split_xy(clean(raw_df))
    X = X[X["restecg"] != 1]
    prep = build_preprocessor().fit(X)
    new = X.head(1).copy()
    new["restecg"] = 1.0  # level never seen in training
    assert np.isfinite(prep.transform(new)).all()


def test_scaled_numeric_columns_are_standardised(raw_df):
    X, _ = split_xy(clean(raw_df))
    Z = build_preprocessor().fit_transform(X)
    num = Z[:, : len(config.NUMERIC_COLS)]
    assert np.allclose(num.mean(axis=0), 0, atol=1e-6)
