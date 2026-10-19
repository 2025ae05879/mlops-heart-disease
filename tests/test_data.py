import pandas as pd
import pytest

from heart import config
from heart.data import clean, load_raw, split_xy, train_test


def test_target_is_binarised(raw_df):
    out = clean(raw_df)
    assert set(out[config.TARGET].unique()) <= {0, 1}
    assert "num" not in out.columns
    assert (out[config.TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_missing_ca_thal_kept_as_nan(raw_df):
    out = clean(raw_df)
    assert len(out) == len(raw_df)  # rows are not dropped
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1


def test_out_of_domain_codes_become_nan(raw_df):
    raw_df.loc[5, "thal"] = 5      # not a valid thal code
    raw_df.loc[6, "cp"] = 9
    out = clean(raw_df)
    assert pd.isna(out.loc[5, "thal"]) and pd.isna(out.loc[6, "cp"])


def test_question_mark_parsed_as_missing(tmp_path, raw_df):
    path = tmp_path / "raw.csv"
    text = raw_df.to_csv(index=False).replace(",,", ",?,")  # NaN -> '?'
    path.write_text(text, encoding="utf-8")
    assert load_raw(path)[["ca", "thal"]].isna().sum().sum() == 2


def test_missing_column_raises(raw_df):
    with pytest.raises(ValueError, match="missing columns"):
        clean(raw_df.drop(columns="thal"))


def test_stratified_split_keeps_ratio(raw_df):
    df = clean(raw_df)
    X_tr, X_te, y_tr, y_te = train_test(df, test_size=0.25, seed=1)
    assert len(X_te) == 15 and list(X_tr.columns) == config.FEATURE_COLS
    assert abs(y_tr.mean() - y_te.mean()) < 0.1
    X, y = split_xy(df)
    assert X.dtypes.eq(float).all() and y.dtype == int
