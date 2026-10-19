import json

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.data import clean, load_raw, split_xy
from heart.models import get_model_specs
from heart.train import ModelResult, build_pipeline, run, select_model


def test_pipeline_fits_and_predicts_probabilities(raw_df):
    X, y = split_xy(clean(raw_df))
    model = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_three_model_families_with_grids():
    specs = get_model_specs()
    assert [s.name for s in specs] == ["logreg", "random_forest", "hist_gb"]
    assert all(k.startswith("clf__") for s in specs for k in s.param_grid)


def _result(name, auc, recall):
    return ModelResult(name, None, {}, {"cv_roc_auc_mean": auc, "cv_recall_mean": recall}, 0.0)


def test_selection_prefers_recall_within_auc_band():
    chosen, why = select_model([_result("a", 0.91, 0.70), _result("b", 0.90, 0.85),
                                _result("c", 0.80, 0.99)])
    assert chosen.name == "b"  # c has best recall but is outside the AUC band
    assert "recall" in why


@pytest.mark.skipif(not config.RAW_FILE.exists(), reason="run scripts/download_data.py first")
def test_quick_training_end_to_end(tmp_path):
    out = run(quick=True, out_dir=tmp_path, save_mlflow_model=False)
    assert out["chosen"] in {"logreg", "random_forest", "hist_gb"}
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "reports" / "model_comparison.csv").exists()
    meta = json.loads((tmp_path / "models" / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["test_roc_auc"] > 0.75  # sanity floor on real data
    model = joblib.load(tmp_path / "models" / "model.joblib")
    X, _ = split_xy(clean(load_raw()))
    assert model.predict(X.head(5)).shape == (5,)
