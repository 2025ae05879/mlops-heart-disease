import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from heart import predict as predict_mod
from heart.data import clean, split_xy
from heart.train import build_pipeline


@pytest.fixture
def model_file(tmp_path, raw_df, monkeypatch):
    X, y = split_xy(clean(raw_df))
    path = tmp_path / "model.joblib"
    joblib.dump(build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y), path)
    monkeypatch.setenv("HEART_MODEL_PATH", str(path))
    predict_mod.load_model.cache_clear()
    yield path
    predict_mod.load_model.cache_clear()


@pytest.fixture
def client(model_file):
    from heart.api import app

    return TestClient(app)


def test_predict_function(model_file, sample_patient):
    (res,) = predict_mod.predict(sample_patient)
    assert res["prediction"] in (0, 1)
    assert 0.5 <= res["confidence"] <= 1
    assert res["confidence"] == round(max(res["probability_disease"],
                                          1 - res["probability_disease"]), 4)


def test_predict_function_rejects_missing_fields(model_file, sample_patient):
    sample_patient.pop("age")
    with pytest.raises(ValueError, match="age"):
        predict_mod.predict(sample_patient)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_endpoint(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    assert set(r.json()) == {"prediction", "label", "probability_disease", "confidence"}


def test_predict_endpoint_allows_missing_ca_thal(client, sample_patient):
    for field in ("ca", "thal"):
        sample_patient.pop(field)
    assert client.post("/predict", json=sample_patient).status_code == 200


def test_predict_endpoint_validates_input(client, sample_patient):
    sample_patient["cp"] = 9
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_health_degraded_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HEART_MODEL_PATH", str(tmp_path / "nope.joblib"))
    predict_mod.load_model.cache_clear()
    from heart.api import app

    c = TestClient(app)
    assert c.get("/health").json()["model_loaded"] is False
    assert c.post("/predict", json={"age": 1}).status_code == 422
    predict_mod.load_model.cache_clear()


def test_metrics_exposes_request_and_prediction_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    client.post("/predict", json={**sample_patient, "cp": 9})  # 422 -> error series
    body = client.get("/metrics").text
    assert "heart_api_requests_total" in body
    assert 'path="/predict"' in body and 'status="422"' in body
    assert "heart_api_request_duration_seconds_bucket" in body
    assert "heart_api_predictions_total" in body
    assert "heart_api_model_loaded" in body


def test_request_is_logged_as_json(client, sample_patient, capsys):
    import json

    client.post("/predict", json=sample_patient)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    record = json.loads(lines[-1])
    assert record["path"] == "/predict" and record["status"] == 200
    assert record["prediction"] in (0, 1) and record["latency_ms"] >= 0
