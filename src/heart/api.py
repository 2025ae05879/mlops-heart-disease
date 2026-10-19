"""FastAPI service:  uvicorn heart.api:app --port 8000

GET  /health   liveness/readiness + whether the model file loaded
POST /predict  one patient (13 raw UCI fields) -> prediction + confidence
GET  /metrics  Prometheus metrics (requests, latency, predictions by class)

Every request is logged as one JSON line (method, path, status, latency, prediction).
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from heart import __version__
from heart.observability import (
    LATENCY,
    MODEL_LOADED,
    POD,
    PREDICTIONS,
    REQUESTS,
    RISK,
    get_logger,
)
from heart.predict import load_model, model_path, predict

app = FastAPI(title="Heart disease risk API", version=__version__)
log = get_logger()


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[57])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="chest pain type 1-4")
    trestbps: float = Field(..., gt=0, examples=[130])
    chol: float = Field(..., gt=0, examples=[240])
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0, examples=[150])
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, examples=[1.0])
    slope: int = Field(..., ge=1, le=3)
    ca: float | None = Field(None, ge=0, le=3, description="may be omitted; imputed")
    thal: float | None = Field(None, description="3/6/7; may be omitted; imputed")


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    request.state.request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request.state.request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"  # bounded label cardinality
        REQUESTS.labels(POD, request.method, path, str(status)).inc()
        LATENCY.labels(POD, request.method, path).observe(elapsed)
        if path != "/metrics":  # keep scrape noise out of the log
            log.info("request", extra={"fields": {
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status": status,
                "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None,
                "prediction": getattr(request.state, "prediction", None),
                "probability": getattr(request.state, "probability", None),
            }})


def _model_ready() -> bool:
    try:
        load_model()
        ok = True
    except Exception:  # noqa: BLE001 - report, don't crash the probe
        ok = False
    MODEL_LOADED.labels(POD).set(int(ok))
    return ok


@app.get("/health")
def health() -> dict:
    loaded = _model_ready()
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
            "model_path": str(model_path()), "version": __version__, "pod": POD}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient, request: Request) -> dict:
    try:
        model = load_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not available: {exc}") from exc
    result = predict(patient.model_dump(), model=model)[0]
    PREDICTIONS.labels(POD, result["label"]).inc()
    RISK.labels(POD).observe(result["probability_disease"])
    request.state.prediction = result["prediction"]
    request.state.probability = result["probability_disease"]
    return result


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    _model_ready()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
