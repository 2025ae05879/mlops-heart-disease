"""Structured JSON logging + Prometheus metrics for the API.

Every metric carries a `pod` label (from the POD_NAME env var set by the
Deployment). Prometheus scrapes the service through the LoadBalancer, so each
scrape lands on a random replica; the pod label keeps the replicas' counters as
separate series instead of one series that keeps "resetting".
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
from datetime import UTC, datetime

from prometheus_client import Counter, Gauge, Histogram

POD = os.environ.get("POD_NAME", socket.gethostname())

REQUESTS = Counter(
    "heart_api_requests_total", "HTTP requests handled",
    ["pod", "method", "path", "status"],
)
LATENCY = Histogram(
    "heart_api_request_duration_seconds", "Request latency in seconds",
    ["pod", "method", "path"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter(
    "heart_api_predictions_total", "Predictions returned, by predicted class",
    ["pod", "label"],
)
RISK = Histogram(
    "heart_api_predicted_probability", "Distribution of predicted P(disease)",
    ["pod"], buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_LOADED = Gauge("heart_api_model_loaded", "1 if the model file is loaded", ["pod"])


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "pod": POD,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _StdoutHandler(logging.StreamHandler):
    """Always writes to the *current* sys.stdout (plays well with test capture)."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def get_logger(name: str = "heart.api") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = _StdoutHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger
