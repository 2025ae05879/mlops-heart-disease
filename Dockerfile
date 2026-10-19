# syntax=docker/dockerfile:1
# ---------- stage 1: build a self-contained virtualenv ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

# ---------- stage 2: slim runtime, non-root ----------
FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="heart-api" \
      org.opencontainers.image.description="Heart-disease risk API (BITS AIMLCZG523 assignment)"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HEART_MODEL_PATH=/app/models/model.joblib
RUN useradd --create-home --uid 10001 heart
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY models/model.joblib models/model_meta.json /app/models/
USER heart
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"
CMD ["uvicorn", "heart.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
