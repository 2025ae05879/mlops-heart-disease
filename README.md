# Heart-disease risk classifier — an end-to-end MLOps project

BITS Pilani WILP · AIMLCZG523 MLOps · Assignment 01 · Jayanth Saish Pamarti

This project predicts angiographic heart disease (`num > 0`) for the UCI Cleveland patients and
runs the model as a monitored API. It covers a reproducible data pull, EDA, a tuned scikit-learn
pipeline, MLflow tracking, tests + GitHub Actions, a Docker image, a Kubernetes deployment, and
Prometheus/Grafana monitoring.

**Repository:** `<GitHub URL - add after first push>` · **Report:** `reports/report.docx` / `reports/report.pdf`

![architecture](reports/figures/architecture.png)

## Repo map

| path | what it holds |
|---|---|
| `scripts/` | `download_data.py`, `load_test.py`, `build_report.py` |
| `src/heart/` | `config` (paths, column roles, seed) · `data` · `features` · `models` · `evaluate` · `train` · `tracking` (MLflow) · `predict` · `api` (FastAPI) · `observability` (JSON logs, metrics) |
| `notebooks/01_eda.ipynb` | executed EDA notebook |
| `data/raw`, `data/processed` | verbatim UCI file with column names · cleaned CSV |
| `reports/` | `model_comparison.{csv,md}`, `figures/`, `report.docx/.pdf` |
| `models/` | `model.joblib`, `mlflow_model/`, `model_meta.json` |
| `tests/` | pytest suite: data, features, model, predict, API, metrics, logging |
| `.github/workflows/ci.yml` | ruff → data → pytest → train → upload artifacts |
| `Dockerfile`, `requirements-serve.txt` | serving image |
| `k8s/` | `deployment.yaml`, `service.yaml` |
| `monitoring/` | docker-compose for Prometheus + Grafana, scrape config, provisioned dashboard |
| `samples/` | example `/predict` payloads |
| `screenshots/`, `screenshots/proof/` | screenshots and saved command outputs |

## Setup (Windows PowerShell; on Linux/macOS use `.venv/bin/...`)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:PYTHONIOENCODING = "utf-8"
```

Docker Desktop is needed from phase 6 on, with its Kubernetes setting turned on for phase 7.

## Run, phase by phase

**1 · Data & EDA**
```powershell
.\.venv\Scripts\python.exe scripts\download_data.py      # UCI URL, falls back to ucimlrepo id 45
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace notebooks\01_eda.ipynb
```
`?` appears only in `ca` (4 rows) and `thal` (2 rows). Those rows are kept and the values are imputed
inside the pipeline (median / mode), fitted on training folds only. The target is `num > 0`, which
gives 46% positives.

**2–3 · Train, tune, track**
```powershell
.\.venv\Scripts\python.exe -m heart.train               # add --quick for a smoke run, --no-mlflow to skip tracking
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db   # http://127.0.0.1:5000
```
- **Models:** LogReg, RandomForest and HistGradientBoosting, each tuned with GridSearchCV and
  StratifiedKFold(5, seed 523) on an 80/20 stratified split.
- **Selection:** the highest CV recall among models within 0.02 of the best CV ROC-AUC. The result
  is logistic regression (CV AUC 0.910, test AUC 0.913); see `reports/model_comparison.md`.
- **MLflow:** one run per model with params, metrics, ROC and confusion plots, and the model.
  MLflow 3 refuses a plain `./mlruns` folder by default, so run metadata lives in
  `mlruns/mlflow.db` and artifacts in `mlruns/artifacts/`.

**4 · Packaging.** The full pipeline is saved as `models/model.joblib` and `models/mlflow_model/`.
Dependencies are pinned in `requirements.txt`, and the package installs with `pip install -e .`.

**5 · Tests & CI**
```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\python.exe -m pytest -v
```
`.github/workflows/ci.yml` runs the same steps on push/PR, trains the model and uploads models,
reports and mlruns as an artifact. *It is written but has not run on GitHub yet* (the repo isn't
pushed).

**6 · Docker**
```powershell
docker build -t heart-api:0.2.0 .
docker run -d --name heart-api-local -p 8000:8000 heart-api:0.2.0
curl.exe http://localhost:8000/health
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" --data-binary "@samples/sample_request.json"
docker rm -f heart-api-local
```
`/predict` returns `{"prediction","label","probability_disease","confidence"}`. Invalid input gets
HTTP 422. `ca` and `thal` may be omitted.

**7 · Kubernetes (Docker Desktop)**
```powershell
kubectl config use-context docker-desktop
kubectl apply -f k8s/
kubectl rollout status deploy/heart-api
kubectl get pods,svc,deploy -o wide
curl.exe http://localhost:8080/health
```
The deployment runs 2 replicas using the local image (`IfNotPresent`), with `/health` probes and
small CPU/memory requests and limits.

> **Port 8080, not 80.** On the development laptop, `localhost:80` is already published by an
> unrelated container, so Docker Desktop's load balancer could not bind port 80
> (`SyncLoadBalancerFailed` in `kubectl describe svc heart-api`). The LoadBalancer service
> therefore maps **8080 → 8000**. If port 80 is free on your machine, you can set `port: 80` in
> `k8s/service.yaml`.

**8 · Monitoring**
```powershell
docker compose -f monitoring/docker-compose.yml up -d
.\.venv\Scripts\python.exe scripts\load_test.py --n 60      # traffic to http://127.0.0.1:8080
```
- **Prometheus:** http://localhost:9090, scraping `host.docker.internal:8080`.
- **Grafana:** http://localhost:3000, login `admin` / `heart-mlops` (local demo only). The
  dashboard is *Heart API - model service health*.
- **Logs:** one JSON line per request, viewable with
  `kubectl logs -l app.kubernetes.io/name=heart-api --prefix`.
- **Metrics:** request count, latency histogram, predictions by class, predicted-probability
  histogram and a model-loaded flag, all labelled by `pod`.
- **Known limitation:** scrapes go through the LoadBalancer over one keep-alive connection, so
  they mostly hit one replica. An in-cluster Prometheus with pod discovery would fix this.
- **Stop everything:** `docker compose -f monitoring/docker-compose.yml down; kubectl delete -f k8s/`

**9 · Report**
```powershell
.\.venv\Scripts\python.exe scripts\build_report.py         # reports/report.docx (+ .pdf via MS Word)
```
Screenshots saved as `screenshots/<name>.png` (names listed as placeholders in the report) are
embedded automatically on the next run.
