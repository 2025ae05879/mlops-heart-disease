"""Send a burst of /predict requests built from real (cleaned) patient rows.

    python scripts/load_test.py --url http://127.0.0.1:8080 --n 50 --invalid-rate 0.06

Use 127.0.0.1 rather than "localhost" on Windows: urllib tries ::1 first and the
Docker Desktop LoadBalancer only answers on IPv4, which adds ~2 s per request.

A small share of deliberately invalid payloads (cp=9) is mixed in so the error-rate
panel in Grafana has something to show. Prints a one-line summary per request and
a final tally of status codes and predicted classes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from heart import config  # noqa: E402


def load_patients() -> list[dict]:
    import pandas as pd

    df = pd.read_csv(config.PROCESSED_FILE)[config.FEATURE_COLS]
    rows = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    return [{k: (None if v is None else float(v)) for k, v in r.items()} for r in rows]


def post(url: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(f"{url}/predict", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--invalid-rate", type=float, default=0.06)
    ap.add_argument("--delay", type=float, default=0.1, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    args.url = args.url.replace("://localhost", "://127.0.0.1")  # see module docstring
    rng = random.Random(args.seed)
    patients = load_patients()
    statuses, labels, latencies = Counter(), Counter(), []
    for i in range(args.n):
        payload = dict(rng.choice(patients))
        if rng.random() < args.invalid_rate:
            payload["cp"] = 9
        t0 = time.perf_counter()
        status, body = post(args.url.rstrip("/"), payload)
        latencies.append((time.perf_counter() - t0) * 1000)
        statuses[status] += 1
        if body:
            labels[body["label"]] += 1
        print(f"{i + 1:3d} HTTP {status} {body.get('label', '-'):>17} "
              f"p={body.get('probability_disease', '-')}  {latencies[-1]:.1f} ms")
        time.sleep(args.delay)
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    print(f"\nsent={args.n} status={dict(statuses)} predictions={dict(labels)} "
          f"p95_client_latency={p95:.1f} ms")
    return 0 if statuses.get(200, 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
