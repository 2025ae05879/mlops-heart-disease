"""Download the UCI Cleveland heart-disease data and write cleaned CSVs.

    python scripts/download_data.py            # download (if needed) + clean
    python scripts/download_data.py --force    # re-download

Primary source : archive.ics.uci.edu  processed.cleveland.data (303 rows, 14 cols,
                 no header, '?' = missing)
Fallback       : `ucimlrepo` package, dataset id 45 (same records)
Outputs        : data/raw/cleveland_raw.csv       (verbatim + column names)
                 data/processed/heart_clean.csv   (see heart.data.clean)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heart import config  # noqa: E402
from heart.data import clean, load_raw, save_processed  # noqa: E402


def from_uci_url() -> pd.DataFrame:
    resp = requests.get(config.UCI_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), header=None, names=config.RAW_COLUMNS,
                     dtype=str, skipinitialspace=True)
    print(f"downloaded {len(resp.content)} bytes from {config.UCI_URL} "
          f"(sha256 {hashlib.sha256(resp.content).hexdigest()[:12]})")
    return df


def from_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    ds = fetch_ucirepo(id=config.UCI_REPO_ID)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df = df[config.RAW_COLUMNS].astype("string").fillna("?")
    print(f"fetched dataset id={config.UCI_REPO_ID} via ucimlrepo")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if raw file exists")
    args = ap.parse_args()

    if args.force or not config.RAW_FILE.exists():
        try:
            raw = from_uci_url()
        except Exception as exc:  # network / HTTP error -> fallback
            print(f"direct download failed ({exc!r}); trying ucimlrepo ...")
            raw = from_ucimlrepo()
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(config.RAW_FILE, index=False, encoding="utf-8")
        print(f"wrote {config.RAW_FILE} shape={raw.shape}")
    else:
        print(f"raw file already present: {config.RAW_FILE}")

    raw = load_raw()
    print("missing values in raw:", raw.isna().sum()[lambda s: s > 0].to_dict())
    cleaned = clean(raw)
    out = save_processed(cleaned)
    print(f"wrote {out} shape={cleaned.shape} "
          f"positive rate={cleaned[config.TARGET].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
