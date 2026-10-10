"""
Dataset Acquisition Pipeline for Sentinel
Fetches the official PaySim dataset from Kaggle (ealaxi/paysim1)
License: CC BY-SA 4.0
Source: Lopez-Rojas et al., "PaySim: A financial mobile money simulator for fraud detection"
"""

import os
import sys
import subprocess
import glob
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_SLUG = "ealaxi/paysim1"

def acquire_paysim() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if CSV already exists
    csv_candidates = list(DATA_DIR.glob("*.csv"))
    if csv_candidates:
        csv_path = csv_candidates[0]
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"[ACQUISITION] Found existing PaySim CSV: {csv_path} ({size_mb:.2f} MB)")
        return csv_path

    print(f"[ACQUISITION] Downloading official PaySim dataset '{DATASET_SLUG}' from Kaggle...")
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "-p", str(DATA_DIR),
            "--unzip"
        ]
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[ACQUISITION] Kaggle output: {res.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download via Kaggle CLI: {e.stderr}", file=sys.stderr)
        print(
            "\n[MANUAL ACQUISITION INSTRUCTIONS]:\n"
            "1. Visit https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "2. Download the ZIP file (approx 186 MB)\n"
            f"3. Extract the CSV file into: {DATA_DIR.resolve()}\n",
            file=sys.stderr
        )
        raise

    csv_candidates = list(DATA_DIR.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found in {DATA_DIR} after unzipping.")
    
    csv_path = csv_candidates[0]
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"[ACQUISITION SUCCESS] PaySim raw dataset acquired: {csv_path.name} ({size_mb:.2f} MB)")
    return csv_path

if __name__ == "__main__":
    acquire_paysim()
