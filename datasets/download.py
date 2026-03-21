#!/usr/bin/env python3
"""
Download benchmark datasets for irithyll.

Datasets:
  - Electricity (Elec2): 45,312 samples, 8 features, binary (already present)
  - Airlines: 539,383 samples, 7 features, binary (flight delay)
  - Covertype: 581,012 samples, 54 features, 7-class (forest cover type)
"""

import gzip
import os
import sys
import urllib.request
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent

AIRLINES_URL = (
    "https://raw.githubusercontent.com/scikit-multiflow/"
    "streaming-datasets/master/airlines.csv"
)
COVERTYPE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "covtype/covtype.data.gz"
)

COVERTYPE_FEATURES = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    *[f"Wilderness_Area_{i}" for i in range(1, 5)],
    *[f"Soil_Type_{i}" for i in range(1, 41)],
]
COVERTYPE_HEADER = ",".join(COVERTYPE_FEATURES + ["Cover_Type"])


def download_airlines() -> None:
    path = DATASETS_DIR / "airlines.csv"
    if path.exists():
        n = sum(1 for _ in open(path)) - 1
        print(f"[OK] Airlines already exists ({n} samples): {path}")
        return

    print("Downloading Airlines dataset...")
    urllib.request.urlretrieve(AIRLINES_URL, path)
    n = sum(1 for _ in open(path)) - 1
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[OK] Airlines: {n} samples, {size_mb:.1f} MB -> {path}")


def download_covertype() -> None:
    path = DATASETS_DIR / "covertype.csv"
    if path.exists():
        n = sum(1 for _ in open(path)) - 1
        print(f"[OK] Covertype already exists ({n} samples): {path}")
        return

    print("Downloading Covertype dataset...")
    response = urllib.request.urlopen(COVERTYPE_URL)
    gz_data = response.read()
    raw = gzip.decompress(gz_data).decode("utf-8")

    # UCI format has no header; add one
    with open(path, "w", newline="") as f:
        f.write(COVERTYPE_HEADER + "\n")
        # Strip trailing whitespace/empty lines
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped:
                f.write(stripped + "\n")

    n = sum(1 for _ in open(path)) - 1
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[OK] Covertype: {n} samples, {size_mb:.1f} MB -> {path}")


def main() -> None:
    print("=== irithyll Dataset Downloader ===\n")

    # Electricity should already exist
    elec_path = DATASETS_DIR / "electricity.csv"
    if elec_path.exists():
        print(f"[OK] Electricity already exists: {elec_path}")
    else:
        print(f"[WARN] Electricity not found at {elec_path}")

    download_airlines()
    download_covertype()

    print("\nAll datasets ready.")
    # List what's in the directory
    for p in sorted(DATASETS_DIR.glob("*.csv")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
