#!/usr/bin/env python3
"""
Download benchmark datasets for irithyll.

Classification datasets:
  - Electricity (Elec2): 45,312 samples, 8 features, binary (already present)
  - Airlines: 539,383 samples, 7 features, binary (flight delay)
  - Covertype: 581,012 samples, 54 features, 7-class (forest cover type)

Regression datasets:
  - CCPP (Combined Cycle Power Plant): 9,568 samples, 4 features (UCI #294)
  - Sunspots (monthly): ~2,820 samples, univariate time series
  - Air Quality: ~6,900 usable samples, 11 features, sensor drift (UCI #360)
"""

import gzip
import os
import sys
import urllib.request
import zipfile
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
SUNSPOTS_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/monthly-sunspots.csv"
)
CCPP_URL = (
    "https://archive.ics.uci.edu/static/public/294/"
    "combined+cycle+power+plant.zip"
)
AIR_QUALITY_URL = (
    "https://archive.ics.uci.edu/static/public/360/"
    "air+quality.zip"
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


def download_sunspots() -> None:
    path = DATASETS_DIR / "sunspots.csv"
    if path.exists():
        n = sum(1 for _ in open(path)) - 1
        print(f"[OK] Sunspots already exists ({n} samples): {path}")
        return

    print("Downloading Sunspots dataset...")
    urllib.request.urlretrieve(SUNSPOTS_URL, path)
    n = sum(1 for _ in open(path)) - 1
    print(f"[OK] Sunspots: {n} samples -> {path}")


def download_ccpp() -> None:
    path = DATASETS_DIR / "ccpp.csv"
    if path.exists():
        n = sum(1 for _ in open(path)) - 1
        print(f"[OK] CCPP already exists ({n} samples): {path}")
        return

    zip_path = DATASETS_DIR / "ccpp.zip"
    print("Downloading CCPP dataset...")
    urllib.request.urlretrieve(CCPP_URL, zip_path)

    # Extract xlsx and convert to CSV (requires openpyxl)
    try:
        import openpyxl
    except ImportError:
        print("[WARN] openpyxl not installed. Run: pip install openpyxl")
        print("       Extracting raw xlsx instead.")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extract("CCPP/Folds5x2_pp.xlsx", DATASETS_DIR)
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("CCPP/Folds5x2_pp.xlsx", DATASETS_DIR)

    xlsx_path = DATASETS_DIR / "CCPP" / "Folds5x2_pp.xlsx"
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Sheet1"]
    count = 0
    with open(path, "w", newline="") as f:
        f.write("AT,V,AP,RH,PE\n")
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                continue
            vals = [str(v) for v in row if v is not None]
            if len(vals) == 5:
                f.write(",".join(vals) + "\n")
                count += 1
    wb.close()
    print(f"[OK] CCPP: {count} samples -> {path}")


def download_air_quality() -> None:
    path = DATASETS_DIR / "air_quality.csv"
    if path.exists():
        n = sum(1 for _ in open(path)) - 1
        print(f"[OK] Air Quality already exists ({n} samples): {path}")
        return

    zip_path = DATASETS_DIR / "air_quality.zip"
    print("Downloading Air Quality dataset...")
    urllib.request.urlretrieve(AIR_QUALITY_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("AirQualityUCI.csv", DATASETS_DIR)

    raw_path = DATASETS_DIR / "AirQualityUCI.csv"
    # UCI Air Quality uses ; separator and , for decimals (European format).
    # NMHC(GT) is ~90% missing (-200), so we drop it.
    # Features: sensor readings + meteorological. Target: C6H6(GT) (benzene).
    keep_idx = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0]  # feature columns
    target_idx = 3  # C6H6(GT)
    header = "PT08_S1_CO,PT08_S2_NMHC,NOx_GT,PT08_S3_NOx,NO2_GT,PT08_S4_NO2,PT08_S5_O3,T,RH,AH,CO_GT,C6H6_GT"

    count = 0
    skipped = 0
    with open(raw_path, "r") as inp, open(path, "w", newline="") as out:
        inp.readline()  # skip header
        out.write(header + "\n")
        for line in inp:
            line = line.strip()
            if not line:
                continue
            fields = line.split(";")
            vals = [v.replace(",", ".").strip() for v in fields[2:15]]
            try:
                nums = [float(v) for v in vals]
            except ValueError:
                skipped += 1
                continue
            check = [nums[i] for i in keep_idx] + [nums[target_idx]]
            if any(n == -200.0 for n in check):
                skipped += 1
                continue
            row = [vals[i] for i in keep_idx] + [vals[target_idx]]
            out.write(",".join(row) + "\n")
            count += 1

    print(f"[OK] Air Quality: {count} samples -> {path} (skipped {skipped} with missing)")


def main() -> None:
    print("=== irithyll Dataset Downloader ===\n")

    # --- Classification datasets ---
    print("--- Classification ---")
    elec_path = DATASETS_DIR / "electricity.csv"
    if elec_path.exists():
        print(f"[OK] Electricity already exists: {elec_path}")
    else:
        print(f"[WARN] Electricity not found at {elec_path}")

    download_airlines()
    download_covertype()

    # --- Regression datasets ---
    print("\n--- Regression ---")
    download_sunspots()
    download_ccpp()
    download_air_quality()

    print("\nAll datasets ready.")
    for p in sorted(DATASETS_DIR.glob("*.csv")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
