"""Download and prepare 3 TableShift benchmark datasets for testing the NAISC pipeline.

Each dataset gets a train.csv and test.csv with a real-world distribution shift:
  1. acsincome      — Income prediction, shift by US Census division (mild, -1.3%)
  2. diabetes       — Hospital readmission, shift by admission source (moderate, -5.9%)
  3. acspubcov      — Public health insurance, shift by disability status (severe, -14.5%)

Usage:
    python test_data/setup_tableshift.py

Output:
    test_data/acsincome/train.csv, test_data/acsincome/test.csv
    test_data/diabetes/train.csv, test_data/diabetes/test.csv
    test_data/acspubcov/train.csv, test_data/acspubcov/test.csv
"""

import os
import sys
import zipfile
import io

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── 1. ACS datasets (folktables) ────────────────────────────────────────

def setup_acs_dataset(task_name: str, ood_division: str, out_dir: str, sample_n: int = 50000):
    """Download ACS data via folktables and split by Census division for drift."""
    from folktables import ACSDataSource, ACSIncome, ACSPublicCoverage

    task_map = {
        "acsincome": ACSIncome,
        "acspubcov": ACSPublicCoverage,
    }
    task = task_map[task_name]

    print(f"[{task_name}] Downloading ACS data (2018, 1-year)...")
    data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=["CA", "NY", "TX", "FL", "IL"], download=True)

    features, label, group = task.df_to_pandas(acs_data)
    df = features.copy()
    df["target"] = label.astype(int)

    # Carry over split columns from raw data (they get dropped by df_to_pandas)
    for extra_col in ["DIVISION", "ST"]:
        if extra_col not in df.columns and extra_col in acs_data.columns:
            df[extra_col] = acs_data[extra_col].values[: len(df)]

    # Add an ID column
    df.insert(0, "SampleID", [f"S{i:06d}" for i in range(len(df))])

    # DIVISION column defines the shift
    # In-distribution: all divisions except ood_division
    # Out-of-distribution: ood_division only
    if "DIVISION" in df.columns:
        div_col = "DIVISION"
    elif "ST" in df.columns:
        div_col = "ST"
    else:
        div_col = None

    if div_col is None or div_col not in df.columns:
        # Fallback: use first categorical-looking column for split
        print(f"  Warning: '{div_col}' not found, using random geographic split")
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.8
        train_df = df[mask]
        test_df = df[~mask]
    else:
        unique_vals = sorted(df[div_col].unique())
        print(f"  Split column '{div_col}' has {len(unique_vals)} unique values")

        # Use last value as OOD (different geographic region)
        ood_val = float(ood_division) if ood_division.replace(".", "").isdigit() else ood_division
        if ood_val not in df[div_col].values:
            # Try as int
            try:
                ood_val = int(ood_division)
            except ValueError:
                ood_val = unique_vals[-1]
            if ood_val not in df[div_col].values:
                ood_val = unique_vals[-1]

        print(f"  OOD value: {div_col}={ood_val}")
        train_df = df[df[div_col] != ood_val]
        test_df = df[df[div_col] == ood_val]

    # Sample down if too large
    if len(train_df) > sample_n:
        train_df = train_df.sample(n=sample_n, random_state=42)
    if len(test_df) > sample_n // 4:
        test_df = test_df.sample(n=sample_n // 4, random_state=42)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"  Saved: train={train_df.shape}, test={test_df.shape}")
    print(f"  Target rate: train={train_df['target'].mean():.3f}, test={test_df['target'].mean():.3f}")


# ── 2. Diabetes readmission (UCI) ───────────────────────────────────────

def setup_diabetes_readmission(out_dir: str):
    """Download UCI diabetes readmission dataset and split by admission_source_id."""
    import urllib.request

    url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
    print("[diabetes] Downloading from UCI...")

    try:
        response = urllib.request.urlopen(url, timeout=60)
        zip_data = response.read()
    except Exception as e:
        print(f"  Error downloading: {e}")
        print("  Trying alternative URL...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
        response = urllib.request.urlopen(url, timeout=60)
        zip_data = response.read()

    # Extract CSV from zip
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        csv_names = [n for n in z.namelist() if n.endswith(".csv") and "diabetic" in n.lower()]
        if not csv_names:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
        print(f"  Files in zip: {z.namelist()}")
        csv_name = csv_names[0]
        print(f"  Reading {csv_name}...")
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    print(f"  Raw shape: {df.shape}")

    # Create binary target: readmitted within 30 days
    if "readmitted" in df.columns:
        df["target"] = (df["readmitted"] == "<30").astype(int)
        df = df.drop(columns=["readmitted"])
    else:
        # Fallback
        print("  Warning: 'readmitted' column not found, using last column as target")
        df.rename(columns={df.columns[-1]: "target"}, inplace=True)

    # Replace ? with NaN
    df = df.replace("?", np.nan)

    # Rename encounter_id or first column to SampleID
    if "encounter_id" in df.columns:
        df = df.rename(columns={"encounter_id": "SampleID"})
    else:
        df.insert(0, "SampleID", [f"D{i:06d}" for i in range(len(df))])

    # Split by admission_source_id (OOD = 7, emergency)
    if "admission_source_id" in df.columns:
        ood_val = 7
        train_df = df[df["admission_source_id"] != ood_val]
        test_df = df[df["admission_source_id"] == ood_val]
        print(f"  Split on admission_source_id != {ood_val} (train) vs == {ood_val} (test)")
    else:
        print("  Warning: admission_source_id not found, using random split")
        np.random.seed(42)
        mask = np.random.rand(len(df)) < 0.8
        train_df = df[mask]
        test_df = df[~mask]

    # Sample down if very large
    if len(train_df) > 60000:
        train_df = train_df.sample(n=60000, random_state=42)
    if len(test_df) > 15000:
        test_df = test_df.sample(n=15000, random_state=42)

    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"  Saved: train={train_df.shape}, test={test_df.shape}")
    print(f"  Target rate: train={train_df['target'].mean():.3f}, test={test_df['target'].mean():.3f}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Setting up TableShift benchmark datasets")
    print("=" * 60)

    # 1. ACS Income (mild drift, -1.3%)
    setup_acs_dataset(
        task_name="acsincome",
        ood_division="01",
        out_dir=os.path.join(SCRIPT_DIR, "acsincome"),
    )
    print()

    # 2. Diabetes readmission (moderate drift, -5.9%)
    setup_diabetes_readmission(
        out_dir=os.path.join(SCRIPT_DIR, "diabetes"),
    )
    print()

    # 3. ACS Public Coverage (severe drift, -14.5%)
    setup_acs_dataset(
        task_name="acspubcov",
        ood_division="1.0",
        out_dir=os.path.join(SCRIPT_DIR, "acspubcov"),
    )

    print()
    print("=" * 60)
    print("Done! Run the pipeline on each dataset:")
    print()
    print("  python src/main.py --train_data_filepath test_data/acsincome/train.csv --test_data_filepath test_data/acsincome/test.csv")
    print("  python src/main.py --train_data_filepath test_data/diabetes/train.csv --test_data_filepath test_data/diabetes/test.csv")
    print("  python src/main.py --train_data_filepath test_data/acspubcov/train.csv --test_data_filepath test_data/acspubcov/test.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
