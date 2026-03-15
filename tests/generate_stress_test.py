"""
Generate synthetic stress test datasets by upscaling public data.

Usage:
    python tests/generate_stress_test.py --scale medium   # ~500K rows, 150 cols
    python tests/generate_stress_test.py --scale large    # ~2M rows, 200 cols
    python tests/generate_stress_test.py --rows 1000000 --extra_cols 100

Output goes to stress_test_data/ directory.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

PRESETS = {
    "small": {"rows": 200_000, "extra_cols": 50},
    "medium": {"rows": 500_000, "extra_cols": 105},  # ~150 total cols
    "large": {"rows": 2_000_000, "extra_cols": 155},  # ~200 total cols
}

def estimate_memory_gb(n_rows, n_cols):
    """Rough estimate: mix of float64 (8B) and object (avg ~50B)."""
    return n_rows * n_cols * 20 / 1e9  # ~20 bytes avg per cell


def generate_extra_numerical(n_rows, n_cols, rng):
    """Generate random numerical features with varied distributions."""
    data = {}
    for i in range(n_cols):
        dist_type = i % 5
        col_name = f"NumFeature_{i+1:03d}"
        if dist_type == 0:
            data[col_name] = rng.normal(50, 15, n_rows)
        elif dist_type == 1:
            data[col_name] = rng.exponential(10, n_rows)
        elif dist_type == 2:
            data[col_name] = rng.uniform(0, 100, n_rows)
        elif dist_type == 3:
            # Sparse - mostly zeros
            vals = rng.exponential(5, n_rows)
            vals[rng.random(n_rows) < 0.7] = 0
            data[col_name] = vals
        else:
            # Integer-like
            data[col_name] = rng.integers(0, 50, n_rows)
    return pd.DataFrame(data)


def generate_extra_categorical(n_rows, n_cols, rng):
    """Generate random categorical features."""
    data = {}
    for i in range(n_cols):
        col_name = f"CatFeature_{i+1:03d}"
        card = rng.choice([3, 5, 10, 20, 50])  # varied cardinality
        categories = [f"cat_{j}" for j in range(card)]
        data[col_name] = rng.choice(categories, n_rows)
    return pd.DataFrame(data)


def upscale_dataframe(df, target_rows, rng):
    """Upscale a dataframe by resampling with jitter on numerical columns."""
    n_repeats = (target_rows // len(df)) + 1

    # Sample with replacement to reach target size
    indices = rng.integers(0, len(df), target_rows)
    upscaled = df.iloc[indices].reset_index(drop=True)

    # Add jitter to numerical columns to avoid exact duplicates
    num_cols = upscaled.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        std = upscaled[col].std()
        if std > 0 and not np.isnan(std):
            noise = rng.normal(0, std * 0.05, target_rows)  # 5% noise
            upscaled[col] = upscaled[col] + noise

    # Regenerate unique CustomerIDs
    upscaled["CustomerID"] = [f"stress_{i:08x}" for i in range(target_rows)]

    return upscaled


def inject_drift(df_test, drift_fraction=0.3, rng=None):
    """Inject synthetic drift into test set to make it realistic."""
    num_cols = df_test.select_dtypes(include=[np.number]).columns.tolist()

    # Shift a fraction of numerical features
    n_drift = max(1, int(len(num_cols) * drift_fraction))
    drift_cols = rng.choice(num_cols, n_drift, replace=False)

    for col in drift_cols:
        shift_type = rng.choice(["mean_shift", "variance_change", "skew"])
        if shift_type == "mean_shift":
            std = df_test[col].std()
            if std > 0 and not np.isnan(std):
                df_test[col] = df_test[col] + std * rng.uniform(0.5, 2.0)
        elif shift_type == "variance_change":
            mean = df_test[col].mean()
            df_test[col] = mean + (df_test[col] - mean) * rng.uniform(1.5, 3.0)
        else:  # skew
            df_test[col] = np.abs(df_test[col])

    # Shift some categorical features
    cat_cols = df_test.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ("CustomerID", "Month")]
    n_cat_drift = max(1, int(len(cat_cols) * 0.2))
    if cat_cols:
        drift_cat_cols = rng.choice(cat_cols, min(n_cat_drift, len(cat_cols)), replace=False)
        for col in drift_cat_cols:
            # Introduce a new category for some rows
            mask = rng.random(len(df_test)) < 0.15
            df_test.loc[mask, col] = "new_category"

    return df_test


def main():
    parser = argparse.ArgumentParser(description="Generate stress test data")
    parser.add_argument("--scale", choices=["small", "medium", "large"], default=None)
    parser.add_argument("--rows", type=int, default=None, help="Target train rows")
    parser.add_argument("--extra_cols", type=int, default=None, help="Extra columns to add")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test/train ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="stress_test_data")
    args = parser.parse_args()

    if args.scale:
        preset = PRESETS[args.scale]
        target_rows = args.rows or preset["rows"]
        extra_cols = args.extra_cols or preset["extra_cols"]
    else:
        target_rows = args.rows or 500_000
        extra_cols = args.extra_cols or 105

    test_rows = int(target_rows * args.test_ratio)
    rng = np.random.default_rng(args.seed)

    # Estimate memory
    total_cols_est = 45 + extra_cols
    mem_est = estimate_memory_gb(target_rows + test_rows, total_cols_est)
    print(f"=== Stress Test Generator ===")
    print(f"Train rows:  {target_rows:>12,}")
    print(f"Test rows:   {test_rows:>12,}")
    print(f"Total cols:  {total_cols_est:>12,} (45 original + {extra_cols} extra)")
    print(f"Est memory:  {mem_est:>11.1f} GB")
    print()

    if mem_est > 16:
        print(f"WARNING: Estimated {mem_est:.1f}GB memory usage. You may run out of RAM.")
        resp = input("Continue? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit(0)

    # Load public data
    print("Loading public data...")
    src_dir = os.path.join(os.path.dirname(__file__), "..", "public_data")
    train_src = pd.read_csv(os.path.join(src_dir, "train.csv"))
    test_src = pd.read_csv(os.path.join(src_dir, "test.csv"))
    print(f"  Source train: {train_src.shape}, test: {test_src.shape}")

    # Upscale
    print(f"Upscaling train to {target_rows:,} rows...")
    t0 = time.time()
    train_up = upscale_dataframe(train_src, target_rows, rng)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"Upscaling test to {test_rows:,} rows...")
    t0 = time.time()
    test_up = upscale_dataframe(test_src, test_rows, rng)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Add extra features
    n_extra_num = int(extra_cols * 0.7)
    n_extra_cat = extra_cols - n_extra_num
    print(f"Adding {n_extra_num} numerical + {n_extra_cat} categorical extra features...")

    t0 = time.time()
    train_extra_num = generate_extra_numerical(target_rows, n_extra_num, rng)
    train_extra_cat = generate_extra_categorical(target_rows, n_extra_cat, rng)
    train_up = pd.concat([train_up, train_extra_num, train_extra_cat], axis=1)

    test_extra_num = generate_extra_numerical(test_rows, n_extra_num, rng)
    test_extra_cat = generate_extra_categorical(test_rows, n_extra_cat, rng)
    test_up = pd.concat([test_up, test_extra_num, test_extra_cat], axis=1)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Inject drift into test set
    print("Injecting synthetic drift into test set...")
    test_up = inject_drift(test_up, drift_fraction=0.3, rng=rng)

    # Remove ChurnStatus from test (like real test set)
    if "ChurnStatus" in test_up.columns:
        test_up = test_up.drop(columns=["ChurnStatus"])

    print(f"Final shapes - train: {train_up.shape}, test: {test_up.shape}")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    print(f"Writing train to {train_path}...")
    t0 = time.time()
    train_up.to_csv(train_path, index=False)
    train_size = os.path.getsize(train_path) / 1e9
    print(f"  Done in {time.time()-t0:.1f}s ({train_size:.2f} GB)")

    print(f"Writing test to {test_path}...")
    t0 = time.time()
    test_up.to_csv(test_path, index=False)
    test_size = os.path.getsize(test_path) / 1e9
    print(f"  Done in {time.time()-t0:.1f}s ({test_size:.2f} GB)")

    print(f"\n=== Done ===")
    print(f"Run the pipeline with:")
    print(f"  python src/main.py --train_data_filepath {train_path} --test_data_filepath {test_path}")


if __name__ == "__main__":
    main()
