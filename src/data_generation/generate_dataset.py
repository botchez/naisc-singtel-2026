"""
Generate a synthetic dataset with 10,000,000 rows and 500 features.

Mandatory columns (from original dataset):
  - CustomerID   (primary key, unique hex string)
  - Month        (time column)
  - ChurnStatus  (target column)

The remaining 497 features are a mix of:
  - Numerical     (continuous floats, integers, skewed distributions)
  - Categorical   (random labels, yes/no flags, multi-level categories)
  - Time-based    (datetimes, durations, day-of-week, hour-of-day)
  - Sparse        (mostly NaN / zero with occasional values)

Writes output in chunks to avoid memory issues.
Supports both CSV and Parquet output.

Usage:
    pip install numpy pandas pyarrow
    python generate_dataset.py

Adjust TOTAL_ROWS, OUTPUT_FORMAT, and OUTPUT_PATH below as needed.
"""

import numpy as np
import pandas as pd
import uuid
import time
import os
import sys
from datetime import datetime, timedelta

# ─────────────────────────── CONFIG ───────────────────────────
TOTAL_ROWS = 2_000_000
CHUNK_SIZE = 200_000          # rows per chunk (tune to your RAM)
OUTPUT_FORMAT = "csv"     # "csv" or "parquet"
OUTPUT_PATH = "synthetic_2M.csv"  # change extension if using csv
RANDOM_SEED = 32
# ──────────────────────────────────────────────────────────────

NUM_CHUNKS = (TOTAL_ROWS + CHUNK_SIZE - 1) // CHUNK_SIZE

# ── Category pools (reused across chunks) ─────────────────────
GENDERS = ["Male", "Female", "Non-Binary"]
YES_NO = ["Yes", "No"]
COUNTRIES = ["United States", "Canada", "United Kingdom", "Germany", "Australia"]
STATES = [
    "California", "Texas", "New York", "Florida", "Illinois",
    "Ohio", "Georgia", "Michigan", "Pennsylvania", "Arizona",
]
OFFERS = ["Offer A", "Offer B", "Offer C", "Offer D", "Offer E", ""]
INTERNET_TYPES = ["Fiber Optic", "DSL", "Cable", "Satellite", ""]
CONTRACT_TYPES = ["Month-to-Month", "One Year", "Two Year"]
PAYMENT_MODES = ["Credit Card", "Bank Withdrawal", "Mailed Check", "Electronic Check"]
CONNECTIVITY = ["Yes", "No", ""]
MONTHS = [f"25-{m}" for m in ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"]]

# Extra categorical pools for generated features
COLOR_POOL = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
TIER_POOL = ["bronze", "silver", "gold", "platinum", "diamond"]
REGION_POOL = ["north", "south", "east", "west", "central"]
STATUS_POOL = ["active", "inactive", "pending", "suspended", "cancelled"]
CHANNEL_POOL = ["web", "mobile", "store", "phone", "email", "chat"]
DEVICE_POOL = ["desktop", "laptop", "tablet", "phone", "smart_tv", "iot"]
PLAN_POOL = ["basic", "standard", "premium", "enterprise"]
LANG_POOL = ["en", "es", "fr", "de", "zh", "ja", "ko", "pt", "ar", "hi"]
BOOL_SPARSE = ["Y", "N", ""]


def build_feature_specs():
    """
    Define 500 columns total:
      - 45 original-schema columns
      - 455 synthetic columns (numerical / categorical / time / sparse)
    Returns a list of (col_name, generator_function) tuples.
    """
    specs = []  # each entry: (name, func(rng, n) -> array)

    # ── 1. Original 45 columns ────────────────────────────────
    # These mirror the schema of the uploaded train/test CSVs.

    specs.append(("CustomerID",       lambda rng, n: np.array([uuid.uuid4().hex[:16] for _ in range(n)])))
    specs.append(("UserGender",       lambda rng, n: rng.choice(GENDERS, n)))
    specs.append(("UserAge",          lambda rng, n: rng.integers(18, 95, n)))
    specs.append(("YoungAdultFlag",   lambda rng, n: np.where(rng.integers(18, 95, n) < 30, "Yes", "No")))
    specs.append(("RetireeStatus",    lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("Married",          lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("Dependents",       lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("NumberofDependents", lambda rng, n: rng.integers(0, 6, n)))
    specs.append(("Country",          lambda rng, n: rng.choice(COUNTRIES, n)))
    specs.append(("State",            lambda rng, n: rng.choice(STATES, n)))
    specs.append(("LocationCity",     lambda rng, n: np.array(["City_" + str(i) for i in rng.integers(1, 5000, n)])))
    specs.append(("AreaCode",         lambda rng, n: rng.integers(10000, 99999, n)))
    specs.append(("Latitude",         lambda rng, n: np.round(rng.uniform(25.0, 48.0, n), 6)))
    specs.append(("Longitude",        lambda rng, n: np.round(rng.uniform(-125.0, -70.0, n), 6)))
    specs.append(("Population",       lambda rng, n: rng.integers(100, 500000, n)))
    specs.append(("ReferredaFriend",  lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("NumberofReferrals", lambda rng, n: rng.integers(0, 15, n)))
    specs.append(("TenureinMonths",   lambda rng, n: rng.integers(1, 72, n)))
    specs.append(("Offer",            lambda rng, n: rng.choice(OFFERS, n)))
    specs.append(("VoiceService",     lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("AvgMonthlyLongDistanceCharges", lambda rng, n: np.round(rng.uniform(0, 60, n), 2)))
    specs.append(("AdditionalLines",  lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("ConnectivityType", lambda rng, n: rng.choice(CONNECTIVITY, n)))
    specs.append(("InternetType",     lambda rng, n: rng.choice(INTERNET_TYPES, n)))
    specs.append(("DataUsageAvg",     lambda rng, n: rng.integers(0, 50, n)))
    specs.append(("CyberSecuritySvc", lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("CloudStorageSvc",  lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("HardwareInsurance", lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("PrioritySupport",  lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("VideoSvc_A",       lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("VideoSvc_B",       lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("AudioSvc",         lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("UnlimitedData",    lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("Contract",         lambda rng, n: rng.choice(CONTRACT_TYPES, n)))
    specs.append(("DigitalInvoicing", lambda rng, n: rng.choice(YES_NO, n)))
    specs.append(("TransactionMode",  lambda rng, n: rng.choice(PAYMENT_MODES, n)))
    specs.append(("MonthlyCharge",    lambda rng, n: np.round(rng.uniform(18, 130, n), 2)))
    specs.append(("TotalCharges",     lambda rng, n: np.round(rng.uniform(50, 9000, n), 2)))
    specs.append(("TotalRefunds",     lambda rng, n: np.round(rng.uniform(0, 50, n), 2)))
    specs.append(("TotalExtraDataCharges", lambda rng, n: np.round(rng.uniform(0, 100, n), 2)))
    specs.append(("TotalLongDistanceCharges", lambda rng, n: np.round(rng.uniform(0, 4000, n), 2)))
    specs.append(("TotalRevenue",     lambda rng, n: np.round(rng.uniform(50, 12000, n), 2)))
    specs.append(("CustomerLifetimeValue", lambda rng, n: rng.integers(1000, 10000, n)))
    specs.append(("ChurnStatus",      lambda rng, n: rng.choice(["Yes", "No"], n, p=[0.27, 0.73])))
    specs.append(("Month",            lambda rng, n: rng.choice(MONTHS, n)))

    assert len(specs) == 45, f"Expected 45 original cols, got {len(specs)}"

    # ── 2. Synthetic features (455 more → 500 total) ─────────
    # Distribute as: ~150 numerical, ~120 categorical, ~60 time-based, ~125 sparse
    feat_id = 0

    # --- NUMERICAL features (150) ---
    for i in range(50):
        name = f"num_uniform_{feat_id}"
        lo, hi = -1000 + i * 10, 1000 + i * 20
        specs.append((name, lambda rng, n, lo=lo, hi=hi: np.round(rng.uniform(lo, hi, n), 4)))
        feat_id += 1

    for i in range(30):
        name = f"num_normal_{feat_id}"
        mu, sigma = i * 5.0, max(1.0, i * 2.0)
        specs.append((name, lambda rng, n, mu=mu, sigma=sigma: np.round(rng.normal(mu, sigma, n), 4)))
        feat_id += 1

    for i in range(20):
        name = f"num_int_{feat_id}"
        lo, hi = 0, 100 * (i + 1)
        specs.append((name, lambda rng, n, lo=lo, hi=hi: rng.integers(lo, hi, n)))
        feat_id += 1

    for i in range(15):
        name = f"num_exponential_{feat_id}"
        scale = 1.0 + i
        specs.append((name, lambda rng, n, s=scale: np.round(rng.exponential(s, n), 4)))
        feat_id += 1

    for i in range(15):
        name = f"num_lognormal_{feat_id}"
        mu, sigma = i * 0.3, 0.5 + i * 0.1
        specs.append((name, lambda rng, n, mu=mu, sigma=sigma: np.round(rng.lognormal(mu, sigma, n), 4)))
        feat_id += 1

    for i in range(10):
        name = f"num_poisson_{feat_id}"
        lam = 1 + i * 3
        specs.append((name, lambda rng, n, lam=lam: rng.poisson(lam, n)))
        feat_id += 1

    for i in range(10):
        name = f"num_beta_{feat_id}"
        a, b = 0.5 + i * 0.2, 0.5 + i * 0.3
        specs.append((name, lambda rng, n, a=a, b=b: np.round(rng.beta(a, b, n), 6)))
        feat_id += 1

    # --- CATEGORICAL features (120) ---
    for i in range(20):
        name = f"cat_binary_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(YES_NO, n)))
        feat_id += 1

    for i in range(15):
        name = f"cat_color_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(COLOR_POOL, n)))
        feat_id += 1

    for i in range(15):
        name = f"cat_tier_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(TIER_POOL, n)))
        feat_id += 1

    for i in range(15):
        name = f"cat_region_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(REGION_POOL, n)))
        feat_id += 1

    for i in range(15):
        name = f"cat_status_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(STATUS_POOL, n)))
        feat_id += 1

    for i in range(10):
        name = f"cat_channel_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(CHANNEL_POOL, n)))
        feat_id += 1

    for i in range(10):
        name = f"cat_device_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(DEVICE_POOL, n)))
        feat_id += 1

    for i in range(10):
        name = f"cat_plan_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(PLAN_POOL, n)))
        feat_id += 1

    for i in range(10):
        name = f"cat_lang_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(LANG_POOL, n)))
        feat_id += 1

    # --- TIME-BASED features (60) ---
    BASE_TS = int(datetime(2020, 1, 1).timestamp())
    RANGE_TS = int(datetime(2025, 12, 31).timestamp()) - BASE_TS

    for i in range(20):
        name = f"time_datetime_{feat_id}"
        specs.append((name, lambda rng, n: pd.to_datetime(
            rng.integers(BASE_TS, BASE_TS + RANGE_TS, n), unit="s"
        )))
        feat_id += 1

    for i in range(10):
        name = f"time_hour_{feat_id}"
        specs.append((name, lambda rng, n: rng.integers(0, 24, n)))
        feat_id += 1

    for i in range(10):
        name = f"time_dayofweek_{feat_id}"
        specs.append((name, lambda rng, n: rng.choice(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n)))
        feat_id += 1

    for i in range(10):
        name = f"time_duration_sec_{feat_id}"
        specs.append((name, lambda rng, n: rng.integers(0, 86400, n)))
        feat_id += 1

    for i in range(10):
        name = f"time_month_num_{feat_id}"
        specs.append((name, lambda rng, n: rng.integers(1, 13, n)))
        feat_id += 1

    # --- SPARSE features (125) ---
    for i in range(50):
        name = f"sparse_float_{feat_id}"
        sparsity = 0.85 + (i % 10) * 0.01  # 85-94% NaN
        def _gen_sparse_float(rng, n, sp=sparsity):
            vals = rng.uniform(-100, 100, n).astype(np.float64)
            mask = rng.random(n) < sp
            vals[mask] = np.nan
            return np.round(vals, 4)
        specs.append((name, _gen_sparse_float))
        feat_id += 1

    for i in range(40):
        name = f"sparse_int_{feat_id}"
        sparsity = 0.80 + (i % 15) * 0.01  # 80-94% zero
        def _gen_sparse_int(rng, n, sp=sparsity):
            vals = rng.integers(1, 500, n)
            mask = rng.random(n) < sp
            vals[mask] = 0
            return vals
        specs.append((name, _gen_sparse_int))
        feat_id += 1

    for i in range(35):
        name = f"sparse_cat_{feat_id}"
        sparsity = 0.88 + (i % 8) * 0.01  # 88-95% empty
        pool = COLOR_POOL if i % 3 == 0 else (TIER_POOL if i % 3 == 1 else STATUS_POOL)
        def _gen_sparse_cat(rng, n, sp=sparsity, pool=pool):
            vals = rng.choice(pool, n)
            mask = rng.random(n) < sp
            vals[mask] = ""
            return vals
        specs.append((name, _gen_sparse_cat))
        feat_id += 1

    total_features = len(specs)
    assert total_features == 500, f"Expected 500 features, got {total_features}"
    return specs


def generate_chunk(specs, rng, chunk_size):
    """Generate one chunk as a DataFrame."""
    data = {}
    for col_name, gen_func in specs:
        data[col_name] = gen_func(rng, chunk_size)
    return pd.DataFrame(data)


def main():
    print("=" * 65)
    print(f"  Synthetic Dataset Generator")
    print(f"  Rows: {TOTAL_ROWS:,}  |  Features: 500  |  Format: {OUTPUT_FORMAT}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Chunk size: {CHUNK_SIZE:,}  |  Chunks: {NUM_CHUNKS}")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_SEED)
    specs = build_feature_specs()

    start_time = time.time()
    rows_written = 0

    if OUTPUT_FORMAT == "csv":
        # Write header first, then append chunks
        header_written = False
        for chunk_idx in range(NUM_CHUNKS):
            remaining = TOTAL_ROWS - rows_written
            this_chunk = min(CHUNK_SIZE, remaining)
            if this_chunk <= 0:
                break

            df = generate_chunk(specs, rng, this_chunk)
            df.to_csv(
                OUTPUT_PATH,
                mode="a" if header_written else "w",
                header=not header_written,
                index=False,
            )
            header_written = True
            rows_written += this_chunk

            elapsed = time.time() - start_time
            pct = rows_written / TOTAL_ROWS * 100
            rate = rows_written / elapsed if elapsed > 0 else 0
            eta = (TOTAL_ROWS - rows_written) / rate if rate > 0 else 0
            print(f"  Chunk {chunk_idx+1}/{NUM_CHUNKS}  |  "
                  f"{rows_written:>12,} rows ({pct:5.1f}%)  |  "
                  f"{rate:,.0f} rows/s  |  ETA: {eta/60:.1f} min")

    elif OUTPUT_FORMAT == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        for chunk_idx in range(NUM_CHUNKS):
            remaining = TOTAL_ROWS - rows_written
            this_chunk = min(CHUNK_SIZE, remaining)
            if this_chunk <= 0:
                break

            df = generate_chunk(specs, rng, this_chunk)
            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PATH, table.schema, compression="snappy")

            writer.write_table(table)
            rows_written += this_chunk

            elapsed = time.time() - start_time
            pct = rows_written / TOTAL_ROWS * 100
            rate = rows_written / elapsed if elapsed > 0 else 0
            eta = (TOTAL_ROWS - rows_written) / rate if rate > 0 else 0
            print(f"  Chunk {chunk_idx+1}/{NUM_CHUNKS}  |  "
                  f"{rows_written:>12,} rows ({pct:5.1f}%)  |  "
                  f"{rate:,.0f} rows/s  |  ETA: {eta/60:.1f} min")

        if writer:
            writer.close()
    else:
        print(f"Unknown format: {OUTPUT_FORMAT}. Use 'csv' or 'parquet'.")
        sys.exit(1)

    elapsed = time.time() - start_time
    file_size = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)
    print("=" * 65)
    print(f"  Done!  {rows_written:,} rows written in {elapsed/60:.1f} minutes")
    print(f"  File size: {file_size:.2f} GB  →  {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
