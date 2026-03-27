"""
Split a large CSV into train.csv and test.csv (80/20 split).
Reads and writes in chunks to avoid memory issues.

Usage:
    python split_train_test.py
"""

import polars as pl
import numpy as np
import os
import time

# ─────────────── CONFIG ───────────────
INPUT_PATH = "synthetic_2M.csv"
TRAIN_PATH = "train_2M.csv"
TEST_PATH = "test_2M.csv"
TEST_RATIO = 0.2
CHUNK_SIZE = 250_000
RANDOM_SEED = 42
# ──────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
train_written = 0
test_written = 0
header_done = False
start = time.time()

batches = pl.scan_csv(INPUT_PATH).collect_batches(chunk_size=CHUNK_SIZE)
for i, chunk in enumerate(batches):

    mask = rng.random(len(chunk)) >= TEST_RATIO
    chunk_with_mask = chunk.with_columns(pl.Series("_mask", mask))
    train_chunk = chunk_with_mask.filter(pl.col("_mask")).drop("_mask")
    test_chunk = chunk_with_mask.filter(~pl.col("_mask")).drop("_mask")

    with open(TRAIN_PATH, "wb" if not header_done else "ab") as f:
        train_chunk.write_csv(f, include_header=not header_done)
    with open(TEST_PATH, "wb" if not header_done else "ab") as f:
        test_chunk.write_csv(f, include_header=not header_done)

    train_written += mask.sum()
    test_written += (~mask).sum()
    header_done = True

    total = train_written + test_written
    elapsed = time.time() - start
    print(f"  Chunk {i+1}  |  train: {train_written:,}  test: {test_written:,}  |  {total:,} rows  |  {elapsed:.0f}s")

print(f"\nDone! train: {train_written:,} rows  |  test: {test_written:,} rows")
print(f"  {TRAIN_PATH}: {os.path.getsize(TRAIN_PATH)/1e9:.2f} GB")
print(f"  {TEST_PATH}: {os.path.getsize(TEST_PATH)/1e9:.2f} GB")
