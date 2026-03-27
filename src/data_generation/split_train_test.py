"""
Split a large CSV into train.csv and test.csv (80/20 split).
Reads and writes in chunks to avoid memory issues.

Usage:
    python split_train_test.py
"""

import pandas as pd
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

for i, chunk in enumerate(pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE)):
    mask = rng.random(len(chunk)) >= TEST_RATIO

    chunk[mask].to_csv(TRAIN_PATH, mode="a" if header_done else "w", header=not header_done, index=False)
    chunk[~mask].to_csv(TEST_PATH, mode="a" if header_done else "w", header=not header_done, index=False)

    train_written += mask.sum()
    test_written += (~mask).sum()
    header_done = True

    total = train_written + test_written
    elapsed = time.time() - start
    print(f"  Chunk {i+1}  |  train: {train_written:,}  test: {test_written:,}  |  {total:,} rows  |  {elapsed:.0f}s")

print(f"\nDone! train: {train_written:,} rows  |  test: {test_written:,} rows")
print(f"  {TRAIN_PATH}: {os.path.getsize(TRAIN_PATH)/1e9:.2f} GB")
print(f"  {TEST_PATH}: {os.path.getsize(TEST_PATH)/1e9:.2f} GB")
