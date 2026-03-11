# Synthetic Dataset Generator

Generates a synthetic dataset with 10M rows and 500 features for ML benchmarking. Modeled after a telecom churn-prediction schema with a mix of numerical, categorical, time-based, and sparse columns. Writes in chunks to keep memory usage low.

## Setup

```bash
pip install numpy pandas pyarrow
```

## Run

```bash
python generate_dataset.py
```

## Config

Edit the top of `generate_dataset.py` to change row count, chunk size, output format (csv/parquet), or file path.
