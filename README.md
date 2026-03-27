# NAISC Singtel 2026 — Adaptive Drift Intelligence Challenge

**Team Name:** junyiding

## Quick Start (Docker)

```bash
docker build -t naisc .; docker run --rm --cpus=2 naisc
```

## Setup (without Docker)

```bash
pip install -r requirements.txt
```

## Run

```bash
python ./src/main.py --train_data_filepath <train_data_filepath> --test_data_filepath <test_data_filepath>
```

Example with stress test data:

```bash
python ./src/main.py --train_data_filepath stress_test_data/train.csv --test_data_filepath stress_test_data/test.csv
```

## Output

- Console: drift detection summary, runtime, AU-PRC metrics
- `prediction.csv`: CustomerID and probability scores
- `model.joblib`: trained LightGBM model

## Dashboard

Interactive drift visualization dashboard:

```bash
streamlit run src/dashboard.py
```

Opens in browser at `http://localhost:8501`.
