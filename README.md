# NAISC Singtel 2026 — Adaptive Drift Intelligence Challenge

**Team Name:** junyiding

## Quick Start (Docker)

```bash
docker build -t naisc . && docker run --rm naisc

# Build once (or after code changes)                                                                                                                                                                                                                                                docker build -t naisc .                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                    
# Run as many times as you want — no reinstall
  docker run --cpus=2 -v ${PWD}/stress_test_data:/app/stress_test_data naisc python ./src/main.py --train_data_filepath stress_test_data/train.csv --test_data_filepath stress_test_data/test.csv
```

## Setup (without Docker)

```bash
pip install -r requirements.txt
```

## Run

```bash
python ./src/main.py --train_data_filepath <train_data_filepath> --test_data_filepath <test_data_filepath>
```

Example with public data:

```bash
python ./src/main.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
```

## Output

- Console: drift detection summary, runtime, AU-PRC metrics
- `prediction.csv`: CustomerID and probability scores
- `model.joblib`: trained LightGBM model

## Dashboard

Interactive drift visualization dashboard:

```bash
python -m streamlit run src/dashboard.py
```

Opens in browser at `http://localhost:8501`.
