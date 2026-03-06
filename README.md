# NAISC Singtel 2026 — Adaptive Drift Intelligence Challenge

**Team Name:** [TODO: Add team name]

## Setup

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
