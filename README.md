# NAISC 2026 — Singtel Track: Adaptive Drift Intelligence Challenge

**Team:** junyiding  
**Result:** 🏅 Finalist  
**Competition:** National AI Student Challenge (NAISC) 2026, Singtel track — *Guarding Model Integrity in a Shifting Data World*  
**Challenge period:** 6 March – 17 April 2026 (submitted 12PM SGT, 17 April 2026)

An automated pipeline that **detects**, **quantifies** and **mitigates** data drift between train and test sets, then trains a LightGBM classifier under the challenge's fixed-hyperparameter constraint. Scores are reported as AU-PRC. See [`specs.md`](specs.md) for the official challenge brief.

## Approach

| Stage | Module | What it does |
| :--- | :--- | :--- |
| Profiling | `src/profiler.py` | Loads data (Polars), aligns train/test columns, infers feature types, downcasts dtypes |
| Drift detection | `src/drift_detector.py` | KS test, PSI and Wasserstein distance for numerical features; Chi², JSD and Cramér's V for categorical ones. Benjamini–Hochberg correction, adjustment for small samples, and a missingness check that raises severity |
| Mitigation | `src/mitigation.py` | Cleans data quality issues (e.g. abbreviation mapping), then tries fixes in order for each drifted feature: clipping, log, quantile binning/matching, rank transform, rare-category grouping, frequency encoding, robust scaling |
| Feature engineering | `src/feature_engineer.py` | Builds derived features and prunes near-duplicates |
| Modelling | `src/model.py` | Fixed-hyperparameter LightGBM, with a weight tournament that picks the best sample weighting (recency, domain or blend) |
| Dashboard | `src/dashboard.py` | Streamlit and Plotly UI for exploring drift and results |

On the public data, mitigation raised test AU-PRC from **0.7576** (baseline) to **0.8727**. [`CHANGELOG.md`](CHANGELOG.md) logs the 34 iterations of experiments, including the ones we reverted. [`report.pdf`](report.pdf) is the write-up we submitted.

## Quick Start (Docker)

The public dataset is not committed. Place `train.csv` and `test.csv` in `public_data/` first.

```bash
docker build -t naisc .
docker run --rm naisc
```

Run against another dataset without rebuilding:

```bash
docker run --rm --cpus=2 -v ${PWD}/stress_test_data:/app/stress_test_data naisc \
  python ./src/main.py --train_data_filepath stress_test_data/train.csv --test_data_filepath stress_test_data/test.csv
```

Stress-test data can be generated with `tests/generate_stress_test.py`.

## Setup (without Docker)

Python 3.12 recommended.

```bash
pip install -r requirements.txt
```

## Run

```bash
python ./src/main.py --train_data_filepath <train_data_filepath> --test_data_filepath <test_data_filepath>
```

Example:

```bash
python ./src/main.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
```

## Output

- **Console:** drift detection and mitigation summary, runtime breakdown, AU-PRC metrics
- **`prediction.csv`:** `CustomerID` and predicted probability scores
- **`model.joblib`:** the trained LightGBM model

## Dashboard

```bash
python -m streamlit run src/dashboard.py
```

Opens at `http://localhost:8501`.
