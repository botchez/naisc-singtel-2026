# Changelog

## [2026-03-06 23:55:02 SGT] Initial Pipeline

**Branch:** `junwen` | **Commit:** `66ff7c9`

### Changes

- Built initial data loading, drift detection, and baseline LightGBM model pipeline.
- Implemented per-column statistical drift tests (PSI, KS, Chi-squared, JSD, Wasserstein).
- Trained baseline LightGBM model with fixed competition hyperparameters.

### Files Added

- `src/main.py`, `src/profiler.py`, `src/drift_detector.py`, `src/model.py`, `src/utils.py`

---

## [2026-03-07 01:39:18 SGT] Dashboard & Model Update

**Branch:** `junwen` | **Commits:** `f1117b3`, `5e26ff7`, `a1b344e`

### Changes

- Added Streamlit dashboard for drift detection visualization.
- Updated model output artifacts.
- Updated team name in README.

### Files Added/Modified

- `src/dashboard.py` (new)
- `model.joblib`, `README.md`

---

## [2026-03-07 21:53:52 SGT] Drift Mitigation Pipeline

**Branch:** `junwen` | **Commit:** `cdf44de`

### Changes

- Introduced `src/mitigation.py` with three mitigation stages: temporal sample reweighting, case/hyphen normalization, and temporal column dropping.
- Pipeline now trains both baseline and mitigated models, compares AU-PRC, and selects the best.

### Files Added/Modified

- `src/mitigation.py` (new)
- `src/main.py`, `src/model.py`

---

## [2026-03-08 00:29:33 SGT] Drift-Informed Feature Selection

**Branch:** `junwen` | **Commits:** `04e2fdc`, `7349978`

### Changes

- Replaced temporal reweighting with drift-informed feature selection — drops all high-severity drift features instead of reweighting samples.
- New mitigation pipeline: normalize data quality (city abbreviation mapping + lowercase/hyphen) → re-run drift detection on normalized data → drop high-drift features.
- Simplified pipeline to single mitigated model with fixed `prediction.csv` and `model.joblib` output paths.
- Removed baseline model training and model comparison/selection logic.
- Added combined drift detection & mitigation summary table with columns, types, methods, and runtime.
- Fixed white-on-white text in dashboard dark theme.

### Results

- Test AU-PRC improved from 0.7380 to 0.8375 (+0.0995).
- Generalization gap shrank from 0.1388 to 0.0227.

### Files Modified

- `src/main.py`, `src/mitigation.py`, `src/model.py`, `src/dashboard.py`

---

## [2026-03-08 11:08:05 SGT] Reporting & UX Enhancements

**Branch:** `junwen` | **Commits:** `be26a24`, `6faf983`, `9cc2af1`, `fbe13fb`

### Changes

- Added settings file for pipeline configuration.
- Enhanced drift mitigation summary output and logging.
- Added model performance `print` function and updated metrics return in `evaluate_and_save`.
- Added ASCII typewriter animation for console output.
- Enhanced model performance summary with baseline vs mitigated delta comparison and improved formatting.

### Files Modified

- `src/main.py`, `src/model.py`, `src/utils.py`

---

## [2026-03-08 13:07:08 SGT] Progressive Mitigation Strategies

**Branch:** `junwen` | **Commits:** `ea48c29`, `04a371e`

### Changes

- Added progressive per-feature mitigation: strategies are tried mildest-to-strongest (clip → log → clip+log → quantile bin / rare-group / cyclical encode); the first one that reduces severity below "high" wins.
- Updated normalization to include city abbreviation mapping.
- Refactored code structure across all modules for improved readability and maintainability.

### Files Modified

- `src/mitigation.py`, `src/main.py`, `src/profiler.py`, `src/drift_detector.py`, `src/model.py`, `src/utils.py`

---

## [2026-03-08 14:18:34 SGT] Auto-Detection & Column Alignment

**Branch:** `junwen` | **Commit:** `2a69a50`

### Changes

- Added column alignment between train and test datasets via fuzzy matching.
- Added auto-detection for ID and target columns (no hardcoded column names).
- Improved abbreviation mapping and feature type detection.

### Files Modified

- `src/profiler.py`, `src/utils.py`

---

## [2026-03-08 14:58:20 SGT] Multivariate Shift Analysis

**Branch:** `junwen` | **Commit:** `25d52b9`

### Changes

- Enhanced drift detection with multivariate shift analysis using a domain classifier.
- Added density ratio weighting for sample reweighting.
- Added specifications document.

### Files Added/Modified

- `specs.md` (new)
- `src/drift_detector.py`, `src/mitigation.py`

---

## [2026-03-08 15:52:35 SGT] Simplify Drift Mitigation

**Branch:** `junwen` | **Commit:** `5e18435`

### Changes

- Removed multivariate shift detection and density ratio weighting (over-engineering for this use case).
- Simplified `apply_all_mitigations` return values.
- Updated model training to exclude sample weights.

### Files Modified

- `src/drift_detector.py`, `src/mitigation.py`, `src/model.py`, `src/main.py`

---

## [2026-03-08 22:50:21 SGT] Domain Classifier Weights, Intra-Training Drift & Feature Engineering

**Branch:** `junwen` | **Commit:** `714427e`

### Changes

- Generated natural-language drift descriptions (skewness, mean shift, range explosion, new categories).
- Renamed mitigation strategies to match challenge vocabulary (Seasonality Matching, Feature Scaling, Drop Feature).
- Extracted OOF P(test) from domain classifier for sample weighting with auto-selection (power dampening).
- Added intra-training temporal drift detection (early vs late month split).
- Added recency weights with exponential decay; combined domain+recency weight candidates.
- Auto-select best weight variant (domain, recency, combined, or uniform) via test AU-PRC validation.
- Created `feature_engineer.py`: missing indicators, frequency encoding, ratio features (all dynamic).
- Fixed table wrapping for long drift descriptions.

### Files Added/Modified

- `src/feature_engineer.py` (new)
- `src/main.py`, `src/drift_detector.py`, `src/mitigation.py`, `src/model.py`, `src/utils.py`

---

## [2026-03-09 00:24:18 SGT] Robustness Overhaul: Drift Detection & Mitigation Gaps

**Branch:** `junwen`

### Model Performance

| Metric         | Baseline | Mitigated | Delta   |
|----------------|----------|-----------|---------|
| AU-PRC (train) | 0.8904   | 0.8896    | -0.0008 |
| AU-PRC (test)  | 0.7317   | 0.8571    | +0.1254 |

**Winner:** Domain(p=0.3) sample weights (CV AU-PRC=0.8603)

### Runtime

| Stage               | Time   |
|---------------------|--------|
| Drift Detection     | 1.92s  |
| Drift Mitigation    | 9.12s  |
| Feature Engineering | 1.29s  |
| Model Training      | 57.58s |
| **Total**           | 69.91s |

### Changes

9 robustness gaps were identified and addressed to harden the pipeline for arbitrary/hidden datasets.

#### 1. Expanded Categorical Mitigation (`mitigation.py`)
- Added `frequency_encode_mitigation()` — replaces categories with training-set frequency.
- Categorical high-drift features now try 3 escalating strategies before dropping: group rare categories, frequency encode, frequency encode + numerical strategies (clip/rank/quantile).
- Previously only tried 1 strategy (group rare) before dropping.

#### 2. Post-Engineering Drift Check (`main.py`, `feature_engineer.py`)
- `engineer_features()` now returns the list of new feature names.
- After engineering, each new feature is drift-checked: high-drift features are dropped, moderate-drift features get clipping applied.
- Prevents unchecked drift from ratio, aggregation, and frequency-encoded features leaking into the model.

#### 3. Missingness Drift Detection (`drift_detector.py`)
- Added `_missingness_severity()` and `_escalate_severity()` helpers.
- Both `detect_numerical_drift` and `detect_categorical_drift` now compute train/test null percentages and escalate severity when missingness patterns diverge (>20pp diff = high, >5pp = moderate).
- Returned dicts now include `train_null_pct` and `test_null_pct`.

#### 4. Small Dataset Safety (`utils.py`, `drift_detector.py`, `mitigation.py`, `model.py`, `main.py`)
- Added `safe_n_splits(n_samples, desired, min_per_fold)` utility function.
- PSI bins adapt to sample size: `min(10, n // 20)`.
- Quantile binning, target encoding CV, domain classifier CV, and weight tournament CV all adapt fold counts dynamically.
- Weight tournament skipped entirely for datasets with <50 rows, defaulting to uniform weights.

#### 5. Conditional Normalization (`mitigation.py`)
- Lowercasing is now drift-aware: computes JSD before/after lowercasing each column, only applies if JSD does not increase.
- On this run, 2 columns were correctly skipped (`PrioritySupport`, `DigitalInvoicing`) where lowercasing would have increased drift.

#### 6. High-Cardinality Chi-Squared Fix (`drift_detector.py`)
- For columns with >50 categories, groups into top-30 + `_other` bucket before running chi-squared test.
- JSD is still computed on the full distribution (always reliable).
- Prevents spurious low p-values from sparse contingency tables.

#### 7. Progressive Moderate-Drift Treatment (`mitigation.py`)
- Moderate-drift numerical features now try multiple strategies: clip, log, clip+log, rank transform. Accepts the best improvement even if severity stays moderate.
- Moderate-drift categorical features try group rare, then frequency encode.
- Previously only tried a single strategy per type and only accepted if severity dropped to "low".

#### 8. `_clip_then_log` Negative Fix (`mitigation.py`)
- Now shifts values by `abs(min) + epsilon` when clipped values are still negative, ensuring `log_transform` always receives non-negative input.
- Previously, `_clip_then_log` was effectively identical to plain `log_transform` for distributions with negative tails.

#### 9. Reduced Weight Tournament (`main.py`)
- Candidate grid reduced from ~70+ to 16 strategies (4 domain powers, 3 recency decays, 4 multiplicative combos, 4 additive blends).
- Removed squared/sqrt domain weight variants (monotonic transforms, unlikely to add value).
- Added early stopping: if after 2 CV folds a candidate's mean AU-PRC is 0.02+ below the current best, remaining folds are skipped.

### Files Modified

- `src/utils.py` — added `safe_n_splits()` helper
- `src/drift_detector.py` — missingness drift, adaptive PSI bins, high-cardinality chi2 fix, adaptive domain classifier folds
- `src/mitigation.py` — frequency encode mitigation, conditional normalization, progressive moderate treatment, clip_then_log fix, adaptive quantile bins
- `src/feature_engineer.py` — returns new feature names for post-engineering drift check
- `src/model.py` — adaptive target encoding CV folds
- `src/main.py` — post-engineering drift check, reduced weight tournament with early stopping, adaptive CV folds
