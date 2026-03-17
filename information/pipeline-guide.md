# NAISC Pipeline: Comprehensive Technical Guide

This document is a complete walkthrough of the automated drift detection and mitigation pipeline for the NAISC Singtel 2026 challenge. It covers every function, its inputs, outputs, and the reasoning behind each design decision.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Entry Point: main.py](#2-entry-point-mainpy)
3. [Stage 1: Data Ingestion & Profiling (profiler.py)](#3-stage-1-data-ingestion--profiling-profilerpy)
4. [Stage 1b: Decision Sample & Pipeline Cap (main.py)](#4-stage-1b-decision-sample--pipeline-cap-mainpy)
5. [Stage 2: Drift Detection (drift_detector.py)](#5-stage-2-drift-detection-drift_detectorpy)
6. [Stage 3: Drift Mitigation (mitigation.py)](#6-stage-3-drift-mitigation-mitigationpy)
7. [Stage 4: Feature Engineering (feature_engineer.py)](#7-stage-4-feature-engineering-feature_engineerpy)
8. [Stage 5: Preprocessing & Encoding (model.py)](#8-stage-5-preprocessing--encoding-modelpy)
9. [Stage 6: Model Training & Weight Tournament (main.py + model.py)](#9-stage-6-model-training--weight-tournament-mainpy--modelpy)
10. [Stage 7: Evaluation & Output (model.py + main.py)](#10-stage-7-evaluation--output-modelpy--mainpy)
11. [Shared Utilities (utils.py)](#11-shared-utilities-utilspy)
12. [Data Flow Diagram](#12-data-flow-diagram)
13. [Key Design Principles](#13-key-design-principles)

---

## 1. Pipeline Overview

The pipeline transforms raw train/test CSVs into a churn prediction model through eight stages. The core insight is that the hidden evaluation dataset will contain **data drift** — distributional shifts between training and test data — and the pipeline must detect and counteract these shifts automatically.

All differentiation comes from data handling. The LightGBM model uses **fixed hyperparameters** that cannot be changed:

```python
LGBMClassifier(verbosity=-1, objective="binary", is_unbalance=True, random_state=42, importance_type="gain")
```

The evaluation metric is **AU-PRC** (Area Under Precision-Recall Curve), which rewards accurate ranking of the minority class (churners).

**High-level data flow:**

```
train.csv, test.csv
    |
    v
[1] Load & Profile  -->  feature_types dict
    |
[1b] Decision Sample + Pipeline Cap
    |
[2] Drift Detection  -->  drift_df (per-feature severity) + month_rank (recency ordering)
    |
[3] Drift Mitigation  -->  train_m, test_m (cleaned) + drop_cols + shift_info (domain classifier)
    |
[4] Feature Engineering  -->  train_m, test_m (enriched with new features)
    |
[5] Preprocessing  -->  X_train, y_train, X_test (model-ready matrices)
    |
[6] Weight Tournament  -->  best_weights (optimal sample weighting scheme)
    |
[7] Final Model + Evaluation  -->  prediction.csv, model.joblib
```

---

## 2. Entry Point: main.py

`main()` (`main.py:252`) is the orchestrator. It calls into each module sequentially and passes state between stages via local variables. Each stage is timed with a `Timer` context manager, and summary tables are printed at the end.

**Key orchestration decisions:**

- **Two models are always trained**: a _baseline_ (no mitigation, uniform weights) and a _mitigated_ (full pipeline). Both are saved, and the mitigated version is copied to the final `prediction.csv` / `model.joblib`.
- **State carries forward**: `feature_types` (from Stage 1) flows through every subsequent stage as the universal schema reference. `drift_df` (from Stage 2) drives Stage 3's mitigation strategy selection. `shift_info` (from Stage 3) provides domain classifier probabilities for Stage 6's weighting.
- **Display functions** (`print_drift_mitigation_summary`, `print_multivariate_shift_summary`, `print_runtime_summary`) are called only after all computation completes, so the console output is clean and grouped.

---

## 3. Stage 1: Data Ingestion & Profiling (profiler.py)

### 3.1 Loading Data

#### `load_data(train_path, test_path) -> (train_df, test_df)` — `profiler.py:28`

| Parameter | Type | Description |
|---|---|---|
| `train_path` | `str` | Filesystem path to training CSV |
| `test_path` | `str` | Filesystem path to test CSV |
| **Returns** | `tuple[DataFrame, DataFrame]` | Loaded and downcasted DataFrames |

**Logic:**
1. Reads both CSVs with `pd.read_csv(low_memory=False)`.
2. **Duplicate column handling**: If any column names appear more than once, appends `_dup1`, `_dup2`, etc. This prevents silent column overwrites that pandas would otherwise allow.
3. **Memory downcasting** via `_downcast_dataframe()` (`profiler.py:6`): Converts `float64` -> `float32`, and `int64` -> the smallest integer type that fits the actual data range (`int8`/`int16`/`int32`). This is critical for the 10M-row datasets — downcasting can halve memory usage.
4. Logs the shape and memory before/after.

**Rationale:** The hidden eval dataset may have up to 10M rows and 500 features. Without downcasting, a 10M x 500 float64 DataFrame would use ~40 GB. With float32, it's ~20 GB. The int downcasting is even more aggressive — a column with values 0-12 (like month) shrinks from 8 bytes/row to 1 byte/row.

### 3.2 Column Alignment

#### `align_train_test_columns(train, test) -> (train, test_aligned)` — `profiler.py:118`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Training data |
| `test` | `DataFrame` | Test data |
| **Returns** | `tuple[DataFrame, DataFrame]` | Aligned DataFrames (test columns renamed to match train) |

**Logic:**
1. Finds columns in train but not in test, and vice versa.
2. For mismatched columns, normalizes names via `_normalize_col_name()` (`profiler.py:63`) — lowercase, strip underscores/spaces/hyphens — and checks for fuzzy matches. If `test` has `"customer_id"` and `train` has `"CustomerID"`, they'll match.
3. Renames test columns to train's naming convention.
4. Logs warnings for columns that remain unmatched (expected: target column is typically train-only).

**Rationale:** The hidden dataset may use different column naming conventions (e.g., snake_case vs CamelCase). This ensures column alignment without hardcoding any names.

### 3.3 Feature Type Detection

#### `detect_feature_types(df, id_col, target_col, temporal_col, test_df) -> feature_types` — `profiler.py:160`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `DataFrame` | — | Training data for type inference |
| `id_col` | `str` | `"CustomerID"` | Hint for ID column name |
| `target_col` | `str` | `"ChurnStatus"` | Hint for target column name |
| `temporal_col` | `str` | `"Month"` | Hint for temporal column name |
| `test_df` | `DataFrame \| None` | `None` | Test data (only columns in both are classified) |
| **Returns** | `dict[str, list[str]]` | `{"numerical": [...], "categorical": [...], "temporal": [...], "id": [...], "target": [...]}` |

**Logic:**

1. **Auto-detection with fallback**: Each of `id_col`, `target_col`, `temporal_col` first tries exact match. If not found:
   - **ID**: `_auto_detect_id_column()` (`profiler.py:82`) — tries pattern matching against `_ID_PATTERNS` (`profiler.py:68`), then looks for columns ending in "id" with >90% unique values.
   - **Target**: `_auto_detect_target_column()` (`profiler.py:102`) — tries `_TARGET_PATTERNS` (`profiler.py:74`), then looks for suggestive column names containing "target", "label", etc.
   - **Temporal**: Tries fuzzy normalized name match (`profiler.py:196`), then falls back to `_looks_temporal()` (`profiler.py:304`) heuristic which tries parsing values against 11 date formats.

2. **Column filtering**: When `test_df` is provided, only columns present in **both** datasets are classified as features. Train-only columns (besides id/target) are logged and skipped. This prevents the model from learning features that won't exist at prediction time.

3. **Per-column classification**:
   - Skip all-null columns (uninformative).
   - Skip zero-variance columns (e.g., `Country` and `State` in public data, which have only one unique value).
   - Skip near-zero-variance columns (>99% same value) — these add noise without predictive power.
   - If matched as `temporal_col`, classify as temporal.
   - If non-numeric but matches `_looks_temporal()` heuristic, classify as temporal.
   - If non-numeric but >80% parseable as numbers (`_looks_numeric()`, `profiler.py:295`), classify as numerical (handles columns like `TotalCharges` which have numeric values with a few blank strings).
   - Otherwise: numeric dtype -> numerical, object dtype -> categorical.

4. **Large dataset optimization**: If the DataFrame exceeds 100K rows, per-column stats (nunique, value_counts) are computed on a 100K-row random sample.

**Output structure example:**
```python
{
    "numerical": ["TenureinMonths", "MonthlyCharge", "TotalRevenue", ...],
    "categorical": ["Contract", "InternetType", "LocationCity", ...],
    "temporal": ["Month"],
    "id": ["CustomerID"],
    "target": ["ChurnStatus"],
}
```

This `feature_types` dict is the **single source of truth** for the entire pipeline. Every subsequent stage references it rather than hardcoding column names.

### 3.4 Data Profiling

#### `profile_data(df, feature_types, label) -> DataFrame` — `profiler.py:320`

| Parameter | Type | Description |
|---|---|---|
| `df` | `DataFrame` | Dataset to profile |
| `feature_types` | `dict` | Type classification from `detect_feature_types` |
| `label` | `str` | Label for logging (e.g., "train", "test") |
| **Returns** | `DataFrame` | Summary with columns: column, type, dtype, n_unique, null_pct, sample_value |

Produces a summary of each column's characteristics. Used primarily for logging — the console output shows counts like `43 columns (17 num, 23 cat, 1 temporal)`.

---

## 4. Stage 1b: Decision Sample & Pipeline Cap (main.py)

### 4.1 Decision Sample

For datasets larger than 100K rows, many operations (drift detection, feature selection, weight tournament) don't benefit from using all rows — statistical tests converge well before 100K. The pipeline creates a **single decision sample** that's reused across these operations (`main.py:285-303`).

**Logic:**
1. Parse the temporal column (if any) via `_parse_month_labels()` (`utils.py:26`) to get month-period labels for stratified sampling.
2. If `len(train) > 100,000`: create `_decision_idx` using `stratified_subsample_idx()` (`utils.py:39`) — 100K rows, preserving month distribution.
3. Create `train_decision = train.iloc[_decision_idx]` for drift detection (`main.py:303`).

**Rationale:** Using the same 100K sample across all "decision" operations (drift detection, feature importance, tournament) ensures consistency. Stratified sampling preserves the temporal distribution, which matters because drift is temporal.

### 4.2 Pipeline Cap

For extremely large datasets, the full pipeline (mitigation, feature engineering, model training) must complete within 10 minutes (`main.py:306-342`).

**Logic:**
1. Compute `_max_pipeline_rows = 480 / (5.75e-7 * n_cols)` (`main.py:308-311`). This formula is calibrated from benchmarks: at 2M rows x 200 cols, the pipeline takes ~230s.
2. If `len(train) > _max_pipeline_rows`: subsample train using stratified temporal sampling (`main.py:316-320`), then **recompute** `_month_vals`, `_decision_idx`, and `train_decision` on the capped data (`main.py:323-336`) — the pre-cap indices are now invalid.

**Example scaling:**
| Dataset | Max Rows | Cap Needed? |
|---|---|---|
| 70K x 45 (public) | ~18.5M | No |
| 2M x 200 | ~4.2M | No |
| 10M x 500 | ~1.67M | Yes, capped to 1.67M |

---

## 5. Stage 2: Drift Detection (drift_detector.py)

Drift detection identifies features whose distributions differ between train and test. This is the diagnostic foundation: Stage 3's mitigation strategies are driven by these results.

### 5.1 Train vs Test Drift

#### `run_drift_analysis(train, test, feature_types) -> drift_df` — `drift_detector.py:464`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Training data (decision sample) |
| `test` | `DataFrame` | Test data (subsampled if >100K) |
| `feature_types` | `dict` | Feature type classification |
| **Returns** | `DataFrame` | One row per feature: column, feature_type, severity (high/moderate/low), statistical metrics, drift_description |

**Logic:**
1. Builds a task list of `(column, type)` pairs from feature_types.
2. For each feature, calls `detect_numerical_drift()` or `detect_categorical_drift()`.
3. Generates a natural-language drift description via `generate_drift_description()`.
4. For >20 features, parallelizes via `joblib.Parallel` with threading (`drift_detector.py:490-494`).
5. Applies **Benjamini-Hochberg FDR correction** via `_apply_bh_correction()` (`drift_detector.py:392`) across all p-values, then re-classifies severity using corrected p-values.
6. Sorts output by severity (high first) (`drift_detector.py:511`).

### 5.2 Numerical Drift Detection

#### `detect_numerical_drift(train_col, test_col) -> dict` — `drift_detector.py:107`

| Parameter | Type | Description |
|---|---|---|
| `train_col` | `Series` | Training column values |
| `test_col` | `Series` | Test column values |
| **Returns** | `dict` | `{ks_stat, ks_pval, psi, wasserstein, wasserstein_norm, iqr_ratio, tail_divergence, severity, ...}` |

**Statistical tests (each catches different drift patterns):**

| Test | What It Catches | How |
|---|---|---|
| **KS test** (Kolmogorov-Smirnov) | Any distributional difference | Maximum vertical distance between CDFs. P-value indicates significance. |
| **PSI** (Population Stability Index) | Binned distribution changes | `compute_psi()` (`drift_detector.py:27`). Divides range into quantile bins, compares bin proportions. PSI > 0.25 = significant shift. |
| **Wasserstein distance** | Mean/location shifts | "Earth mover's distance" — minimum work to transform one distribution into another. Normalized by train std for scale-invariance. |
| **IQR ratio** | Spread changes | `test_IQR / train_IQR`. Ratio > 2 or < 0.5 = significant spread change. |
| **Tail divergence** | Extreme value shifts | Max shift in 1st/99th percentiles, normalized by train std. Catches tail behavior changes missed by histogram-based PSI. |

**Subsampling:** Both columns are subsampled to 50K rows via `_subsample_array()` (`drift_detector.py:11`) before testing. The cap is set by `_MAX_DRIFT_SAMPLE` (`drift_detector.py:8`). Beyond this, additional rows add negligible statistical power but cost O(n log n) compute.

**Severity classification** (`_numerical_severity()`, `drift_detector.py:257`):
- **High**: PSI >= 0.25, or (PSI >= 0.15 and KS p < 0.001), or (Wasserstein_norm >= 1.0 and PSI >= 0.1), or (IQR ratio > 2x and PSI >= 0.1), or (tail_div > 2 and PSI >= 0.05).
- **Moderate**: PSI >= 0.1, or (KS p < 0.05 and KS_D >= 0.05), or (tail_div > 1 and PSI >= 0.03).
- **Low**: Everything else.

**Small-sample correction** (`_small_sample_factor()`, `drift_detector.py:244`)**:** A `ss_factor` widens PSI thresholds when samples are small (harmonic mean < 5000), preventing false positives from noisy histogram estimates.

### 5.3 Categorical Drift Detection

#### `detect_categorical_drift(train_col, test_col) -> dict` — `drift_detector.py:168`

| Parameter | Type | Description |
|---|---|---|
| `train_col` | `Series` | Training column values |
| `test_col` | `Series` | Test column values |
| **Returns** | `dict` | `{chi2_stat, chi2_pval, jsd, cramers_v, unseen_in_test, severity, ...}` |

**Statistical tests:**

| Test | What It Catches | How |
|---|---|---|
| **Chi-squared test** | Any categorical distribution change | Contingency table test. For high-cardinality (>50 categories), groups rare categories into `_other` to avoid sparse-table inflation. |
| **JSD** (Jensen-Shannon Divergence) | Symmetric distributional divergence | `compute_jsd()` (`drift_detector.py:61`). Average of KL divergences in both directions. Range [0, ln(2)]. More robust than chi-squared for unbalanced categories. |
| **Cramer's V** | Sample-size-normalized effect size | `compute_cramers_v()` (`drift_detector.py:50`). `sqrt(chi2 / (n * (k-1)))`. Unlike chi2 p-value, V is independent of sample size — a p < 0.001 on 100K rows might still be a tiny effect. |

**Severity classification** (`_categorical_severity()`, `drift_detector.py:291`):
- **High**: JSD >= 0.1, or (Cramer's V >= 0.3 and JSD >= 0.05).
- **Moderate**: JSD >= 0.05, or chi2 p < 0.001, or Cramer's V >= 0.15.
- **Low**: Everything else.

### 5.4 Drift Description Generation

#### `generate_drift_description(train_col, test_col, feature_type, drift_metrics) -> str` — `drift_detector.py:309`

Produces a human-readable characterization of the drift, e.g.:

> "Feature demonstrates greater left-skewness in test set compared to training set; mean shifts higher in test set"

For numerical features, it checks:
- **Skewness shift** (>0.3 skew difference)
- **Mean shift** (>0.15 std shift)
- **Range expansion/contraction** (>1.3x or <0.7x)
- **Variance change** (>1.3x or <0.7x)

For categorical/temporal features, it checks:
- **Unseen categories** in test
- **Dominant category shift** (mode changes)
- **Proportion shifts** (JSD >= 0.05)

### 5.5 Benjamini-Hochberg FDR Correction

#### `_apply_bh_correction(drift_df, ss_factor) -> drift_df` — `drift_detector.py:392`

When testing 40+ features simultaneously, some will appear significant by chance. The BH procedure controls the **false discovery rate** — the expected proportion of false positives among declared significant results.

**Logic:**
1. Collects all p-values (KS for numerical, chi2 for categorical).
2. Sorts p-values and applies the BH formula via `_bh_correct()` (`drift_detector.py:70`): `adjusted_p = raw_p * n / rank`.
3. Stores corrected p-values in `ks_pval_bh` / `chi2_pval_bh` columns.
4. Re-classifies severity using corrected p-values (with the same `ss_factor` for consistency).
5. Re-applies missingness escalation via `_missingness_severity()` (`drift_detector.py:86`) and `_escalate_severity()` (`drift_detector.py:100`) — so missingness drift isn't lost when severity is overwritten.

### 5.6 Intra-Training Drift Detection

#### `detect_intra_training_drift(train, feature_types) -> (intra_drift_df, month_rank)` — `drift_detector.py:724`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Full (post-cap) training data |
| `feature_types` | `dict` | Feature type classification |
| **Returns** | `tuple[DataFrame, Series \| None]` | Intra-training drift results + per-row month rank ordering |

**Logic:**
1. Finds the temporal column and parses it into month periods (tries 11 date formats).
2. Sorts unique months chronologically, splits into early (first half) and late (second half).
3. Runs the same drift detection between early and late halves — this reveals **temporal drift within the training data itself**.
4. Returns only high/moderate-drift features.
5. Builds `month_rank` (`drift_detector.py:818-820`): a Series mapping each row to its month's ordinal rank (0 = earliest month, N-1 = latest month). This is used by `compute_recency_weights()` in Stage 6.

**Rationale:** If a feature is shifting within training data, it's likely to shift further in the test data (which is from a later time period). The `month_rank` enables recency weighting — giving more importance to recent training rows that are more representative of test conditions.

### 5.7 Multivariate Shift Detection

#### `detect_multivariate_shift(train, test, feature_types, extra_drop_cols, n_splits) -> dict` — `drift_detector.py:516`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Mitigated training data |
| `test` | `DataFrame` | Mitigated test data |
| `feature_types` | `dict` | Feature type classification |
| `extra_drop_cols` | `list[str] \| None` | Columns to exclude (dropped features) |
| `n_splits` | `int` | CV folds (default 3) |
| **Returns** | `dict` | `{auc, fold_aucs, shift_detected, train_oof_proba, top_shift_features}` |

This is a **domain classifier** approach. Instead of testing each feature independently, it asks: "Can a model tell whether a row came from train or test?"

**Logic:**
1. Prepares features: drops id/target/extra_drop (`drift_detector.py:535-538`), caps at 100 features by variance (`drift_detector.py:544-553`), encodes categoricals as integers (`drift_detector.py:570-576`), fills NaN with median (`drift_detector.py:577-586`).
2. Concatenates train and test, labels train as 0 and test as 1.
3. Trains a lightweight LightGBM (50 trees, `drift_detector.py:608-615`) with **stratified 3-fold CV** (`drift_detector.py:602`).
4. Computes **out-of-fold AUC** (area under ROC curve for domain classification).
   - AUC ~ 0.5 = no shift (model can't distinguish train from test).
   - AUC ~ 1.0 = extreme shift (train and test are trivially separable).
   - AUC > 0.65 = shift detected.
5. **Extracts per-training-row P(test)** (`drift_detector.py:636-650`): the probability that each training row "looks like" a test row. If train was subsampled for the domain classifier, the last fold's classifier scores the full training set (using saved encoding mappings from `_cat_maps` and `_fill_vals`, `drift_detector.py:568-569`).
6. Logs top-5 features driving the shift (by feature importance gain).

**Rationale:** Univariate tests can miss drift that only appears in feature interactions. The domain classifier captures **multivariate covariate shift**. The per-row P(test) probabilities become sample weights in Stage 6 — training rows that resemble test data are upweighted.

---

## 6. Stage 3: Drift Mitigation (mitigation.py)

### 6.1 Orchestrator

#### `apply_all_mitigations(train, test, drift_df, feature_types) -> (train_m, test_m, drop_cols, mitigation_log, norm_log, shift_info)` — `mitigation.py:561`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Full training data |
| `test` | `DataFrame` | Full test data |
| `drift_df` | `DataFrame` | Drift results from Stage 2 |
| `feature_types` | `dict` | Feature type classification |
| **Returns** | | |
| `train_m` | `DataFrame` | Mitigated training data |
| `test_m` | `DataFrame` | Mitigated test data |
| `drop_cols` | `list[str]` | Columns to drop (all strategies failed) |
| `mitigation_log` | `list[dict]` | Per-feature log of strategy applied |
| `norm_log` | `dict[str, list[str]]` | Per-column normalization actions |
| `shift_info` | `dict` | Multivariate shift detection results |

**Sub-stages:**

```
3a. Data quality normalization (abbreviations, casing, hyphens)
3b. Re-check drift for normalized columns only
3c. Progressive mitigation for high-drift features
3d. Progressive mitigation for moderate-drift features (capped at top 50)
3e. Multivariate shift detection (domain classifier)
```

### 6.2 Data Quality Normalization (Stage 3a)

#### `normalize_data_quality(train, test, feature_types) -> (train_clean, test_clean, norm_log)` — `mitigation.py:113`

This stage fixes surface-level data quality issues that masquerade as drift.

**Three normalization steps:**

1. **Abbreviation detection** (`detect_abbreviation_mappings()`, `mitigation.py:72`):
   - Scans each categorical column for 2-3 character uppercase strings (e.g., "LA", "NYC", "SD").
   - Matches abbreviations to multi-word values whose word initials match (e.g., "LA" -> "Los Angeles").
   - Applies mapping to both train and test.
   - *Why this matters:* If train has "Los Angeles" but test has "LA", the model would treat them as different categories, losing signal.

2. **Conditional lowercase + hyphen removal** (`mitigation.py:144-183`):
   - For each categorical column, computes JSD **before** and **after** lowercasing + hyphen removal (`mitigation.py:167-174`).
   - Only applies if normalization reduces or maintains JSD (`mitigation.py:176`) — i.e., doesn't increase drift.
   - *Why conditional:* Some columns have meaningful case distinctions. Blindly lowercasing "Yes"/"No" -> "yes"/"no" wouldn't help. The JSD check ensures normalization only happens when it genuinely closes the gap.
   - *Example:* `Contract` has "Month-to-Month" in train vs "month-to-month" in test. Lowercasing + hyphen removal aligns them.

3. **Fuzzy matching for unseen test categories** (`mitigation.py:190-227`):
   - For low-cardinality categoricals (`_MAX_FUZZY_TRAIN_CATS=200`, `_MAX_FUZZY_UNSEEN=50`, `mitigation.py:194-195`), uses `difflib.get_close_matches()` to map unseen test values to their closest train equivalent (`mitigation.py:217-219`).
   - Two-pass: strict cutoff (0.6) then relaxed (0.45).
   - High-cardinality columns are skipped (`mitigation.py:210-211`) — fuzzy matching is O(unseen x train_cats) and only useful for typo-like mismatches.

### 6.3 Drift Re-check (Stage 3b)

After normalization, drift is rechecked only for modified columns (not all 500). This is an efficiency optimization — why re-test 480 unchanged columns? (`mitigation.py:583-611`)

**Logic:**
1. For each column in `norm_log`, re-runs `_analyze_single_feature()` (`drift_detector.py:451`).
2. Updates the `drift_df_norm` DataFrame in place (`mitigation.py:599-604`).
3. Re-applies BH correction and re-sorts by severity (`mitigation.py:606-610`).

### 6.4 Progressive Mitigation — High Drift (Stage 3c)

#### `mitigate_feature_progressive(train, test, col, feature_type, original_severity) -> (train, test, log_entry)` — `mitigation.py:435`

For each high-drift feature, tries a sequence of increasingly aggressive strategies. The first strategy that reduces severity below "high" wins. If all fail, the feature is dropped.

**Strategy ladders by feature type:**

**Temporal features:**
1. Cyclical encoding (sin/cos of month). Accept if both sin and cos components are below "high". If cyclical encoding fails -> drop. Temporal columns are inherently drifted (test has future months not in train), so dropping is the common outcome.

**Numerical features:**
1. Clip outliers (1st-99th percentile of train)
2. Log transform (log1p, requires non-negative)
3. Clip + Log (clip first, then log)
4. Robust scaling (median + IQR normalization)
5. Discretization (quantile binning using train edges)
6. Rank transform (percentile rank based on train CDF)

Each strategy is first evaluated on a 50K subsample via `_eval_subsample()` (`mitigation.py:8`). If it reduces severity, it's applied to full data (`mitigation.py:494-500`). This subsample-then-apply pattern saves computation when strategies fail.

**Categorical features:**
1. Group rare categories (below 1% frequency -> `_other`)
2. Frequency encoding (replace category with training frequency 0-1)
3. Frequency + clip / rank / quantile (combine frequency encoding with numerical strategies)

### 6.5 Progressive Mitigation — Moderate Drift (Stage 3d)

Same approach but with a softer goal: reduce severity to "low" if possible, otherwise reduce the drift metric (PSI for numerical, JSD for categorical) as much as possible.

**Numerical strategies for moderate drift:**
1. Gentle clipping
2. Log transform
3. Clip + Log
4. Robust scaling
5. **Quantile matching** (CDF-to-CDF mapping — aligns test distribution to train)
6. Rank transform

**Categorical strategies for moderate drift:**
1. Rare category grouping
2. Frequency encoding

The best-improving strategy is applied even if severity stays "moderate" — partial improvement still helps the model.

**Cap at 50** (`_MAX_MODERATE`, `mitigation.py:654`)**:** For datasets with hundreds of features, only the top-50 most-drifted moderate features are processed (sorted by PSI/JSD, `mitigation.py:655-658`). This prevents the mitigation stage from dominating runtime.

### 6.6 Mitigation Strategy Reference

| Strategy | Function | Location | Input | Output | Rationale |
|---|---|---|---|---|---|
| Median align | `median_align()` | `mitigation.py:234` | num, num | num, num | Shifts test median to match train median. Simple location correction. |
| Clip outliers | `clip_outliers()` | `mitigation.py:250` | num, num | num, num | Clips both to train's [1st, 99th] percentiles. Tames extreme values. |
| Log transform | `log_transform()` | `mitigation.py:259` | num, num | num, num | log1p, compresses right-skewed distributions. Only works for non-negative data. |
| Robust scaling | `robust_scale()` | `mitigation.py:345` | num, num | num, num | (x - median) / IQR using train stats. Robust to outliers unlike z-score. |
| Quantile binning | `quantile_bin()` | `mitigation.py:268` | num, num | int, int | Discretizes into bins defined by train quantile edges. Eliminates continuous drift. |
| Quantile matching | `quantile_match()` | `mitigation.py:376` | num, num | num, num | CDF-to-CDF mapping. Each test value -> its percentile rank in test -> mapped to corresponding train quantile. Preserves test rank order, aligns distributions. |
| Rank transform | `rank_transform()` | `mitigation.py:413` | num, num | float, float | Converts to percentile rank (0-1) based on train CDF. Distribution-agnostic. |
| Clip+Log | `_clip_then_log()` | `mitigation.py:363` | num, num | num, num | Clip outliers first, shift to non-negative, then log1p. Combines both strategies. |
| Group rare cats | `group_rare_categories()` | `mitigation.py:284` | cat, cat | cat, cat | Categories <1% frequency in train -> `_other`. Also maps unseen test categories to `_other`. |
| Frequency encode | `frequency_encode_mitigation()` | `mitigation.py:329` | cat, cat | float, float | Replace category with train-set frequency (0-1). Converts categorical -> numerical. |
| Cyclical encode | `encode_temporal_cyclical()` | `mitigation.py:298` | temporal, temporal | sin+cos, sin+cos | Parse month -> sin(2pi*m/12), cos(2pi*m/12). Captures seasonality without ordinal bias. |

### 6.7 Sample Weight Functions

These don't modify the data — they produce per-row weights used in Stage 6.

#### `compute_domain_weights(train_oof_proba, power) -> ndarray` — `mitigation.py:19`

Converts domain classifier P(test) into sample weights. Training rows that look more like test data get higher weight.

**Logic:**
1. Clip probabilities to [1e-6, 1-1e-6].
2. Apply power dampening: `weights = proba^power`. Power < 1 (default 0.5) compresses the weight distribution — preventing a few "very test-like" rows from dominating.
3. Cap extreme values at 99th percentile.
4. Normalize to mean=1.0 (so total effective sample size is preserved).

#### `compute_recency_weights(month_rank, decay) -> ndarray` — `mitigation.py:48`

Exponential decay weighting by temporal recency. Recent months get weight ~1.0; older months decay.

**Logic:**
1. `weights = exp(decay * (rank - max_rank))`. The most recent month gets exp(0) = 1.0. Older months decay exponentially.
2. Normalize to mean=1.0.

---

## 7. Stage 4: Feature Engineering (feature_engineer.py)

#### `engineer_features(train, test, feature_types, drop_cols) -> (train_enriched, test_enriched, new_feature_names)` — `feature_engineer.py:6`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Mitigated training data |
| `test` | `DataFrame` | Mitigated test data |
| `feature_types` | `dict` | Feature type classification |
| `drop_cols` | `list[str]` | Columns marked for dropping |
| **Returns** | `tuple[DataFrame, DataFrame, list[str]]` | Enriched DataFrames + list of new feature names |

All feature engineering is **dynamic** — no hardcoded column names. Features are derived from the `feature_types` dict and statistical properties of the data. A name collision guard (`existing_cols`, `feature_engineer.py:29`) prevents duplicate feature names across all stages.

### 7.1 Missing Value Indicators (`feature_engineer.py:32-50`)

For columns with >1% null rate, adds a binary `{col}_is_null` feature. This lets the model learn from missingness patterns — e.g., if `TotalCharges` is null for new customers with 0 tenure, that pattern is predictive.

### 7.2 Frequency Encoding (`feature_engineer.py:52-73`)

For high-cardinality categoricals (>10 unique values), adds `{col}_freq` — the training-set frequency of each category value. This provides a numerical representation that captures category popularity without the dimensionality explosion of one-hot encoding.

Low-cardinality categoricals (<= 10 unique) are handled later by target encoding in Stage 5.

### 7.3 Ratio Features (`feature_engineer.py:75-119`)

Finds the top-5 most correlated numerical pairs (|r| > 0.3) and creates `{col_a}_div_{col_b}` = a / (b + 1). The +1 prevents division by zero.

**Correlation cap** (`_MAX_CORR_COLS=50`, `feature_engineer.py:81`)**:** For datasets with >50 numerical features, only the top-50 by variance are used for the correlation matrix (avoiding O(m^2) computation on 500 features).

**Rationale:** Ratio features capture relationships that tree models can technically learn but require many splits to approximate. `TotalCharges / TotalRevenue` is a single feature expressing relative spending efficiency.

### 7.4 Aggregation Features (`feature_engineer.py:121-147`)

Groups `mean(numerical)` by `categorical`, creating features like `MonthlyCharge_mean_by_Contract`.

- Top 3 categoricals (by cardinality, 3-50 unique) x top 5 numericals = up to 15 features.
- Test values are mapped using training group means. Unseen test categories get the global mean.

**Rationale:** Aggregation features encode group-level patterns. If customers on "Month-to-Month" contracts have higher average MonthlyCharge, this feature makes that pattern directly accessible to the model.

### 7.5 Near-Duplicate Removal (`feature_engineer.py:152-193`)

After all features are added, computes the correlation matrix among all numerical features and drops one of any pair with |corr| > 0.98. When choosing which to drop, engineered features are preferred over originals (keep the raw signal, discard the derivative).

**Column cap** (`_MAX_DEDUP_COLS=200`, `feature_engineer.py:157`)**:** For >200 numerical features, only the top-200 by variance are checked (avoiding O(m^2) on 500+ columns).

### 7.6 Post-Engineering Drift Check (`main.py:363-375`)

Back in `main.py`, each newly engineered feature is tested for drift via `detect_numerical_drift()`. Any engineered feature with "high" drift is dropped — it would confuse the model more than help.

---

## 8. Stage 5: Preprocessing & Encoding (model.py)

### 8.1 Feature Preparation

#### `prepare_features(train, test, feature_types, extra_drop_cols) -> (X_train, y_train, X_test, customer_ids)` — `model.py:60`

| Parameter | Type | Description |
|---|---|---|
| `train` | `DataFrame` | Training data (original or mitigated) |
| `test` | `DataFrame` | Test data (original or mitigated) |
| `feature_types` | `dict` | Feature type classification |
| `extra_drop_cols` | `list[str] \| None` | Additional columns to exclude |
| **Returns** | | |
| `X_train` | `DataFrame` | Model-ready training features |
| `y_train` | `Series` | Binary target (0/1) |
| `X_test` | `DataFrame` | Model-ready test features |
| `customer_ids` | `list[str]` | Test row identifiers for prediction output |

**Processing steps:**

1. **Extract customer IDs** from test (`model.py:74-78`), fallback to row indices if ID column missing.

2. **Encode target** via `_encode_target()` (`model.py:10`): Handles "Yes"/"No", 1/0, True/False, "1"/"0", etc. Falls back to minority-class-is-positive heuristic for unknown binary labels (`model.py:38-42`).

3. **Column selection**: Only columns present in both train and test, minus id/target/drop_cols.

4. **Numeric coercion**: Columns classified as numerical but stored as object (e.g., `TotalCharges` with blank strings) are converted via `pd.to_numeric(errors="coerce")`.

5. **Dtype alignment**: If train and test disagree on a column's dtype, both are coerced to numeric.

6. **Temporal parsing** (`model.py:113-116`): Temporal columns are converted to month number (1-12) via `_parse_temporal_month()` (`model.py:48`).

7. **Categorical encoding** (split by cardinality):

   **High-cardinality (>10 unique) -> LightGBM native categorical** (`model.py:131-157`)**:**
   - Integer label encoding + pandas `category` dtype (`model.py:144-154`).
   - Unseen test categories get NaN (LightGBM handles missing values natively by learning the optimal split direction).
   - *Why native:* LightGBM's native categorical handling uses optimal histogram-based splits, which outperforms one-hot or target encoding for high-cardinality features.

   **Low-cardinality (<=10 unique) -> Target encoding** (`model.py:178-231`)**:**
   - K-fold encoding to prevent leakage (`model.py:220-226`): for each fold, category means are computed from out-of-fold rows only. Validation rows never see their own labels in their encoding.
   - **Bayesian smoothing** (`model.py:128`, `model.py:197`): `smoothed_mean = (count * raw_mean + m * global_mean) / (count + m)` with m=10. Categories with few samples shrink toward the global mean, preventing overfitting to small groups.
   - **Large dataset optimization** (`_MAX_TE_SAMPLE=200_000`, `model.py:160-176`): For >200K rows, uses a stratified 200K subsample for fold mean computation, but maps results onto all rows (`model.py:202-218`).
   - **Single-class guard** (`model.py:123-126`, `model.py:183-186`): If y_train has only one class (edge case in hidden eval), all categories get the global mean.

   Test set uses the full-data smoothed means (not fold-specific).

8. **NaN handling**: Numeric NaN is left as-is (LightGBM handles it natively). Non-numeric NaN is filled with mode.

### 8.2 Target Encoding: Why K-Fold?

Naive target encoding (computing category mean directly from training data) leaks the target — each row's encoding includes information from its own label. K-fold encoding fixes this:

```
For each fold:
  1. Compute category means using ONLY out-of-fold rows
  2. Assign these means to the held-out fold's rows
```

Each row's encoding comes from data it was never part of, eliminating leakage.

---

## 9. Stage 6: Model Training & Weight Tournament (main.py + model.py)

### 9.1 Baseline Model (`main.py:387-400`)

A model trained on unmitigated data with uniform weights, using the raw `train` and `test` DataFrames. Subsampled to 50K rows (`_MAX_BASELINE_SAMPLE`, `main.py:388`) for speed (it's only for comparison metrics). Saved as `prediction_baseline.csv` and `model_baseline.joblib`.

### 9.2 Feature Selection (`main.py:407-446`)

Using the mitigated data:
1. Train an initial model on decision sample if data is large (`main.py:412-417`).
2. **Drop zero-importance features** (`main.py:422-428`): Features that LightGBM never split on. These add noise without predictive value.
3. **Add top-feature interactions** (`main.py:430-444`): Take the top-8 features by importance (`main.py:433`), create all pairwise products (`col_a * col_b`, `main.py:436-442`). This gives the model explicit access to interaction effects. For public data, this creates ~21 interaction terms.

### 9.3 Weight Tournament (`main.py:448-603`)

The weight tournament finds the optimal blend of domain weights (from the domain classifier) and recency weights (from month rank). It evaluates candidates via `_eval_candidate()` (`main.py:484`) by training a model with each weight configuration and measuring test AU-PRC (or train AU-PRC if test labels unavailable).

**Two weight families are optimized:**

1. **Additive Blend** (`_build_blend()`, `main.py:508`): `w = alpha * domain^power + (1-alpha) * recency(decay)`
   - `alpha` controls the domain/recency mix
   - `power` dampens domain weights
   - `decay` controls recency steepness

2. **Geometric Mean** (`_build_geometric()`, `main.py:517`): `w = domain^(alpha*power) * recency^(1-alpha)`
   - Multiplicative combination — a row must score well on **both** dimensions to get high weight

**Optimization** (`_make_cached_objective()`, `main.py:539`)**:** Each family uses **Nelder-Mead** (simplex method) starting from `x0=[0.5, 0.7, 0.15]` (`main.py:567-568`), with at most 12 function evaluations. A parameter cache (`_eval_cache`, `main.py:530`) prevents duplicate model trains when the optimizer revisits rounded parameter values.

**Tournament structure:**
```
1. Evaluate Uniform (no weights) -> baseline score
2. Optimize Blend (Nelder-Mead, ~12 evals)
3. Optimize Geometric (Nelder-Mead, ~12 evals)
4. Pick overall winner across all ~25 candidates
```

**Fallback modes:**
- If only domain weights available (no temporal column): grid search over power values [0.3, 0.5, 0.7, 1.0].
- If only recency weights available (no domain shift): grid search over decay values [0.10, 0.20, 0.40].
- If neither: Uniform weights (no tournament).

**Subsampling** (`_TOURNEY_SAMPLE_SIZE=300_000`, `main.py:466`)**:** For datasets >300K rows, the tournament uses a stratified 300K-row subsample for each model train. The final model (after the winner is selected) is trained on full data.

**Dummy test labels guard** (`main.py:457-462`)**:** If test labels exist but have only 1 unique value (judges may provide dummy labels), `has_test_labels` is set to False and the tournament optimizes on train score instead.

### 9.4 Model Training

#### `train_model(X_train, y_train, sample_weight) -> LGBMClassifier` — `model.py:247`

| Parameter | Type | Description |
|---|---|---|
| `X_train` | `DataFrame` | Training features |
| `y_train` | `Series` | Binary target |
| `sample_weight` | `ndarray \| None` | Per-row weights (None = uniform) |
| **Returns** | `LGBMClassifier` | Trained model |

Uses the fixed hyperparameters. The only knob is `sample_weight`, which is how all the drift detection work manifests in the model — rows more representative of test conditions get higher weight.

---

## 10. Stage 7: Evaluation & Output (model.py + main.py)

### 10.1 Evaluation and Saving

#### `evaluate_and_save(model, X_train, y_train, X_test, customer_ids, test, feature_types, tag) -> metrics` — `model.py:266`

1. Computes train AU-PRC.
2. Computes test AU-PRC (only if test has valid target labels with >= 2 unique values).
3. Saves `prediction_{tag}.csv` with columns `[CustomerID, probability_score]`.
4. Saves `model_{tag}.joblib`.

### 10.2 Final Output

The mitigated model's outputs are copied to the final filenames (`main.py:613-614`):
```
prediction_mitigated.csv -> prediction.csv
model_mitigated.joblib -> model.joblib
```

### 10.3 Console Summary Tables

Four tables are printed after all computation completes:

1. **Drift & Mitigation Table** (`print_drift_mitigation_summary()`, `main.py:20`): Shows each drifted feature, its type, a natural-language drift description, and the mitigation action taken.
2. **Multivariate Shift Summary** (`print_multivariate_shift_summary()`, `main.py:147`): Domain classifier AUC, fold AUC std, shift detected yes/no, winning weight configuration.
3. **Model Performance Summary** (`print_model_performance_summary()`, `model.py:309`): Baseline vs Mitigated AU-PRC for train and test, with deltas.
4. **Runtime Summary** (`print_runtime_summary()`, `main.py:177`): Per-stage timing breakdown.

---

## 11. Shared Utilities (utils.py)

### `safe_n_splits(n_samples, desired, min_per_fold) -> int` — `utils.py:15`
Returns a safe number of CV folds: `max(2, min(desired, n_samples // min_per_fold))`. Prevents errors when datasets are too small for the desired fold count.

### `_parse_month_labels(series) -> Series | None` — `utils.py:26`
Parses a temporal series into month-period strings (e.g., "2025-01") for stratification. Tries 11 date formats (`_TEMPORAL_FORMATS`, `utils.py:20`). Returns None if parsing fails.

### `stratified_subsample_idx(n_total, n_sample, month_values, seed) -> ndarray` — `utils.py:39`
Returns indices for a month-stratified subsample that preserves the proportional distribution of months. Falls back to random sampling if no temporal information.

**Algorithm** (`utils.py:71-109`):
1. Compute per-group allocation proportional to group size (`utils.py:83-89`).
2. Distribute fractional remainders to groups with largest fractional parts (`utils.py:92-100`).
3. Sample within each group (`utils.py:103-109`).

### `Timer` — `utils.py:114`
Context manager that records elapsed wall-clock time. Used to instrument each pipeline stage.

---

## 12. Data Flow Diagram

```
                        train.csv                    test.csv
                           |                            |
                     load_data()                   load_data()
                           |                            |
                     _downcast_dataframe()         _downcast_dataframe()
                           |                            |
                     align_train_test_columns()----------+
                           |                            |
                     detect_feature_types(train, test_df=test)
                           |
                      feature_types dict ─────────────────────────────────────┐
                           |                                                  |
                   Decision Sample (100K)                                     |
                   Pipeline Cap (if needed)                                   |
                           |                                                  |
              ┌────────────┴────────────┐                                     |
              v                         v                                     |
   run_drift_analysis()    detect_intra_training_drift()                      |
              |                         |                                     |
         drift_df                  month_rank                                 |
              |                         |                                     |
              v                         |                                     |
   apply_all_mitigations() ─────────────┤                                     |
    ├─ normalize_data_quality()         |                                     |
    ├─ re-check drift                   |                                     |
    ├─ progressive high-drift           |                                     |
    ├─ progressive moderate-drift       |                                     |
    └─ detect_multivariate_shift()      |                                     |
              |                         |                                     |
    train_m, test_m, drop_cols,         |                                     |
    shift_info (train_oof_proba)        |                                     |
              |                         |                                     |
              v                         |                                     |
    engineer_features() ────────────────┤                                     |
    ├─ missing indicators               |                                     |
    ├─ frequency encoding               |                                     |
    ├─ ratio features                   |                                     |
    ├─ aggregation features             |                                     |
    └─ near-duplicate removal           |                                     |
              |                         |                                     |
              v                         |                                     |
    post-engineering drift check        |                                     |
              |                         |                                     |
              v                         |                                     |
  ┌─── prepare_features() ──────────────┤──────────────────────────────── uses |
  |   (target encoding, temporal parse, |                                     |
  |    categorical encoding, dtype fix) |                                     |
  |           |                         |                                     |
  |    X_train_m, y_train_m,            |                                     |
  |    X_test_m, customer_ids           |                                     |
  |           |                         |                                     |
  |    Feature selection                |                                     |
  |    ├─ zero-importance pruning       |                                     |
  |    └─ top-feature interactions      |                                     |
  |           |                         |                                     |
  |    Weight tournament ───────────────┘
  |    ├─ Uniform baseline         train_oof_proba + month_rank
  |    ├─ Blend (Nelder-Mead)      -> domain_weights + recency_weights
  |    └─ Geometric (Nelder-Mead)  -> blended/geometric weights
  |           |
  |    best_weights (winner)
  |           |
  |    train_model(X_train_m, y_train_m, sample_weight=best_weights)
  |           |
  |    evaluate_and_save()
  |           |
  |    prediction_mitigated.csv -> prediction.csv
  |    model_mitigated.joblib   -> model.joblib
  |
  └─── Baseline path (separate):
       prepare_features(train, test)  # unmitigated
       train_model(X_train_b, y_train_b)  # uniform weights
       evaluate_and_save(tag="baseline")
```

---

## 13. Key Design Principles

### No Hardcoded Column Names

Every column reference flows through `feature_types`. The three default names (`CustomerID`, `ChurnStatus`, `Month`) are only hints with full auto-detection fallback. This ensures the pipeline works on the hidden eval dataset with unknown column names.

### No Data Leakage

- Target is excluded from feature engineering (`skip_cols`), mitigation, and domain classification.
- Target encoding uses k-fold (each row's encoding comes from out-of-fold data only).
- Aggregation features group by feature columns, never the target.

### No Row Filtering

Mitigation and feature engineering only modify or add/drop **columns**, never rows. This preserves index alignment: `month_rank`, `train_oof_proba`, `best_weights`, and `X_train_m` all have exactly `len(train)` rows throughout.

### Subsample-Then-Apply

Expensive operations (strategy evaluation, correlation matrices, domain classifiers) run on subsamples. Winners are applied to full data. This keeps the pipeline within the 10-minute budget even at 10M rows.

### Conservative Mitigation

Strategies are only applied if they demonstrably reduce drift (measured by re-running statistical tests after transformation). This prevents "cures worse than the disease" — a mitigation that increases drift would harm the model.

### Fixed Model, Variable Data

Since the LightGBM hyperparameters are frozen, all pipeline value comes from three levers:
1. **What the model sees** (feature engineering + feature selection)
2. **How clean the data is** (mitigation strategies reducing train/test distribution gaps)
3. **What the model emphasizes** (sample weights from domain classifier + recency)
