# NAISC Pipeline: Comprehensive Technical Guide

This document is a complete walkthrough of the automated drift detection and mitigation pipeline for the NAISC Singtel 2026 challenge. It covers every function, its inputs, outputs, and the reasoning behind each design decision.

> **Note**: This guide reflects the **polars-refactor** branch. All DataFrames/Series are polars unless otherwise noted. Line numbers are approximate — refer to function names for navigation.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Entry Point: main.py](#2-entry-point-mainpy)
3. [Stage 1: Data Ingestion & Profiling (profiler.py)](#3-stage-1-data-ingestion--profiling-profilerpy)
4. [Stage 1b: Decision Sample & Pipeline Cap (main.py)](#4-stage-1b-decision-sample--pipeline-cap-mainpy)
5. [Stage 2: Drift Detection (drift_detector.py)](#5-stage-2-drift-detection-drift_detectorpy)
6. [Stage 3: Drift Mitigation (mitigation.py)](#6-stage-3-drift-mitigation-mitigationpy)
7. [Stage 3b: Feature Strategy Tournament (main.py)](#7-stage-3b-feature-strategy-tournament-mainpy)
8. [Stage 4: Feature Engineering (feature_engineer.py)](#8-stage-4-feature-engineering-feature_engineerpy)
9. [Stage 5: Preprocessing & Encoding (model.py)](#9-stage-5-preprocessing--encoding-modelpy)
10. [Stage 6: Model Training & Prediction (main.py + model.py)](#10-stage-6-model-training--prediction-mainpy--modelpy)
11. [Stage 7: Evaluation & Output (model.py + main.py)](#11-stage-7-evaluation--output-modelpy--mainpy)
12. [Shared Utilities (utils.py)](#12-shared-utilities-utilspy)
13. [Polars Compatibility Layer (polars_compat.py)](#13-polars-compatibility-layer-polars_compatpy)
14. [Data Flow Diagram](#14-data-flow-diagram)
15. [Key Design Principles](#15-key-design-principles)

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
[1]  Load & Profile  -->  feature_types dict
    |
[1b] Decision Sample + Pipeline Cap
    |
[2]  Drift Detection  -->  drift_df (per-feature severity) + month_rank
    |
[3]  Drift Mitigation  -->  train_m, test_m (cleaned) + drop_cols
    |
[3b] Feature Strategy Tournament  -->  per-feature {current, raw, drop} evaluation
    |
[4]  Feature Engineering  -->  train_m, test_m (enriched with new features)
    |
[5]  Preprocessing  -->  X_train, y_train, X_test (model-ready matrices)
    |
[6]  Model Training  -->  feature selection, FE validation gate,
     |                     iterative mitigation, self-training
     v
[7]  Evaluation & Output  -->  prediction.csv, model.joblib
```

---

## 2. Entry Point: main.py

`main()` is the orchestrator. It calls into each module sequentially and passes state between stages via local variables. Each stage is timed with a `Timer` context manager, and summary tables are printed at the end.

**Key orchestration decisions:**

- **Two models are always trained**: a _baseline_ (no mitigation, uniform weights) and a _mitigated_ (full pipeline). Both are saved, and the mitigated version is copied to the final `prediction.csv` / `model.joblib`.
- **State carries forward**: `feature_types` (from Stage 1) flows through every subsequent stage as the universal schema reference. `drift_df` (from Stage 2) drives Stage 3's mitigation strategy selection.
- **Uniform sample weights**: Domain classifier weighting and weight tournament are disabled. All models train with uniform weights.
- **Display functions** (`print_drift_mitigation_summary`, `print_multivariate_shift_summary`, `print_runtime_summary`) are called only after all computation completes, so the console output is clean and grouped.

---

## 3. Stage 1: Data Ingestion & Profiling (profiler.py)

### 3.1 Loading Data

#### `load_data(train_path, test_path) -> (train_df, test_df)`

| Parameter | Type | Description |
|---|---|---|
| `train_path` | `str` | Filesystem path to training CSV |
| `test_path` | `str` | Filesystem path to test CSV |
| **Returns** | `tuple[pl.DataFrame, pl.DataFrame]` | Loaded polars DataFrames |

**Logic:**
1. Reads both CSVs with `pl.read_csv(path, infer_schema_length=0)`. The `infer_schema_length=0` forces **all columns to load as Utf8 (string)** — this is deliberate to avoid polars erroring on mixed-type columns (e.g., a numeric column with blank strings). Type inference happens later in `detect_feature_types`.
2. **Duplicate column handling**: Polars rejects duplicate column names on read, auto-renaming them with `_duplicated_N` suffix. The code detects and logs these.
3. **Memory downcasting** via `downcast_dataframe()`: Converts `Float64` -> `Float32`, and `Int64` -> the smallest integer type that fits the actual data range (`Int8`/`Int16`/`Int32`). Called in `main.py` **after** the numeric coercion loop (which casts Utf8 numericals to `Float64`), so it operates on properly-typed columns and is not a no-op.
4. Logs the shape and estimated memory before/after.

### 3.2 Column Alignment

#### `align_train_test_columns(train, test) -> (train, test_aligned)`

| Parameter | Type | Description |
|---|---|---|
| `train` | `pl.DataFrame` | Training data |
| `test` | `pl.DataFrame` | Test data |
| **Returns** | `tuple[pl.DataFrame, pl.DataFrame]` | Aligned DataFrames (test columns renamed to match train) |

**Logic:**
1. Finds columns in train but not in test, and vice versa.
2. For mismatched columns, normalizes names via `_normalize_col_name()` — lowercase, strip underscores/spaces/hyphens — and checks for fuzzy matches. If `test` has `"customer_id"` and `train` has `"CustomerID"`, they'll match.
3. Renames test columns to train's naming convention via `test.rename(rename_map)`.
4. Logs warnings for columns that remain unmatched (expected: target column is typically train-only).

**Rationale:** The hidden dataset may use different column naming conventions (e.g., snake_case vs CamelCase). This ensures column alignment without hardcoding any names.

### 3.3 Feature Type Detection

#### `detect_feature_types(df, id_col, target_col, temporal_col, test_df) -> feature_types`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pl.DataFrame` | — | Training data for type inference |
| `id_col` | `str` | `"CustomerID"` | Hint for ID column name |
| `target_col` | `str` | `"ChurnStatus"` | Hint for target column name |
| `temporal_col` | `str` | `"Month"` | Hint for temporal column name |
| `test_df` | `pl.DataFrame \| None` | `None` | Test data (only columns in both are classified) |
| **Returns** | `dict[str, list[str]]` | `{"numerical": [...], "categorical": [...], "temporal": [...], "id": [...], "target": [...]}` |

**Logic:**

1. **Exact name matching**: Each of `id_col`, `target_col`, `temporal_col` is matched by exact name. If not found, a warning is logged and the column is set to None (no auto-detection fallback in the polars version — the column is simply skipped).

2. **Column filtering**: When `test_df` is provided, only columns present in **both** datasets are classified as features. Train-only columns (besides id/target) are logged and skipped. This prevents the model from learning features that won't exist at prediction time.

3. **Per-column classification** (on a 100K-row sample for efficiency):
   - Skip all-null columns (uninformative).
   - Skip zero-variance columns (only 1 unique non-null value).
   - Skip near-zero-variance columns (>99% same value) — these add noise without predictive power.
   - If matched as `temporal_col`, classify as temporal.
   - If non-numeric but >80% parseable as numbers (`_looks_numeric()` heuristic via `to_numeric_coerce`), classify as numerical. This handles columns like `TotalCharges` which have numeric values with a few blank strings.
   - If numeric dtype → numerical. (Note: with `infer_schema_length=0`, this branch only triggers for columns that were explicitly cast earlier.)
   - Otherwise → categorical.

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

#### `profile_data(df, feature_types, label) -> pl.DataFrame`

| Parameter | Type | Description |
|---|---|---|
| `df` | `pl.DataFrame` | Dataset to profile |
| `feature_types` | `dict` | Type classification from `detect_feature_types` |
| `label` | `str` | Label for logging (e.g., "train", "test") |
| **Returns** | `pl.DataFrame` | Summary with columns: column, type, dtype, n_unique, null_pct, sample_value |

Produces a summary of each column's characteristics. Used primarily for logging — the console output shows counts like `43 columns (17 num, 23 cat, 1 temporal)`.

---

## 4. Stage 1b: Decision Sample & Pipeline Cap (main.py)

### 4.1 Decision Sample

For datasets larger than 100K rows, many operations (drift detection, feature selection, weight tournament) don't benefit from using all rows — statistical tests converge well before 100K. The pipeline creates a **single decision sample** that's reused across these operations.

**Logic:**
1. Parse the temporal column (if any) via `_parse_month_labels()` (`utils.py`) to get month-period labels for stratified sampling.
2. If `len(train) > 100,000`: create `_decision_idx` using `stratified_subsample_idx()` (`utils.py`) — 100K rows, preserving month distribution.
3. Create `train_decision = train[_decision_idx]` for drift detection (polars integer-based row selection).

**Rationale:** Using the same 100K sample across all "decision" operations (drift detection, feature importance, tournament) ensures consistency. Stratified sampling preserves the temporal distribution, which matters because drift is temporal.

### 4.2 Pipeline Cap

For extremely large datasets, the full pipeline (mitigation, feature engineering, model training) must complete within 10 minutes.

**Logic:**
1. Compute `_max_pipeline_rows = 300 / (8.0e-7 * n_cols)`. This formula is calibrated from benchmarks (2M rows x 200 cols on 2-core Docker). The cap is floored at `_DECISION_SAMPLE_SIZE` (100K) to never go below the decision sample.
2. If `len(train) > _max_pipeline_rows`: subsample train using stratified temporal sampling, then **recompute** `_month_vals`, `_decision_idx`, and `train_decision` on the capped data — the pre-cap indices are now invalid.

**Example scaling:**
| Dataset | Max Rows | Cap Needed? |
|---|---|---|
| 70K x 45 (public) | ~8.3M | No |
| 2M x 200 | ~1.9M | No |
| 10M x 500 | ~750K | Yes, capped to 750K |

---

## 5. Stage 2: Drift Detection (drift_detector.py)

Drift detection identifies features whose distributions differ between train and test. This is the diagnostic foundation: Stage 3's mitigation strategies are driven by these results.

### 5.1 Train vs Test Drift

#### `run_drift_analysis(train, test, feature_types) -> drift_df`

| Parameter | Type | Description |
|---|---|---|
| `train` | `pl.DataFrame` | Training data (decision sample) |
| `test` | `pl.DataFrame` | Test data (subsampled if >100K) |
| `feature_types` | `dict` | Feature type classification |
| **Returns** | `pl.DataFrame` | One row per feature: column, feature_type, severity (high/moderate/low), statistical metrics, drift_description |

**Logic:**
1. Builds a task list of `(column, type)` pairs from feature_types.
2. For each feature, calls `detect_numerical_drift()` or `detect_categorical_drift()`.
3. Generates a natural-language drift description via `generate_drift_description()`.
4. For >20 features, parallelizes via `joblib.Parallel` with threading.
5. Applies **Benjamini-Hochberg FDR correction** via `_apply_bh_correction()` across all p-values, then re-classifies severity using corrected p-values.
6. Sorts output by severity (high first).

### 5.2 Numerical Drift Detection

#### `detect_numerical_drift(train_col, test_col) -> dict`

| Parameter | Type | Description |
|---|---|---|
| `train_col` | `pl.Series` | Training column values |
| `test_col` | `pl.Series` | Test column values |
| **Returns** | `dict` | `{ks_stat, ks_pval, psi, wasserstein, wasserstein_norm, iqr_ratio, tail_divergence, severity, ...}` |

**Statistical tests (each catches different drift patterns):**

| Test | What It Catches | How |
|---|---|---|
| **KS test** (Kolmogorov-Smirnov) | Any distributional difference | Maximum vertical distance between CDFs. P-value indicates significance. |
| **PSI** (Population Stability Index) | Binned distribution changes | `compute_psi()`. Divides range into quantile bins, compares bin proportions. PSI > 0.25 = significant shift. |
| **Wasserstein distance** | Mean/location shifts | "Earth mover's distance" — minimum work to transform one distribution into another. Normalized by train std for scale-invariance. |
| **IQR ratio** | Spread changes | `test_IQR / train_IQR`. Ratio > 2 or < 0.5 = significant spread change. |
| **Tail divergence** | Extreme value shifts | Max shift in 1st/99th percentiles, normalized by train std. Catches tail behavior changes missed by histogram-based PSI. |

**Subsampling:** Both columns are subsampled to 100K rows via `_subsample_array()` before testing. The cap is `_MAX_DRIFT_SAMPLE = 100_000`. Beyond this, additional rows add negligible statistical power but cost O(n log n) compute.

**Severity classification** (`_numerical_severity()`):
- **High**: PSI >= 0.25, or (PSI >= 0.15 and KS p < 0.001), or (Wasserstein_norm >= 1.0 and PSI >= 0.1), or (IQR ratio > 2x and PSI >= 0.1), or (tail_div > 2 and PSI >= 0.05).
- **Moderate**: PSI >= 0.1, or (KS p < 0.05 and KS_D >= 0.05), or (tail_div > 1 and PSI >= 0.03).
- **Low**: Everything else.

**Small-sample correction** (`_small_sample_factor()`): A `ss_factor` widens PSI thresholds when samples are small (harmonic mean < 5000), preventing false positives from noisy histogram estimates.

### 5.3 Categorical Drift Detection

#### `detect_categorical_drift(train_col, test_col) -> dict`

| Parameter | Type | Description |
|---|---|---|
| `train_col` | `pl.Series` | Training column values |
| `test_col` | `pl.Series` | Test column values |
| **Returns** | `dict` | `{chi2_stat, chi2_pval, jsd, cramers_v, unseen_in_test, severity, ...}` |

**Statistical tests:**

| Test | What It Catches | How |
|---|---|---|
| **Chi-squared test** | Any categorical distribution change | Contingency table test. For high-cardinality (>50 categories), groups rare categories into `_other` to avoid sparse-table inflation. |
| **JSD** (Jensen-Shannon Divergence) | Symmetric distributional divergence | `compute_jsd()`. Average of KL divergences in both directions. Range [0, ln(2)]. More robust than chi-squared for unbalanced categories. |
| **Cramer's V** | Sample-size-normalized effect size | `compute_cramers_v()`. `sqrt(chi2 / (n * (k-1)))`. Unlike chi2 p-value, V is independent of sample size — a p < 0.001 on 100K rows might still be a tiny effect. |

**Count alignment:** Category counts are aligned using polars joins — both train and test value_counts are joined against the union of all categories, with missing counts filled as 0.

**Severity classification** (`_categorical_severity()`):
- **High**: JSD >= 0.1, or (Cramer's V >= 0.3 and JSD >= 0.05).
- **Moderate**: JSD >= 0.05, or chi2 p < 0.001, or Cramer's V >= 0.15.
- **Low**: Everything else.

### 5.4 Drift Description Generation

#### `generate_drift_description(train_col, test_col, feature_type, drift_metrics) -> str`

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

#### `_apply_bh_correction(drift_df, ss_factor) -> drift_df`

When testing 40+ features simultaneously, some will appear significant by chance. The BH procedure controls the **false discovery rate** — the expected proportion of false positives among declared significant results.

**Logic:**
1. Collects all p-values (KS for numerical, chi2 for categorical).
2. Sorts p-values and applies the BH formula via `_bh_correct()`: `adjusted_p = raw_p * n / rank`.
3. Stores corrected p-values in `ks_pval_bh` / `chi2_pval_bh` columns.
4. Re-classifies severity using corrected p-values (with the same `ss_factor` for consistency).
5. Re-applies missingness escalation via `_missingness_severity()` and `_escalate_severity()` — so missingness drift isn't lost when severity is overwritten.

### 5.6 Intra-Training Drift Detection

#### `detect_intra_training_drift(train, feature_types) -> (intra_drift_df, month_rank)`

| Parameter | Type | Description |
|---|---|---|
| `train` | `pl.DataFrame` | Full (post-cap) training data |
| `feature_types` | `dict` | Feature type classification |
| **Returns** | `tuple[pl.DataFrame, pl.Series \| None]` | Intra-training drift results + per-row month rank |

**Logic:**
1. Finds the temporal column and parses it into month periods (tries 11 date formats via `str.to_datetime` with `str.to_titlecase()` to restore month abbreviations after lowercase normalization).
2. Sorts unique months chronologically, splits into early (first half) and late (second half).
3. Subsamples each half to 100K rows (`_MAX_INTRA_SAMPLE = 100_000`) for efficiency.
4. Runs the same drift detection between early and late halves (the primary drift signal for `intra_drift_df`). Returns only high/moderate-drift features.
5. Builds `month_rank`: a Series mapping each row to its month's ordinal rank (0 = earliest month, N-1 = latest month) via `pl.Series.replace_strict()`. Used by `compute_recency_weights()` in Stage 6.

**Rationale:** The early/late split provides the primary intra-training drift signal — identifying features whose distributions shift over time within training data. The `month_rank` output feeds into recency weighting, which allows the model to emphasize recent (more test-representative) training months.

### 5.7 Multivariate Shift Detection (DISABLED)

#### `detect_multivariate_shift(train, test, feature_types, extra_drop_cols, n_splits) -> dict`

**Status:** This function exists in `drift_detector.py` but is **not called** by the current pipeline. `apply_all_mitigations()` returns a stub `shift_info = {auc: 0.0, shift_detected: False, ...}`. The domain classifier was disabled because its sample weights did not improve AU-PRC on the public dataset — the univariate drift mitigation and feature strategy tournament proved more effective.

The function trains a LightGBM domain classifier to distinguish train from test rows, producing per-row P(test) probabilities that could be used as sample weights. It remains available if weight-based approaches are re-enabled.

---

## 6. Stage 3: Drift Mitigation (mitigation.py)

### 6.1 Orchestrator

#### `apply_all_mitigations(train, test, drift_df, feature_types) -> (train_m, test_m, drop_cols, mitigation_log, norm_log, shift_info)`

| Parameter | Type | Description |
|---|---|---|
| `train` | `pl.DataFrame` | Full training data |
| `test` | `pl.DataFrame` | Full test data |
| `drift_df` | `pl.DataFrame` | Drift results from Stage 2 |
| `feature_types` | `dict` | Feature type classification |
| **Returns** | | |
| `train_m` | `pl.DataFrame` | Mitigated training data |
| `test_m` | `pl.DataFrame` | Mitigated test data |
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
```

Note: Domain classifier / multivariate shift detection is disabled. `shift_info` is returned as a stub (`auc=0.0, shift_detected=False`).

### 6.2 Data Quality Normalization (Stage 3a)

#### `normalize_data_quality(train, test, feature_types) -> (train_clean, test_clean, norm_log)`

This stage fixes surface-level data quality issues that masquerade as drift.

**Three normalization steps:**

1. **Abbreviation detection** (`detect_abbreviation_mappings()`):
   - Scans each categorical column (Utf8/String dtype) for 2-3 character uppercase strings (e.g., "LA", "NYC", "SD").
   - Matches abbreviations to multi-word values whose word initials match (e.g., "LA" -> "Los Angeles").
   - Ties broken by combined frequency across both datasets.
   - Applies mapping to both train and test via `pl.when().then().otherwise()` expressions.
   - *Why this matters:* If train has "Los Angeles" but test has "LA", the model would treat them as different categories, losing signal.

2. **Conditional lowercase + hyphen removal**:
   - For each categorical column, computes JSD **before** and **after** lowercasing + hyphen removal (on a 100K subsample for efficiency).
   - Only applies if normalization reduces or maintains JSD — i.e., doesn't increase drift.
   - Uses `pl.col(col).cast(pl.Utf8).str.to_lowercase().str.replace_all("-", " ", literal=True)`.
   - *Why conditional:* Some columns have meaningful case distinctions. Blindly lowercasing "Yes"/"No" -> "yes"/"no" wouldn't help. The JSD check ensures normalization only happens when it genuinely closes the gap.
   - *Example:* `Contract` has "Month-to-Month" in train vs "month-to-month" in test. Lowercasing + hyphen removal aligns them.

3. **Fuzzy matching for unseen test categories**:
   - For low-cardinality categoricals (`_MAX_FUZZY_TRAIN_CATS=200`, `_MAX_FUZZY_UNSEEN=50`), uses `difflib.get_close_matches()` to map unseen test values to their closest train equivalent.
   - Two-pass: strict cutoff (0.6) then relaxed (0.45).
   - High-cardinality columns are skipped — fuzzy matching is O(unseen x train_cats) and only useful for typo-like mismatches.
   - Applied via chained `pl.when().then().otherwise()` expressions.

### 6.3 Drift Re-check (Stage 3b)

After normalization, drift is rechecked only for modified columns (not all 500). This is an efficiency optimization — why re-test 480 unchanged columns?

**Logic:**
1. For each column in `norm_log`, re-runs `_analyze_single_feature()`.
2. Updates the `drift_df_norm` DataFrame by building replacement column values at the matched row index.
3. Re-applies BH correction and re-sorts by severity.

### 6.4 Progressive Mitigation — High Drift (Stage 3c)

#### `mitigate_feature_progressive(train, test, col, feature_type, original_severity) -> (train, test, log_entry)`

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

Each strategy is first evaluated on a 100K subsample via `_eval_subsample()`. If it reduces severity, it's applied to full data. This subsample-then-apply pattern saves computation when strategies fail.

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

**Cap at 50** (`_MAX_MODERATE = 50`): For datasets with hundreds of features, only the top-50 most-drifted moderate features are processed (sorted by PSI/JSD). This prevents the mitigation stage from dominating runtime.

### 6.6 Mitigation Strategy Reference

| Strategy | Function | Input | Output | Rationale |
|---|---|---|---|---|
| Median align | `median_align()` | num, num | num, num | Shifts test median to match train median. Simple location correction. |
| Clip outliers | `clip_outliers()` | num, num | num, num | Clips both to train's [1st, 99th] percentiles. Tames extreme values. Uses `pl.Series.clip()`. |
| Log transform | `log_transform()` | num, num | num, num | log1p via numpy, wrapped back as polars Series. Compresses right-skewed distributions. Only works for non-negative data. |
| Robust scaling | `robust_scale()` | num, num | num, num | (x - median) / IQR using train stats. Robust to outliers unlike z-score. |
| Quantile binning | `quantile_bin()` | num, num | int, int | Discretizes into bins defined by train quantile edges via `np.digitize`. Eliminates continuous drift. |
| Quantile matching | `quantile_match()` | num, num | num, num | CDF-to-CDF mapping. Each test value -> its percentile rank in test -> mapped to corresponding train quantile via `np.interp`. Preserves test rank order, aligns distributions. |
| Rank transform | `rank_transform()` | num, num | float, float | Converts to percentile rank (0-1) based on train CDF via `np.searchsorted`. Distribution-agnostic. |
| Clip+Log | `_clip_then_log()` | num, num | num, num | Clip outliers first, shift to non-negative, then log1p. Combines both strategies. |
| Group rare cats | `group_rare_categories()` | cat, cat | cat, cat | Categories <1% frequency in train -> `_other`. Also maps unseen test categories to `_other`. Uses `pl.when().then().otherwise()` expressions. |
| Frequency encode | `frequency_encode_mitigation()` | cat, cat | float, float | Replace category with train-set frequency (0-1) via `pl.Series.replace_strict()`. Converts categorical -> numerical. |
| Cyclical encode | `encode_temporal_cyclical()` | temporal, temporal | sin+cos, sin+cos | Parse month -> sin(2pi*m/12), cos(2pi*m/12). Captures seasonality without ordinal bias. |

---

## 7. Stage 3b: Feature Strategy Tournament (main.py)

After `apply_all_mitigations()` returns, each **transformed** feature (those that had a mitigation strategy applied, not just normalization or drops) is evaluated in a three-way tournament: keep the current transform, revert to the raw (unmitigated) column, or drop the feature entirely.

**Motivation:** Progressive mitigation picks strategies that reduce drift statistics, but lower drift doesn't always mean better predictions. A transform might reduce PSI but destroy predictive signal. This stage answers: "Does this transform actually help AU-PRC?"

**Which features enter the tournament:**
- Only features where `strategy_applied` is a transform (e.g., "Quantile matching", "Robust Scaling", "Rank Transform").
- Features that were only normalized (lowercased, abbreviation-mapped) are excluded — those are safe corrections.
- Features that were dropped (all strategies failed) are excluded — trusted as correct.
- Frequency encoding is excluded — it fundamentally changes the feature type.

**Logic:**
1. Subsample mitigated training data to 50K rows (`_FST_SAMPLE`), keeping a mapping to original (raw) training data at the same row positions.
2. **Temporal holdout split**: Hold out the last ~20% of months as the validation set. Falls back to random 20% split if no temporal column is available.
3. For each tournament feature, evaluate three strategies with **3-seed averaging** (seeds 42, 123, 456) for stability:
   - **Current**: the mitigated data as-is.
   - **Drop**: exclude the feature entirely.
   - **Raw**: swap the mitigated column with the original unmitigated column.
4. Pick the winner. Require a **0.005 improvement** over current to switch — this filters out noise from subsample variation.

**Outcomes:**
- If **drop** wins: feature is added to `drop_cols`, mitigation log updated to "Drop Feature (tournament)".
- If **raw** wins: the mitigated column is replaced with the original in both `train_m` and `test_m`, mitigation log updated to "Reverted to raw (tournament)".
- If **current** wins (or no alternative exceeds +0.005): transform is kept as-is.

**Cost:** `3 strategies x 3 seeds x N features` model trains on a 50K subsample. Typically 2-5 tournament features, so 18-45 model trains (~15-30 seconds total).

### 6.7 Sample Weight Functions

These functions exist in `mitigation.py` but are **currently unused** — all models train with uniform weights. They remain available if weight-based approaches are re-enabled.

- `compute_domain_weights(train_oof_proba, power)` — converts domain classifier P(test) into sample weights with power dampening and mean=1.0 normalization.
- `compute_recency_weights(month_rank, decay)` — exponential decay by temporal recency, normalized to mean=1.0.

---

## 8. Stage 4: Feature Engineering (feature_engineer.py)

#### `engineer_features(train, test, feature_types, drop_cols) -> (train_enriched, test_enriched, new_feature_names)`

| Parameter | Type | Description |
|---|---|---|
| `train` | `pl.DataFrame` | Mitigated training data |
| `test` | `pl.DataFrame` | Mitigated test data |
| `feature_types` | `dict` | Feature type classification |
| `drop_cols` | `list[str]` | Columns marked for dropping |
| **Returns** | `tuple[pl.DataFrame, pl.DataFrame, list[str]]` | Enriched DataFrames + list of new feature names |

All feature engineering is **dynamic** — no hardcoded column names. Features are derived from the `feature_types` dict and statistical properties of the data. A name collision guard (`existing_cols`) prevents duplicate feature names across all stages.

### 8.1 Missing Value Indicators

For columns with >1% null rate, adds a binary `{col}_is_null` feature via `pl.Series.is_null().cast(pl.Int32)`. New columns are attached via `pl.DataFrame.hstack()`. This lets the model learn from missingness patterns — e.g., if `TotalCharges` is null for new customers with 0 tenure, that pattern is predictive.

### 8.2 Frequency Encoding

For high-cardinality categoricals (>10 unique values), adds `{col}_freq` — the training-set frequency of each category value via `pl.Series.replace_strict(freq_map, default=0.0, return_dtype=pl.Float64)`. This provides a numerical representation that captures category popularity without the dimensionality explosion of one-hot encoding.

Low-cardinality categoricals (<= 10 unique) are handled later by target encoding in Stage 5.

### 8.3 Ratio Features

Finds the top-5 most correlated numerical pairs (|r| > 0.3) and creates `{col_a}_div_{col_b}` = a / (b + 1). The +1 prevents division by zero.

**Correlation computation:** Uses `_pairwise_corr_matrix()` — a custom pairwise-complete correlation matrix that handles nulls by masking per-pair (matching pandas `.corr()` behavior). Built with numpy arrays extracted from polars columns via `.cast(pl.Float64).to_numpy()`.

**Correlation cap** (`_MAX_CORR_COLS=50`): For datasets with >50 numerical features, only the top-50 by variance are used for the correlation matrix (avoiding O(m^2) computation on 500 features).

**Rationale:** Ratio features capture relationships that tree models can technically learn but require many splits to approximate. `TotalCharges / TotalRevenue` is a single feature expressing relative spending efficiency.

### 8.4 Aggregation Features

Groups `mean(numerical)` by `categorical`, creating features like `MonthlyCharge_mean_by_Contract`.

- Top 3 categoricals (by cardinality, 3-50 unique) x top 5 numericals = up to 15 features.
- Train: computed via `pl.col(num).mean().over(cat_col)` window function.
- Test: mapped using training group means via `pl.Series.replace_strict(means_map, default=global_mean)`. Unseen test categories get the global mean.

**Rationale:** Aggregation features encode group-level patterns. If customers on "Month-to-Month" contracts have higher average MonthlyCharge, this feature makes that pattern directly accessible to the model.

### 8.5 Near-Duplicate Removal

After all features are added, computes the correlation matrix among all numerical features via `_pairwise_corr_matrix()` and drops one of any pair with |corr| > 0.98. When choosing which to drop, engineered features are preferred over originals (keep the raw signal, discard the derivative).

**Column cap** (`_MAX_DEDUP_COLS=200`): For >200 numerical features, only the top-200 by variance are checked (avoiding O(m^2) on 500+ columns).

### 8.6 Post-Engineering Drift Check (main.py)

Back in `main.py`, each newly engineered feature is tested for drift via `detect_numerical_drift()`. Any engineered feature with "high" drift is dropped — it would confuse the model more than help.

---

## 9. Stage 5: Preprocessing & Encoding (model.py)

### 9.1 Feature Preparation

#### `prepare_features(train, test, feature_types, extra_drop_cols) -> (X_train, y_train, X_test, customer_ids)`

| Parameter | Type | Description |
|---|---|---|
| `train` | `pl.DataFrame` | Training data (original or mitigated) |
| `test` | `pl.DataFrame` | Test data (original or mitigated) |
| `feature_types` | `dict` | Feature type classification |
| `extra_drop_cols` | `list[str] \| None` | Additional columns to exclude |
| **Returns** | | |
| `X_train` | `pl.DataFrame` | Model-ready training features |
| `y_train` | `pl.Series` | Binary target (0/1) |
| `X_test` | `pl.DataFrame` | Model-ready test features |
| `customer_ids` | `list[str]` | Test row identifiers for prediction output |

**Processing steps:**

1. **Extract customer IDs** from test (`test[id_col].to_list()`), fallback to row indices if ID column missing.

2. **Encode target** via `_encode_target()`: Handles "Yes"/"No", 1/0, True/False, "1"/"0", etc. Uses `pl.Series.cast(pl.Utf8).str.to_lowercase().str.strip_chars()` for string comparison, then `pl.when().then().otherwise()` for mapping. Falls back to minority-class-is-positive heuristic for unknown binary labels.

3. **Column selection**: Only columns present in both train and test, minus id/target/drop_cols.

4. **Numeric coercion**: Columns classified as numerical but stored as Utf8 (e.g., `TotalCharges` with blank strings) are converted via `to_numeric_coerce()` from `polars_compat.py` — casts to Utf8, strips whitespace, then casts to Float64 with `strict=False` (failures become null).

5. **Dtype alignment**: If train and test disagree on a column's dtype, both are coerced to numeric via `to_numeric_coerce()`.

6. **Temporal parsing**: Temporal columns are converted to month number (1-12) via `_parse_temporal_month()` — tries 11 date formats using `pl.Series.str.to_titlecase().str.to_datetime(fmt, strict=False)`, extracts `.dt.month()`.

7. **Categorical encoding** (split by cardinality):

   **High-cardinality (>10 unique) -> LightGBM native categorical:**
   - Integer label encoding via `pl.Series.replace_strict(cat_to_int, default=None, return_dtype=pl.Int64)`.
   - Unseen test categories get null (LightGBM handles missing values natively by learning the optimal split direction).
   - Stored in module-level `_native_cat_cols` list for later conversion to pandas `category` dtype.
   - *Why native:* LightGBM's native categorical handling uses optimal histogram-based splits, which outperforms one-hot or target encoding for high-cardinality features.

   **Low-cardinality (<=10 unique) -> Target encoding:**
   - K-fold encoding to prevent leakage: for each fold, category means are computed from out-of-fold rows only. Validation rows never see their own labels in their encoding.
   - **Bayesian smoothing** (`smoothing_m=10`): `smoothed_mean = (count * raw_mean + m * global_mean) / (count + m)`. Categories with few samples shrink toward the global mean, preventing overfitting to small groups.
   - **Large dataset optimization** (`_MAX_TE_SAMPLE=200_000`): For >200K rows, uses a stratified 200K subsample for fold mean computation (via `stratified_subsample_idx`), but maps results onto all rows using modular fold assignment.
   - **Single-class guard**: If y_train has only one class (edge case in hidden eval), all categories get the global mean.
   - Fold means computed using temporary polars DataFrames via `group_by().agg()`.
   - Test set uses the full-data smoothed means (not fold-specific).

8. **NaN handling**: Numeric NaN is left as-is (LightGBM handles it natively). Non-numeric NaN is filled with mode via `pl.Series.mode()`.

### 9.2 Target Encoding: Why K-Fold?

Naive target encoding (computing category mean directly from training data) leaks the target — each row's encoding includes information from its own label. K-fold encoding fixes this:

```
For each fold:
  1. Compute category means using ONLY out-of-fold rows
  2. Assign these means to the held-out fold's rows
```

Each row's encoding comes from data it was never part of, eliminating leakage.

---

## 10. Stage 6: Model Training & Prediction (main.py + model.py)

### 10.1 Baseline Model

A model trained on unmitigated data with uniform weights, using the raw `train` and `test` DataFrames. Subsampled to 100K rows (`_MAX_BASELINE_SAMPLE`) for speed (it's only for comparison metrics). Saved as `prediction_baseline.csv` and `model_baseline.joblib`.

### 10.2 Feature Selection

Using the mitigated data:
1. Train an initial model on decision sample if data is large.
2. **Drop zero-importance features**: Features that LightGBM never split on. These add noise without predictive value.
3. **Add top-feature interactions**: Take the top-8 features by importance, create all pairwise products (`col_a * col_b` via `pl.col(ca) * pl.col(cb)`). This gives the model explicit access to interaction effects.

### 10.3 FE Validation Gate

Compares the pipeline **with** feature engineering vs **without** (mitigated data only, no engineered features). Both paths go through zero-importance pruning and top-feature interactions independently.

**Logic:**
1. Prepare both variants: `X_train_m` (with FE) and `X_train_nofe` (without FE, from saved `train_m_no_fe`).
2. Each variant gets its own zero-importance pruning and top-feature interactions (since importances differ).
3. Month-stratified 80/20 validation split (same seed for both).
4. Train LightGBM on each, compare validation AU-PRC.
5. If without-FE wins, swap `X_train_m` / `X_test_m` to the no-FE variant and update `test_m`.

**Rationale:** Feature engineering can hurt when it introduces noise or high-drift features. On the public dataset, FE is sometimes dropped (when it adds <0.002 AU-PRC improvement) because the engineered features don't generalize well to the shifted test distribution.

### 10.4 Iterative Mitigation (importance-aware)

After feature selection and FE validation, a second-pass mitigation targets features that are **both high-importance AND still drifted**. This focuses compute on what actually affects predictions.

**Logic:**
1. Train a model on the current `X_train_m` to get feature importances.
2. Identify features above the **50th percentile** of non-zero importances.
3. For each high-importance feature, re-run `detect_numerical_drift()` on the model-ready data.
4. If severity is "high" or "moderate", try:
   - **Quantile matching** first (preserves scale, aligns CDFs).
   - **Rank transform** as fallback (converts to percentile rank, distribution-agnostic).
5. Apply whichever reduces drift.

**Why this matters:** First-pass mitigation (Stage 3) operates on raw features before encoding and interactions. By Stage 6, the model-ready data includes encoded categoricals, interaction terms, and aggregation features — some of which may have drift that wasn't visible at the raw feature level. This second pass catches those cases, but only for features the model actually relies on.

**Cost:** One extra model train for importances + 2 drift-check + transform operations per qualifying feature. Typically <1 second.

### 10.5 Model Training

#### `train_model(X_train, y_train, sample_weight) -> LGBMClassifier`

| Parameter | Type | Description |
|---|---|---|
| `X_train` | `pl.DataFrame` | Training features |
| `y_train` | `pl.Series` | Binary target |
| `sample_weight` | `ndarray \| None` | Per-row weights (None = uniform) |
| **Returns** | `LGBMClassifier` | Trained model |

Uses the fixed hyperparameters. Currently always called with `sample_weight=None` (uniform weights).

**Polars→pandas conversion:** `X_train` is converted via `_to_pandas_with_categoricals()` which calls `to_pandas_for_lgbm()` from `polars_compat.py` and then sets native categorical columns to pandas `category` dtype. `y_train` is converted via `.to_numpy()`.

### 10.6 Self-Training (pseudo-labeling)

After the final model is trained, high-confidence test predictions are used to augment the training data, adapting the model to the test distribution.

**Logic:**
1. Score test data with the current model.
2. Select high-confidence predictions: P(churn) >= 0.95 → pseudo-label 1, P(churn) <= 0.05 → pseudo-label 0.
3. Concatenate pseudo-labeled test rows with original training data.
4. Retrain the model on the augmented dataset.
5. Single iteration only (`_ST_MAX_ITERS=1`) to avoid confirmation bias.

**Safeguards:**
- Minimum 10 confident samples required, otherwise self-training is skipped.
- Only extreme confidence thresholds (95%/5%) are used — moderate-confidence predictions would introduce noise.
- Single iteration prevents the model from reinforcing its own mistakes.

**Rationale:** The test distribution may differ from training in ways that sample weighting can't fix. By incorporating confidently-predicted test rows, the model's decision boundaries shift toward the test distribution. This is particularly helpful when the test population has a different class balance or feature composition.

---

## 11. Stage 7: Evaluation & Output (model.py + main.py)

### 11.1 Evaluation and Saving

#### `evaluate_and_save(model, X_train, y_train, X_test, customer_ids, test, feature_types, tag) -> metrics`

1. Converts polars DataFrames to pandas via `_to_pandas_with_categoricals()` for LightGBM `predict_proba()`.
2. Computes train AU-PRC.
3. Computes test AU-PRC (only if test has valid target labels with >= 2 unique values).
4. Saves `prediction_{tag}.csv` via `pl.DataFrame.write_csv()` with columns `[CustomerID, probability_score]`.
5. Saves `model_{tag}.joblib` via `joblib.dump()`.

### 11.2 Final Output

The mitigated model's outputs are copied to the final filenames:
```
prediction_mitigated.csv -> prediction.csv
model_mitigated.joblib -> model.joblib
```

### 11.3 Console Summary Tables

Four tables are printed after all computation completes:

1. **Drift & Mitigation Table** (`print_drift_mitigation_summary()`): Shows each drifted feature, its type, a natural-language drift description, and the mitigation action taken.
2. **Multivariate Shift Summary** (`print_multivariate_shift_summary()`): Domain classifier AUC, fold AUC std, shift detected yes/no, winning weight configuration.
3. **Model Performance Summary** (`print_model_performance_summary()`): Baseline vs Mitigated AU-PRC for train and test, with deltas.
4. **Runtime Summary** (`print_runtime_summary()`): Per-stage timing breakdown.

---

## 12. Shared Utilities (utils.py)

### `safe_n_splits(n_samples, desired, min_per_fold) -> int`
Returns a safe number of CV folds: `max(2, min(desired, n_samples // min_per_fold))`. Prevents errors when datasets are too small for the desired fold count.

### `_parse_month_labels(series) -> pl.Series | None`
Parses a temporal series into month-period strings (e.g., "2025-01") for stratification. Title-cases the input first (`str.to_titlecase()`) to restore month abbreviations after lowercase normalization, then tries 11 date formats (`_TEMPORAL_FORMATS`). Returns None if parsing fails.

### `stratified_subsample_idx(n_total, n_sample, month_values, seed) -> ndarray`
Returns indices for a month-stratified subsample that preserves the proportional distribution of months. Falls back to random sampling if no temporal information.

**Algorithm:**
1. Compute per-group allocation proportional to group size.
2. Distribute fractional remainders to groups with largest fractional parts.
3. Sample within each group.
4. Handles NaN group separately (rows that didn't parse).

### `Timer`
Context manager that records elapsed wall-clock time via `time.perf_counter()`. Used to instrument each pipeline stage.

---

## 13. Polars Compatibility Layer (polars_compat.py)

Centralizes all polars ↔ pandas/numpy boundary conversions. External libraries (LightGBM, scikit-learn, scipy) do not accept polars objects directly.

| Function | Purpose |
|---|---|
| `to_pandas_for_lgbm(df)` | `pl.DataFrame` → `pd.DataFrame` for LightGBM |
| `to_pandas_df(df)` | Generic polars → pandas DataFrame |
| `to_pandas_series(s)` | Generic polars → pandas Series |
| `to_numpy_for_scipy(s)` | Extract float64 numpy array from polars Series (drops nulls) |
| `is_numeric_col(s)` | Check if a polars Series has numeric dtype |
| `is_numeric_dtype(dtype)` | Check if a polars DataType is numeric |
| `to_numeric_coerce(s)` | Cast Series to Float64, coercing failures to null (mirrors `pd.to_numeric(errors="coerce")`) |
| `from_pandas_df(df)` | pandas → polars DataFrame |
| `from_pandas_series(s)` | pandas → polars Series |

**Note:** Several modules (`drift_detector.py`, `mitigation.py`, `feature_engineer.py`, `model.py`) contain their own `_ensure_pl_df()` and `_ensure_pl_series()` auto-conversion helpers as a transition safety net. These accept both pandas and polars inputs and return polars, guarding against mixed-type state during the refactor. They could be centralized into `polars_compat.py` in a future cleanup.

---

## 14. Data Flow Diagram

```
                        train.csv                    test.csv
                           |                            |
                     pl.read_csv()                 pl.read_csv()
                     (infer_schema_length=0)       (all Utf8)
                           |                            |
                     align_train_test_columns()----------+
                           |                            |
                     detect_feature_types(train, test_df=test)
                           |
                     Numeric coercion (main.py: Utf8 -> Float64)
                     downcast_dataframe() (Float64 -> Float32, etc.)
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
              |                                                               |
              v                                                               |
   apply_all_mitigations()                                                    |
    ├─ normalize_data_quality()                                               |
    ├─ re-check drift (modified cols only)                                    |
    ├─ progressive high-drift mitigation                                      |
    └─ progressive moderate-drift mitigation                                  |
              |                                                               |
    train_m, test_m, drop_cols                                                |
              |                                                               |
              v                                                               |
    Feature Strategy Tournament                                               |
    ├─ temporal holdout (last ~20% months)                                    |
    ├─ per-feature: {current, raw, drop}                                      |
    └─ 3-seed averaging, +0.005 threshold                                     |
              |                                                               |
         drop_cols (updated)                                                  |
              |                                                               |
              v                                                               |
    engineer_features() ──────────────────────────────────────────────── uses  |
    ├─ missing indicators                                                     |
    ├─ frequency encoding                                                     |
    ├─ ratio features                                                         |
    ├─ aggregation features                                                   |
    └─ near-duplicate removal                                                 |
              |                                                               |
    post-engineering drift check (drop high-drift engineered features)        |
              |                                                               |
              v                                                               |
  ┌─── prepare_features() ───────────────────────────────────────────── uses  |
  |   (target encoding, temporal parse,                                       |
  |    categorical encoding, dtype fix)                                       |
  |           |                                                               |
  |    X_train_m, y_train_m, X_test_m                                         |
  |           |                                                               |
  |    Feature selection                                                      |
  |    ├─ zero-importance pruning                                             |
  |    └─ top-8 feature interactions                                          |
  |           |                                                               |
  |    FE validation gate                                                     |
  |    ├─ compare with-FE vs without-FE on holdout                            |
  |    └─ swap to no-FE if it wins                                            |
  |           |                                                               |
  |    Iterative mitigation (importance-aware)                                |
  |    ├─ train model -> get importances                                      |
  |    ├─ find high-importance + still-drifted features                       |
  |    └─ quantile match or rank transform                                    |
  |           |                                                               |
  |    Train final model (uniform weights)                                    |
  |           |                                                               |
  |    Self-training (pseudo-labeling)                                        |
  |    ├─ score test, select P>=0.95 or P<=0.05                               |
  |    └─ retrain on augmented data (1 iteration)                             |
  |           |                                                               |
  |    evaluate_and_save()                                                    |
  |           |                                                               |
  |    prediction_mitigated.csv -> prediction.csv                             |
  |    model_mitigated.joblib   -> model.joblib                               |
  |                                                                           |
  └─── Baseline path (separate):                                             |
       prepare_features(train, test)  # unmitigated                           |
       train_model(X_train_b, y_train_b)  # uniform weights                  |
       evaluate_and_save(tag="baseline")                                      |
```

---

## 15. Key Design Principles

### No Hardcoded Column Names

Every column reference flows through `feature_types`. The three default names (`CustomerID`, `ChurnStatus`, `Month`) are only hints — if not found, the pipeline logs a warning and continues without that column type. This ensures the pipeline works on the hidden eval dataset with unknown column names.

### No Data Leakage

- Target is excluded from feature engineering (`skip_cols`), mitigation, and domain classification.
- Target encoding uses k-fold (each row's encoding comes from out-of-fold data only).
- Aggregation features group by feature columns, never the target.

### No Row Filtering

Mitigation and feature engineering only modify or add/drop **columns**, never rows. This preserves index alignment: `month_rank` and `X_train_m` all have exactly `len(train)` rows throughout. (Exception: self-training appends pseudo-labeled test rows to `X_train_m` for the final retrain.)

### Subsample-Then-Apply

Expensive operations (strategy evaluation, correlation matrices, domain classifiers) run on subsamples. Winners are applied to full data. This keeps the pipeline within the 10-minute budget even at 10M rows.

### Conservative Mitigation

Strategies are only applied if they demonstrably reduce drift (measured by re-running statistical tests after transformation). This prevents "cures worse than the disease" — a mitigation that increases drift would harm the model.

### Fixed Model, Variable Data

Since the LightGBM hyperparameters are frozen, all pipeline value comes from two levers:
1. **What the model sees** (feature engineering + feature selection + self-training augmentation)
2. **How clean the data is** (mitigation strategies reducing train/test distribution gaps)

### Polars-First with Pandas at Boundaries

The pipeline uses polars internally for all data manipulation. Conversion to pandas/numpy happens only at external library boundaries:
- LightGBM `.fit()` / `.predict_proba()` — via `_to_pandas_with_categoricals()`
- scipy statistical tests — via `.to_numpy()`
- sklearn metrics — via `.to_numpy()`

Each module has `_ensure_pl_df()` / `_ensure_pl_series()` guards that auto-convert pandas inputs to polars, providing backward compatibility during the transition from the pandas-based pipeline.

---

## Known Issues (polars-refactor branch)

1. **Duplicate helper functions**: `_ensure_pl_df` and `_ensure_pl_series` are defined in 4 separate files (`drift_detector.py`, `mitigation.py`, `feature_engineer.py`, `model.py`). These should be centralized in `polars_compat.py`.

2. **`pl.Object` dead code**: `model.py` checks for `pl.Object` dtype in categorical detection, but no column will have this dtype with the current loading strategy (everything is `pl.Utf8`).
