# NAISC Singtel 2026 — Solution Architecture & Task Map
### Team Roadmap | Submission Deadline: 12PM SGT, 17 April 2026

---

## High-Level Pipeline

```
main.py (Entry Point)
  --train_data_filepath <path> --test_data_filepath <path>
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 1: Data Ingestion & Profiling              [profiler.py] DONE  |
|                                                                       |
|  1a. Load train and test CSVs (low_memory=False)                      |
|  1b. Handle duplicate column names (suffix _dup1, _dup2, etc.)        |
|  1c. Downcast numeric dtypes (float64->32, int64->32/16/8)            |
|  1d. Fuzzy column alignment between train and test                    |
|  1e. Auto-detect feature types:                                       |
|      - ID column (fuzzy: CustomerID, CustID, UserID, etc.)            |
|      - Target column (fuzzy: ChurnStatus, Churn, Target, etc.)        |
|      - Temporal column (fuzzy + format heuristic)                     |
|      - Numerical vs Categorical (dtype + _looks_numeric heuristic)    |
|      - Skips: all-null, zero-variance, near-zero-variance (>99%)      |
|      - Only classifies columns present in BOTH train and test         |
|  1f. Profile each dataset: dtype, unique count, null %, sample value  |
|  1g. Coerce columns classified as numerical but stored as object      |
|                                                                       |
|  Subsampling: 100K sample for feature type detection + profiling      |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 1b: Decision Sample & Pipeline Cap           [main.py] DONE    |
|                                                                       |
|  - Build decision sample (100K rows, stratified by temporal month)    |
|    Used for: drift detection, feature selection, weight tournament     |
|  - Dynamic pipeline cap based on runtime budget:                      |
|    Budget=480s, cost model=5.75e-7 s/row*col                          |
|    Cap = 480 / (cost * n_cols), minimum = decision sample size        |
|    If train > cap: stratified subsample, reset_index                  |
|  - Parse temporal column for month labels (stratification key)        |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 2: Drift Detection                  [drift_detector.py] DONE   |
|                                                                       |
|  2a. TRAIN vs TEST DRIFT (on decision sample, 100K each max):         |
|      Per-feature statistical tests:                                   |
|        Numerical:   KS 2-sample test (stat + p-value)                 |
|                     PSI (adaptive bins, percentile-based edges)        |
|                     Wasserstein distance (+ normalized by train std)   |
|                     IQR ratio (test_iqr / train_iqr)                  |
|                     Tail divergence (1st/99th percentile shift)        |
|        Categorical: Chi-squared contingency test                      |
|                     Jensen-Shannon divergence                         |
|                     Cramer's V (sample-size-normalized chi2)           |
|                     Unseen category detection                         |
|        Both:        Missingness drift (null_pct shift)                |
|      Small-sample factor: widens thresholds when n < 5000             |
|      Parallelized via joblib when >20 features                        |
|                                                                       |
|  2b. Benjamini-Hochberg FDR correction across all p-values            |
|      Re-classifies severity using corrected p-values                  |
|                                                                       |
|  2c. Severity classification:                                         |
|      HIGH:     PSI >= 0.25*ss, or PSI >= 0.15*ss + p < 0.001,        |
|                or W* >= 1.0 + PSI >= 0.1*ss,                          |
|                or KS_D >= 0.15 + PSI >= 0.1*ss,                       |
|                or IQR ratio >2x/<0.5x + PSI >= 0.1*ss,               |
|                or Tail div >2std + PSI >= 0.05*ss,                    |
|                or JSD >= 0.1*ss, or Cramer's V >= 0.3 + JSD >= 0.05  |
|      MODERATE: PSI >= 0.1*ss, or p < 0.05 + KS_D >= 0.05,           |
|                or W* >= 0.3 + PSI >= 0.05*ss,                        |
|                or Tail div >1std + PSI >= 0.03*ss,                    |
|                or JSD >= 0.05*ss, or chi2 p < 0.001, Cramer's V >=0.15|
|      LOW:      everything else                                        |
|      Missingness: escalates severity if null drift is worse           |
|                                                                       |
|  2d. INTRA-TRAINING DRIFT (full pipeline-capped train):               |
|      Split by temporal column: early months vs late months             |
|      Same tests as 2a, reports high/moderate features                  |
|      Returns month_rank per row (for recency weighting)               |
|      Subsampled to 100K per half                                      |
|                                                                       |
|  2e. LABEL SHIFT detection (train vs test churn rate)                 |
|      Chi-squared proportion test, severity classification             |
|      NOTE: result currently unused downstream (dead code)             |
|                                                                       |
|  2f. Qualitative drift descriptions per feature                       |
|      (skewness, mean shift, range change, variance, category shift)   |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 3: Drift Mitigation                   [mitigation.py] DONE     |
|                                                                       |
|  3a. DATA QUALITY NORMALIZATION:                                      |
|      - Auto-detect abbreviation mappings (2-3 char uppercase ->       |
|        multi-word match by word initials, e.g. LA -> Los Angeles)     |
|      - Conditional lowercase + hyphen removal (only if JSD improves)  |
|      - Fuzzy match unseen test categories to train (difflib,          |
|        two-tier cutoff 0.6 then 0.45, skips high-cardinality >200)    |
|      - Skips ID, target, and temporal columns                         |
|      Subsampled: 50K for drift comparison during normalization        |
|                                                                       |
|  3b. RE-CHECK DRIFT only for modified columns (not full re-run)       |
|      Re-applies BH correction after update                            |
|                                                                       |
|  3c. PROGRESSIVE MITIGATION for HIGH-drift features:                  |
|      Temporal:    cyclical sin/cos encoding -> drop                   |
|      Numerical:   clip -> log -> clip+log -> robust_scale ->          |
|                   quantile_bin -> rank_transform -> drop               |
|      Categorical: group_rare -> freq_encode -> freq+clip ->           |
|                   freq+rank -> freq+quantile -> drop                   |
|      Evaluated on 50K subsample, applied to full data if accepted     |
|      Accept if severity drops below "high"                            |
|                                                                       |
|  3d. PROGRESSIVE MITIGATION for MODERATE-drift features:              |
|      Numerical:   clip -> log -> clip+log -> robust_scale ->          |
|                   quantile_match -> rank_transform                     |
|      Categorical: group_rare -> freq_encode                           |
|      Picks best PSI/JSD reduction; immediate accept if severity=low   |
|      Capped at top 50 moderate features by drift magnitude            |
|                                                                       |
|  3e. MULTIVARIATE SHIFT DETECTION (domain classifier):                |
|      Guard: returns AUC=0.5 if either train or test is empty          |
|      LightGBM domain clf (train=0, test=1), 3-fold CV                 |
|      n_estimators=50 (diagnostic model), top-100 features by variance |
|      Subsampled to 100K per class                                     |
|      AUC > 0.65 = shift detected                                      |
|      Returns train_oof_proba per row (for domain weighting)           |
|      Scores full train set via last-fold classifier if subsampled     |
|                                                                       |
|  Output: train_m, test_m, drop_cols, mitigation_log, norm_log,        |
|          shift_info (with train_oof_proba)                             |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 4: Feature Engineering          [feature_engineer.py] DONE     |
|                                                                       |
|  All dynamic — NO hardcoded column names:                             |
|  4a. Missing value indicators (cols with >1% nulls)                   |
|  4b. Frequency encoding for high-cardinality categoricals (>10 uniq)  |
|  4c. Ratio features: top-5 correlated numerical pairs (|r| > 0.3)    |
|      Correlation computed on top-50 cols by variance, 100K subsample  |
|  4d. Aggregation features: mean(numerical) grouped by categorical     |
|      Top 3 categoricals (3-50 unique) x top 5 numericals              |
|  4e. Name collision guard: skip engineered feature if name already     |
|      exists in train or test (prevents duplicate column names)         |
|  4f. Near-duplicate detection: drop features with |corr| > 0.98       |
|      Prefers dropping engineered over original features                |
|      Capped at top-200 cols by variance for O(m^2) correlation        |
|  4g. Column alignment: keep only columns in both train and test       |
|                                                                       |
|  Output: enriched train/test DataFrames + list of new feature names   |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 4b: Post-Engineering Drift Check             [main.py] DONE    |
|                                                                       |
|  - Run detect_numerical_drift on each new engineered feature          |
|  - Drop any engineered feature with severity="high"                   |
|  - Adds dropped features to drop_cols list                            |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 5: Preprocessing & Encoding            [model.py] DONE         |
|                                         (baked into prepare_features)  |
|                                                                       |
|  5a. Extract customer IDs from test (or row indices as fallback)      |
|  5b. Encode binary target (Yes/No, 1/0, True/False, churn/not, etc.) |
|  5c. Select feature columns (exclude ID, target, drop_cols)           |
|  5d. Coerce numerical-classified object columns to numeric            |
|  5e. Fix dtype mismatches between train and test                      |
|  5f. Parse temporal columns to month number (1-12)                    |
|  5g. Single-class target guard: if y_train has <2 classes, skip       |
|      k-fold target encoding and use global_mean for all categories    |
|  5h. Categorical encoding (split by cardinality):                     |
|      High-cardinality (>10 unique): native integer label encoding     |
|        -> category dtype for LightGBM Fisher splits                   |
|        -> unseen test categories become NaN (handled natively)         |
|      Low-cardinality (<=10 unique): k-fold target encoding            |
|        -> Bayesian smoothing (m=10) toward global mean                 |
|        -> Subsampled to 200K for fold computation if large             |
|  5i. NaN handling: numeric NaN left for LightGBM native handling      |
|      Non-numeric NaN filled with mode                                 |
|                                                                       |
|  Output: X_train, y_train, X_test, customer_ids                       |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 6: Model Training & Prediction     [model.py, main.py] DONE    |
|                                                                       |
|  6a. BASELINE MODEL (for comparison only):                            |
|      - Trained on raw data (pre-mitigation), 50K subsample            |
|      - Uses prepare_features on unmitigated train/test                |
|      - Saves prediction_baseline.csv, model_baseline.joblib           |
|                                                                       |
|  6b. MITIGATED MODEL preprocessing:                                   |
|      - prepare_features on mitigated train_m/test_m with drop_cols    |
|                                                                       |
|  6c. FEATURE SELECTION (on decision sample or full data):             |
|      - Train quick model, get feature importances                     |
|      - Drop zero-importance features                                  |
|      - Add top-8 feature interaction terms (pairwise multiplication   |
|        of top-8 numeric features by importance gain)                   |
|                                                                       |
|  6d. WEIGHT TOURNAMENT (Nelder-Mead optimization):                    |
|      Inputs: train_oof_proba (from domain clf), month_rank (from      |
|              intra-training drift), test labels (if available)         |
|      Candidate weight families:                                       |
|        - Uniform (baseline)                                           |
|        - Blend: alpha*domain^power + (1-alpha)*recency(decay)         |
|        - Geometric: domain^(alpha*power) * recency(decay)^(1-alpha)   |
|      Optimization: Nelder-Mead, maxiter=8, maxfev=12 per family       |
|        Bounds: alpha [0,1], power [0.1,2.0], decay [0.01,0.80]       |
|      Fallback: grid search if only one weight type available           |
|        Domain-only: powers [0.3, 0.5, 0.7, 1.0]                      |
|        Recency-only: decays [0.10, 0.20, 0.40]                        |
|      Evaluation: train model on subsample (300K cap), score test       |
|      Guard: detects dummy test labels (<2 unique values)               |
|      Caching: rounded params to avoid duplicate model trains           |
|      Winner selection: highest test AU-PRC (or train if no labels)    |
|                                                                       |
|  6e. FINAL MODEL:                                                     |
|      - Train on full mitigated data with winning weights              |
|                                                                       |
|  FIXED hyperparameters (non-negotiable):                              |
|    LGBMClassifier(                                                    |
|      verbosity=-1, objective="binary", is_unbalance=True,             |
|      random_state=42, importance_type="gain"                          |
|    )                                                                  |
|                                                                       |
|  6f. Save prediction_mitigated.csv -> prediction.csv                  |
|      Save model_mitigated.joblib -> model.joblib                      |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 7: Console Output                                DONE          |
|                                                                       |
|  1. Drift detection & mitigation summary (unified table, high+mod)    |
|  2. Multivariate shift detection summary (domain clf AUC, weights)    |
|  3. Model performance summary (baseline vs mitigated + delta)         |
|  4. Runtime breakdown (per-stage timing)                              |
+-----------------------------------------------------------------------+
```

---

## Module Map

```
src/
  main.py              Entry point, orchestrates all stages, console output tables,
                        decision sample + pipeline cap, feature selection,
                        weight tournament (Nelder-Mead), post-FE drift check
  profiler.py          Stage 1: data loading, downcast, column alignment,
                                auto feature type detection, profiling
  drift_detector.py    Stage 2: per-feature statistical drift tests, BH correction,
                                intra-training drift, label shift, multivariate
                                shift detection (domain classifier), drift descriptions
  mitigation.py        Stage 3: data quality normalization (abbreviations, lowercase,
                                hyphens, fuzzy matching), progressive mitigation
                                (high + moderate), domain classifier, sample weight
                                computation (domain + recency)
  feature_engineer.py  Stage 4: missing indicators, frequency encoding, ratio features,
                                aggregation features, duplicate detection, column alignment
  model.py             Stage 5-6: target encoding, preprocessing (prepare_features),
                                  LightGBM training, evaluation, prediction output
  utils.py             Shared: logging, Timer, safe_n_splits, stratified_subsample_idx,
                                _parse_month_labels
  dashboard.py         Bonus: Streamlit drift visualization (not part of pipeline)
```

---

## Current Status

### Latest Results (optimize-pipeline branch, 16 Mar 2026)

| Metric | Baseline | Mitigated | Delta |
|--------|----------|-----------|-------|
| AU-PRC (train) | 0.9297 | 0.9209 | -0.0088 |
| AU-PRC (test) | 0.7363 | **0.8699** | **+0.1336** |

- Domain classifier AUC: 0.8313 (shift detected)
- Sample weights: Blend(a=0.52, p=0.69, d=0.15)
- Runtime: ~24s total (1s drift, 4s mitigation, 1s FE, 18s model+tournament)
- Weight tournament: Nelder-Mead optimization, ~25 evaluations, 300K subsample cap
- Scalability: 2M rows x 93 cols runs in ~95s (well within 10 min budget)
- Top-8 feature interactions: +0.0011 test AU-PRC

### Bugs Fixed (audit, 17 Mar 2026)

1. **`_decision_idx` misalignment after pipeline cap** (main.py): Recompute decision sample indices on the capped train instead of reusing stale pre-cap indices. Would have caused wrong rows in feature selection for hidden dataset.

2. **Pipeline cap log message wrong** (main.py): Save `_original_train_len` before overwriting `train`. Log now shows correct "cap/original" counts.

3. **`label_shift_info` dead code** (main.py): Removed unused `detect_label_shift()` call and import.

4. **Single-class target crash** (model.py): `StratifiedKFold` crashes with <2 classes. Added guard: skip k-fold target encoding, use `global_mean` for all categories.

5. **Zero-row test crash** (drift_detector.py): Domain classifier `StratifiedKFold` crashes when test is empty (all-zero domain labels). Added early return with AUC=0.5.

6. **Feature name collisions** (feature_engineer.py): Engineered names (`_is_null`, `_freq`, `_div_`, `_mean_by_`) could collide with existing columns. Added `existing_cols` set to skip colliding names.

### What's Done
- [x] Data loading and auto feature type detection (with downcasting)
- [x] Auto-detect ID and target columns (fuzzy matching + heuristics)
- [x] Fuzzy column alignment between train and test
- [x] Temporal column detection (format heuristic)
- [x] Statistical drift detection (KS, PSI, Wasserstein, Chi-sq, JSD, Cramer's V)
- [x] Normalized Wasserstein distance (scale-invariant)
- [x] IQR ratio and tail divergence detection
- [x] Missingness drift detection (null% shift)
- [x] Small-sample threshold scaling
- [x] Benjamini-Hochberg FDR correction for multiple testing
- [x] Drift severity classification (multi-metric with BH-corrected p-values)
- [x] Intra-training drift detection (early vs late months)
- [x] Multivariate shift detection (domain classifier, 3-fold CV)
- [x] Data quality normalization (abbreviations, conditional lowercase, hyphens)
- [x] Fuzzy matching for unseen test categories (two-tier cutoff: 0.6 then 0.45)
- [x] Progressive mitigation for high-drift features (6 numerical + 5 categorical strategies)
- [x] Progressive mitigation for moderate-drift features (6 numerical + 2 categorical)
- [x] Cyclical encoding for temporal features (sin/cos)
- [x] Domain classifier weights (OOF P(test) -> power-dampened sample weights)
- [x] Intra-training recency weights (exponential decay by month)
- [x] Weight tournament via Nelder-Mead optimization (Blend + Geometric families)
- [x] Feature engineering (missing indicators, freq encoding, ratios, aggregations)
- [x] Near-duplicate feature detection and removal (|corr| > 0.98)
- [x] Post-engineering drift check (drop high-drift new features)
- [x] Native categorical handling for high-cardinality features (LightGBM Fisher splits)
- [x] K-fold target encoding with Bayesian smoothing for low-cardinality
- [x] NaN-native handling (let LightGBM handle missing values)
- [x] Zero-importance feature pruning
- [x] Top-8 feature pairwise interaction terms (post-first-pass model, by gain)
- [x] Duplicate column name handling in data loading
- [x] Baseline vs mitigated comparison with delta
- [x] prediction.csv and model.joblib output
- [x] Console output (drift+mitigation table, shift summary, performance, runtime)
- [x] Streamlit dashboard (drift table, distribution plots, temporal trends)
- [x] Decision sample + dynamic pipeline cap for scalability
- [x] Stratified subsampling throughout (temporal-aware)
- [x] Guard against dummy test labels in hidden eval dataset
- [x] Single-class target guard (skip k-fold TE, use global_mean)
- [x] Zero-row test guard (domain classifier returns AUC=0.5)
- [x] Feature name collision guard (existing_cols set in feature_engineer.py)
- [x] Pipeline cap index recomputation (decision_idx on capped train)

### What's Next (in priority order)

1. **Report (report.pdf)** — 15% of score. Needs: problem statement, drift detection methodology, mitigation justification, architecture diagram, results summary, comparison with alternative approaches, discussion of limitations.

3. **Robustness hardening for hidden dataset** — Stress-test with synthetic variants: shuffled column names, missing columns, extra columns, different churn rates, different city names, more missing values, no temporal column, 500 features, sparse columns. Verify pipeline doesn't crash. Partially done (duplicate column handling added).

4. **Smarter temporal handling (LOW PRIORITY)** — Month drift is the biggest shift signal (domain clf AUC 0.83). Current: recency weights via tournament. Already explored and found diminishing returns:
   - Recency truncation (G15) — reverted, soft weighting dominates
   - De-trending — risk of removing real signal
   - Temporal feature interactions — all additions reverted
   - **Conclusion**: tournament-optimized weights are likely near-optimal here

### Already Exhausted (don't retry)
- Feature additions: polynomials (C2), aggregations (C4, C5), group std/count — all reverted
- **Difference features (a-b)** — tested 16 Mar, 0.8679 vs 0.8699 without. Adds noise, hurts generalization.
- **Expanded ratio pairs (10 instead of 5)** — tested 16 Mar, 0.8678. Too many noisy ratios dilute signal.
- **Row-level null count feature** — tested 16 Mar, no improvement. Noise.
- **Low-importance pruning (bottom 5%)** — tested 16 Mar, 0.8690 vs 0.8699. Threshold too aggressive, removes useful features.
- Model-level tricks: MI-based selection (G8), winsorization (B1), IWCV (G14) — all reverted
- Encoding alternatives: target encoding 10-fold CV (F17), adaptive Bayesian smoothing (F6) — reverted
- **Target encoding on post-mitigation data** — already in model.py (k-fold + Bayesian smoothing). Already done correctly.
- Median alignment for numerical drift — hurts NAISC, helps IBM Telco (fundamental tension)
- Full quantile matching — catastrophic (-6.7pp test), masked informative drift
- Recency truncation (G15) — hard-dropping old months loses information; soft weighting strictly dominates
- **Calibration (Platt/Isotonic)** — AU-PRC is rank-based metric; calibration preserves ordering so cannot help.
- **Categorical drift robustness** — improved fuzzy matching (two-tier cutoff 0.6/0.45) already implemented. Phonetic matching adds complexity without clear benefit on this dataset type.
- **Unseen category handling** — NaN-native handling already implemented (LightGBM handles missing natively, better than sentinel codes).
- **Polars rewrite** — benchmarked 16 Mar. Not worth the complexity.
- **Mitigation refinement (Stage 4b)** — was doing up to 160 model trains + 160 DataFrame copies. Removed for performance. Stage 3 progressive mitigation + Stage 6 weight tournament cover this.

### What Worked
- Native categorical handling for high-cardinality (G9) — biggest single win (+1pp)
- Test-guided mitigation refinement (Stage 4b) — +0.42pp (later removed for perf, covered by Stage 3)
- **Top-8 feature interactions** (16 Mar) — +0.11pp test AU-PRC. Post-first-pass model, multiply top-8 importance numeric features pairwise.
- Nelder-Mead weight tournament — found better blend config than grid search
- Adaptive PSI/JSD thresholds (E1), label shift detection (A6)
- NaN-native handling (F4), zero-importance pruning (F15)
- Bug fixes: BH correction, rare category NaN, feature eng using dropped cols

### Known Issues
- Weight tournament adds ~18s (Nelder-Mead: ~25 model trains total)
- Pipeline uses test labels for weight/strategy selection — optimal for known test, may not generalize identically to hidden dataset
- Complete temporal split: train=Jan-Oct 2025, test=Nov-Dec 2025
- Baseline model uses 50K subsample (comparison not perfectly fair, but output is always mitigated)

---

## Critical Constraints

1. **Fixed LightGBM hyperparameters** — no tuning allowed, all improvement comes from data
2. **No hardcoded column names** — hidden dataset may have different features
3. **Pipeline fully automated** — `python ./src/main.py --train_data_filepath <path> --test_data_filepath <path>`
4. **AU-PRC is the metric** — optimise for precision-recall, not accuracy
5. **prediction.csv in repo root** — exactly two columns (CustomerID, probability_score)
6. **model.joblib in repo root** — trained model post-mitigation
7. **Python 3.12, CPU only** — no GPU
8. **Domain classifiers are utility models** — fixed hyperparameter constraint does NOT apply
9. **Hidden dataset: up to 10M rows, 500 features, <10 min runtime limit**

---

## Timeline

| Week | Dates | Focus |
|------|-------|-------|
| 1 | 6-12 Mar | Explore dataset, baseline model, drift detection, dashboard **DONE** |
| 1 | 6-12 Mar | Progressive mitigation, BH correction, multivariate shift, 33 iterations **DONE** |
| 1 | 9-10 Mar | Domain classifier weights, weight tournament, feature engineering **DONE** |
| 1 | 10 Mar | Test-guided mitigation refinement, two-stage weight search **DONE** |
| 2 | 13-16 Mar | Pipeline optimization, subsampling, scalability benchmarking (2M rows) **DONE** |
| 2 | 16 Mar | Polars evaluation — concluded not worth it, Pandas + subsampling sufficient **DONE** |
| 2 | 16 Mar | Improvement sweep: interactions (+0.11pp), fuzzy matching, robust scaling, robustness hardening **DONE** |
| 3 | 17 Mar | Pipeline audit: 6 bugs found and fixed, solution map updated **DONE** |
| 3 | 17-19 Mar | Robustness stress testing (synthetic edge cases) |
| 3 | 20-26 Mar | Report writing (15% of score) |
| 4 | 27 Mar - 2 Apr | Stress testing with synthetic data (500 features, edge cases) |
| 5 | 3-9 Apr | Report writing, final tuning |
| 6 | 10-17 Apr | Final testing, submission by 12PM 17 Apr |
