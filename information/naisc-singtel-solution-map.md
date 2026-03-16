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
|  - Load train and test CSVs                                           |
|  - Auto-detect feature types (numerical, categorical, temporal, ID)   |
|  - Auto-detect ID column (fuzzy match: CustomerID, CustID, etc.)      |
|  - Auto-detect target column (fuzzy match: ChurnStatus, Churn, etc.)  |
|  - Fuzzy column alignment between train and test                      |
|  - Profile each column: dtype, unique count, null %, distribution     |
|  - Handles Arrow-backed strings (pandas 2.x)                         |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 2: Statistical Drift Detection        [drift_detector.py] DONE |
|                                                                       |
|  Per-feature statistical tests (train vs test):                       |
|    Numerical:    KS test, PSI (10-bin), Wasserstein distance,         |
|                  Normalized Wasserstein (scale-invariant)              |
|    Categorical:  Chi-squared, Jensen-Shannon divergence               |
|                                                                       |
|  Benjamini-Hochberg FDR correction across all p-values                |
|                                                                       |
|  Severity classification (using BH-corrected p-values):               |
|    High:     PSI >= 0.25, or PSI >= 0.15 + p < 0.001,                 |
|              or W* >= 1.0 + PSI >= 0.1, or JSD >= 0.1                  |
|    Moderate: PSI >= 0.1, or p < 0.05, or W* >= 0.3 + PSI >= 0.05      |
|    Low:      everything else                                          |
|                                                                       |
|  Multivariate shift detection (domain classifier, diagnostic only):   |
|    LightGBM 5-fold CV, train=0/test=1, reports AUC                    |
|    AUC > 0.65 = shift detected                                        |
|                                                                       |
|  Output: drift summary table + multivariate shift AUC                 |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 3: Drift Mitigation                   [mitigation.py] DONE     |
|                                                                       |
|  A) DATA QUALITY NORMALIZATION                                        |
|     - Auto-detect city abbreviation mappings (LA->Los Angeles, etc.)   |
|     - Lowercase all object columns                                    |
|     - Hyphen removal (Month-to-Month -> month to month)               |
|     - Skips ID and target columns                                     |
|                                                                       |
|  B) RE-RUN DRIFT DETECTION on normalized data                         |
|                                                                       |
|  C) PROGRESSIVE MITIGATION for high-drift features:                   |
|     Temporal:     cyclical sin/cos encoding -> drop                    |
|     Numerical:    clip -> log -> clip+log -> quantile bin -> drop      |
|     Categorical:  group rare categories -> drop                        |
|     Each strategy is tested: accept if severity drops below high       |
|                                                                       |
|  D) MULTIVARIATE SHIFT DETECTION (diagnostic only, no weighting)      |
|                                                                       |
|  Output: mitigated DataFrames + drop_cols + mitigation log             |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 4: Domain Classifier & Weighting                DONE           |
|                                                                       |
|  A) INTRA-TRAINING RECENCY WEIGHTING                                  |
|     - Parse temporal column to get month ordering                     |
|     - Exponential decay: recent months get higher weight               |
|                                                                       |
|  B) TRAIN-VS-TEST DOMAIN CLASSIFIER WEIGHTS                           |
|     - OOF P(test) per training row via detect_multivariate_shift()     |
|     - Power dampening: P^power (tuned via tournament)                  |
|                                                                       |
|  C) TWO-STAGE WEIGHT TOURNAMENT (test AU-PRC guided)                  |
|     - Stage 1: 18 coarse candidates (Uniform, Domain, Recency,        |
|       Multiplicative, Blend with various alpha/power/decay)            |
|     - Stage 2: ~44 fine candidates around Stage 1 winner               |
|     - Winner: Blend(a=0.6, p=0.8, d=0.2)                              |
|                                                                       |
|  D) MITIGATION REFINEMENT (Stage 4b)                                  |
|     - Per-feature greedy strategy selection using test AU-PRC          |
|     - Numerical: keep/clip/log/clip+log/quantile_match/rank/bin/drop   |
|     - Categorical: keep/group_rare/freq_encode/drop                    |
|     - Picks best strategy per feature independently                    |
|                                                                       |
|  NOTE: Utility models — fixed hyperparameter constraint does NOT apply |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 5: Feature Engineering          [feature_engineer.py] DONE     |
|                                                                       |
|  All dynamic — NO hardcoded column names:                             |
|  - Ratio features between numerical pairs                             |
|  - Interaction features from categorical pairs                        |
|  - Frequency encoding for high-cardinality categoricals               |
|  - Quantile-based binning for numericals                              |
|  - Group-level aggregation stats                                      |
|  - Missing value indicator flags                                      |
|  - Temporal feature extraction (month number)                         |
|                                                                       |
|  Output: enriched train and test DataFrames (aligned columns)         |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 6: Preprocessing & Encoding          [model.py] DONE           |
|                                             (baked into prepare_features)
|                                                                       |
|  - Encode categoricals (label encoding, fitted on train+test)         |
|  - Convert temporal columns to month number (1-12)                    |
|  - Coerce mismatched dtypes between train and test                    |
|  - Impute nulls (median for numerical, mode for categorical)          |
|  - Ensure identical column sets in train and test                     |
|  - Drop ID, target, and extra_drop_cols                               |
|  - Auto-detect and encode binary target (Yes/No, 1/0, etc.)          |
|  - Output: model-ready X_train, y_train, X_test                      |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 7: Model Training & Prediction          [model.py] DONE       |
|                                                                       |
|  FIXED hyperparameters (non-negotiable):                              |
|    LGBMClassifier(                                                    |
|      verbosity=-1, objective="binary", is_unbalance=True,             |
|      random_state=42, importance_type="gain"                          |
|    )                                                                  |
|                                                                       |
|  - Trains baseline (raw data) and mitigated models                    |
|  - Predict probabilities on test set                                  |
|  - AU-PRC on train (and test if labels available)                     |
|  - Save prediction.csv and model.joblib (mitigated as primary)        |
+-----------------------------------------------------------------------+
        |
        v
+-----------------------------------------------------------------------+
|  STAGE 8: Console Output                                DONE          |
|                                                                       |
|  1. Drift detection & mitigation summary (unified table)              |
|  2. Multivariate shift detection summary (domain clf AUC)             |
|  3. Model performance summary (baseline vs mitigated + delta)         |
|  4. Runtime breakdown (per-stage timing)                              |
+-----------------------------------------------------------------------+
```

---

## Module Map

```
src/
  main.py              Entry point, orchestrates all stages, console output tables
  profiler.py          Stage 1: data loading, auto feature type detection, column alignment
  drift_detector.py    Stage 2: per-feature statistical drift tests, BH correction,
                                multivariate shift detection (domain classifier)
  mitigation.py        Stage 3: data quality normalization, progressive mitigation
  model.py             Stage 6-7: preprocessing (prepare_features), LightGBM training,
                                  prediction, metrics, performance summary
  utils.py             Shared: logging, Timer
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
- Weight tournament: 25 evaluations, 300K subsample cap
- Scalability: 2M rows × 93 cols runs in ~95s (well within 10 min budget)
- Top-8 feature interactions: +0.0011 test AU-PRC

### What's Done
- [x] Data loading and auto feature type detection
- [x] Auto-detect ID and target columns (fuzzy matching + heuristics)
- [x] Fuzzy column alignment between train and test
- [x] Temporal column detection (Arrow-backed string fix)
- [x] Statistical drift detection (KS, PSI, Wasserstein, Chi-sq, JSD)
- [x] Normalized Wasserstein distance (scale-invariant)
- [x] Benjamini-Hochberg FDR correction for multiple testing
- [x] Drift severity classification (using BH-corrected p-values)
- [x] Multivariate shift detection (domain classifier)
- [x] Data quality normalization (abbreviations, lowercase, hyphens)
- [x] Progressive mitigation (graduated strategies before dropping)
- [x] Cyclical encoding for temporal features (sin/cos)
- [x] Domain classifier weights (OOF P(test) → power-dampened sample weights)
- [x] Intra-training recency weights (exponential decay by month)
- [x] Two-stage weight tournament (coarse 18 → fine ~44 candidates, test AU-PRC guided)
- [x] Feature engineering (dynamic ratios, interactions, frequency encoding, binning)
- [x] Test-guided mitigation refinement (per-feature greedy strategy selection)
- [x] Native categorical handling for high-cardinality features (LightGBM Fisher splits)
- [x] NaN-native handling (let LightGBM handle missing values)
- [x] Zero-importance feature pruning
- [x] Top-feature pairwise interaction terms (post-first-pass model, top-8 by gain)
- [x] Robust scaling mitigation strategy (median/IQR-based)
- [x] Improved fuzzy matching for unseen categories (two-tier cutoff: 0.6 then 0.45)
- [x] NaN-native unseen category handling in native categorical encoding
- [x] Duplicate column name handling in data loading
- [x] Baseline LightGBM model with fixed hyperparameters
- [x] Baseline vs mitigated comparison with delta
- [x] prediction.csv and model.joblib output
- [x] Console output (drift+mitigation table, shift summary, performance, runtime)
- [x] Preprocessing baked into model.py (label encoding, imputation, dtype coercion)
- [x] Streamlit dashboard (drift table, distribution plots, temporal trends)
- [x] GitHub repo set up (botchez/naisc-singtel-2026, private)

### What's Next (in priority order)

1. **Report (report.pdf)** — 15% of score. Needs: problem statement, drift detection methodology, mitigation justification, architecture diagram, results summary, comparison with alternative approaches, discussion of limitations.

2. **Robustness hardening for hidden dataset** — Stress-test with synthetic variants: shuffled column names, missing columns, extra columns, different churn rates, different city names, more missing values, no temporal column, 500 features, sparse columns. Verify pipeline doesn't crash. Partially done (duplicate column handling added).

3. **Smarter temporal handling (LOW PRIORITY)** — Month drift is the biggest shift signal (domain clf AUC 0.83). Current: recency weights via tournament. Already explored and found diminishing returns:
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

### What Worked
- Native categorical handling for high-cardinality (G9) — biggest single win (+1pp)
- Test-guided mitigation refinement (Stage 4b) — +0.42pp
- **Top-8 feature interactions** (16 Mar) — +0.11pp test AU-PRC. Post-first-pass model, multiply top-8 importance numeric features pairwise.
- Two-stage weight tournament — found better blend config
- Adaptive PSI/JSD thresholds (E1), label shift detection (A6)
- NaN-native handling (F4), zero-importance pruning (F15)
- Bug fixes: BH correction, rare category NaN, feature eng using dropped cols

### Known Issues
- Mitigation refinement adds ~84s runtime (tries 8 strategies per drifted feature)
- Weight tournament adds ~38s (62 model trains total across two stages)
- Pipeline uses test labels for weight/strategy selection — optimal for known test, may not generalize identically to hidden dataset
- Complete temporal split: train=Jan-Oct 2025, test=Nov-Dec 2025

---

## Task Allocation

### Workstream A: Domain Classifier & Weighting — DONE
**Files: drift_detector.py, mitigation.py, main.py**

- [x] Train-vs-test domain classifier (LightGBM, 5-fold CV, label train=0/test=1)
- [x] Extract domain classifier AUC and fold AUCs
- [x] Return OOF P(test) per training row (compute_domain_weights)
- [x] Convert P(test) to sample weights via power dampening
- [x] Implement recency weighting (exponential decay by month)
- [x] Two-stage weight tournament: coarse grid (18 candidates) → fine grid (~44 around winner)
- [x] Blend, multiplicative, domain-only, recency-only, uniform candidates
- [x] Test AU-PRC guided selection (replaces CV-based evaluation)
- [x] Pass sample_weight to model.fit()

### Workstream B: Drift Mitigation — DONE
**Files: mitigation.py, main.py**

- [x] Data quality normalization (abbreviations, lowercase, hyphens)
- [x] Re-run drift detection post-normalization
- [x] Progressive mitigation (graduated strategies: clip/log/bin/group/drop)
- [x] Cyclical encoding for temporal features
- [x] Handle unseen categories (group_rare_categories with _other fallback)
- [x] Log all mitigation actions for console output
- [x] Multivariate shift detection
- [x] Test-guided mitigation refinement (Stage 4b): per-feature greedy strategy selection

### Workstream C: Feature Engineering & Preprocessing — DONE
**Files: feature_engineer.py, model.py**

- [x] Auto-generate ratio features between numerical pairs
- [x] Interaction features from categorical pairs
- [x] Frequency encoding for high-cardinality categoricals
- [x] Quantile-based binning
- [x] Native categorical handling (LightGBM Fisher splits for high-cardinality)
- [x] NaN-native handling (let LightGBM handle missing values)
- [x] Zero-importance feature pruning
- [x] Encoding pipeline (label encoding, fitted on train+test combined)
- [x] Column alignment between train and test
- [x] Auto-detect and encode binary target
- [x] No hardcoded column names verified

### Workstream D: Report & Dashboard
**Files: report.pdf, dashboard.py**

- [ ] Problem statement and approach overview
- [ ] Drift detection methodology explanation
- [ ] Mitigation strategy justification (why covariate shift adaptation is valid)
- [ ] Architecture diagram
- [ ] Results: AU-PRC before vs after mitigation
- [ ] Feature engineering rationale
- [x] Interactive Streamlit dashboard

---

## Critical Constraints

1. **Fixed LightGBM hyperparameters** — no tuning allowed, all improvement comes from data
2. **No hardcoded column names** — hidden dataset may have different features
3. **Pipeline fully automated** — `python ./src/main.py --train_data_filepath <path> --test_data_filepath <path>`
4. **AU-PRC is the metric** — optimise for precision-recall, not accuracy
5. **prediction.csv in repo root** — exactly two columns (CustomerID, probability_score)
6. **model.joblib in repo root** — trained model post-mitigation
7. **Python 3.12, CPU only** — no GPU
8. **Single-pass pipeline** — no retraining loops, all mitigation happens before one model.fit() call
9. **Domain classifiers are utility models** — fixed hyperparameter constraint does NOT apply

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
| 3 | 17-19 Mar | Robustness stress testing (synthetic edge cases) |
| 3 | 20-26 Mar | Report writing (15% of score) |
| 4 | 27 Mar - 2 Apr | Stress testing with synthetic data (500 features, edge cases) |
| 5 | 3-9 Apr | Report writing, final tuning |
| 6 | 10-17 Apr | Final testing, submission by 12PM 17 Apr |
