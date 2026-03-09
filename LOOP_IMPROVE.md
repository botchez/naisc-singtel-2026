# Loop: Perpetual Drift Improvement

You are iterating on the NAISC 2026 Singtel drift-intelligence pipeline. Each iteration you pick ONE high-impact improvement, implement it, validate it, and commit or revert.

---

## Hard Constraints (NEVER violate)

- **Fixed LightGBM hyperparameters** — do not add/change `n_estimators`, `max_depth`, `learning_rate`, or any other model param. Only `verbosity=-1`, `objective="binary"`, `is_unbalance=True`, `random_state=42`, `importance_type="gain"` are allowed.
- **No new pip dependencies** — only use packages already in `requirements.txt`.
- **No hardcoded column names** — all column detection must remain automatic.
- **Must generalize** — do not overfit to the public train/test split. The final evaluation uses hidden datasets.
- **CPU only** — no GPU code.
- **Pipeline must stay fully automated** — no manual intervention, no interactive prompts.

---

## Iteration Protocol

### STEP 1 — Record BEFORE metrics

Run the pipeline and capture AU-PRC output:

```bash
python ./src/main.py --train_data_filepath train.csv --test_data_filepath test.csv 2>&1 | tee /tmp/before_run.log
```

Extract and note down:
- Baseline AU-PRC (train & test)
- Mitigated AU-PRC (train & test)
- Winner strategy and CV AU-PRC
- Any warnings or errors

If the pipeline crashes, fix the crash FIRST (that counts as this iteration's improvement).

### STEP 2 — Read all source files

Re-read every source file fresh — code may have changed since your last iteration:
- `src/main.py`
- `src/profiler.py`
- `src/drift_detector.py`
- `src/mitigation.py`
- `src/feature_engineer.py`
- `src/model.py`
- `src/utils.py`

Understand the current state before making changes.

### STEP 3 — Pick ONE improvement

Select exactly ONE improvement from the candidate list below. Follow priority order (Section A before B, etc.). Skip candidates already implemented (check the code). Pick the one you judge will have the highest impact given current code state.

Do NOT implement multiple improvements in a single iteration.

### STEP 4 — Implement

- Make minimal, surgical changes.
- Follow existing code patterns and style.
- Add the improvement to the appropriate existing file — do NOT create new source files.
- Wrap new logic in try/except with sensible fallbacks so the pipeline never crashes from your change.
- Keep all existing functionality intact.

### STEP 5 — Validate AFTER metrics

Run the pipeline again:

```bash
python ./src/main.py --train_data_filepath train.csv --test_data_filepath test.csv 2>&1 | tee /tmp/after_run.log
```

Compare BEFORE vs AFTER:
- Primary metric: **Mitigated AU-PRC (train)** (since test target may not exist on hidden data)
- Secondary: **CV AU-PRC** of winning weight strategy
- Watch for: train-test gap widening (sign of overfitting)

### STEP 6 — Commit or Revert

**COMMIT** if ANY of these hold:
- Mitigated train AU-PRC improved by ≥ 0.001
- CV AU-PRC improved by ≥ 0.001
- Pipeline crash was fixed (regardless of metric change)
- A bug was fixed that improves correctness (regardless of metric change)

First, update `CHANGELOG.md` (create it if it doesn't exist). Append an entry at the top of the log (below the heading) in this exact format:

```markdown
## Iteration N — <Candidate ID>: <Short Title>
- **Status:** COMMITTED
- **BEFORE:** Baseline train=X.XXXX test=X.XXXX | Mitigated train=X.XXXX test=X.XXXX | CV AU-PRC=X.XXXX
- **AFTER:**  Baseline train=X.XXXX test=X.XXXX | Mitigated train=X.XXXX test=X.XXXX | CV AU-PRC=X.XXXX
- **Delta:**  Mitigated train ±X.XXXX | CV ±X.XXXX
- **Summary:** One-sentence description of what changed and why it helped.
```

Then commit everything together:

```bash
git add src/ CHANGELOG.md
git commit -m "loop: <short description of what changed>

BEFORE: Mitigated AU-PRC train=X.XXXX | CV AU-PRC=X.XXXX
AFTER:  Mitigated AU-PRC train=X.XXXX | CV AU-PRC=X.XXXX

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**REVERT** if:
- Any AU-PRC metric dropped by ≥ 0.002
- Pipeline crashes and you cannot fix it quickly
- Train-test gap increased by > 0.05 (overfitting)

Revert code changes but still log the attempt in `CHANGELOG.md`:

```markdown
## Iteration N — <Candidate ID>: <Short Title>
- **Status:** REVERTED
- **BEFORE:** Baseline train=X.XXXX test=X.XXXX | Mitigated train=X.XXXX test=X.XXXX | CV AU-PRC=X.XXXX
- **AFTER:**  Baseline train=X.XXXX test=X.XXXX | Mitigated train=X.XXXX test=X.XXXX | CV AU-PRC=X.XXXX
- **Delta:**  Mitigated train ±X.XXXX | CV ±X.XXXX
- **Reason:** Why it was reverted.
```

```bash
git checkout -- src/
git add CHANGELOG.md
git commit -m "loop: revert <candidate ID> — logged in CHANGELOG.md

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

**NEUTRAL** (metrics unchanged ± 0.001): commit anyway if the change improves code robustness or generalization, otherwise revert.

### STEP 7 — Research new candidates

After every iteration (commit or revert), spend a few minutes researching new improvement ideas. Goal: keep the ⬜ AVAILABLE pool stocked so you never run out of candidates.

**Search queries to run** (use WebSearch or WebFetch):
- `"covariate shift" "importance weighting" churn prediction site:arxiv.org`
- `"data drift" mitigation LightGBM tabular 2023 OR 2024 OR 2025`
- `"distribution shift" "feature engineering" "precision recall" imbalanced`
- `"BBSE" OR "RLLS" OR "BBSE" label shift correction tabular`
- `"domain adaptation" tabular classification sklearn`
- `"concept drift" detection "sliding window" OR "ADWIN" OR "Page-Hinkley"`

**Sources to check:**
- Papers With Code leaderboard for "dataset shift" and "domain adaptation"
- Recent NeurIPS / ICML / KDD proceedings on tabular drift
- scikit-learn / scipy function docs for statistical tests not yet used

**Evaluation criteria for a candidate to be added:**
1. Implementable with packages already in `requirements.txt` (no new deps)
2. Works on arbitrary column names (no hardcoding)
3. Addresses a gap visible in the current run's drift summary or known weakness
4. Not a duplicate of an existing or previously tried candidate

**Adding a new candidate:**

Append it to the appropriate section table (A–F) or create a new section (G, H, …) if it doesn't fit. Use this format:

```
| XX | ⬜ | Short name | One-sentence description of what it does and why it helps |
```

Also add a brief note below the table explaining where the idea came from (paper, observation, etc.) so future iterations have context.

**Commit the research update** if you added ≥ 1 new candidate:

```bash
git add LOOP_IMPROVE.md
git commit -m "loop: research — add <N> new candidates (<IDs>)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Current Baseline (Iteration 31)

- Baseline train=0.9259 test=0.7576
- Mitigated train=0.9227 test=0.8685
- CV AU-PRC=0.8672

---

## Improvement Candidates

Legend: ✅ COMMITTED | ❌ REVERTED | ⬜ AVAILABLE

### A — Drift Detection Enhancements (Priority 1)

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| A1 | ✅ | KS effect size | Supplement KS p-value with effect size (D-statistic magnitude) for better severity classification |
| A2 | ✅ | Cramér's V | Add Cramér's V for categorical drift alongside Chi² to capture association strength |
| A3 | ✅ | IQR shift detection | Detect shifts in interquartile range, not just distribution shape |
| A4 | ❌ | Hellinger distance | Add Hellinger distance as a bounded [0,1] alternative to JSD — redundant with JSD on this data |
| A5 | ✅ | Domain classifier feature importance | Log which features the domain classifier relies on most — these are the highest-drift features |
| A6 | ✅ | Label shift detection | Compare target class distribution between train/test (if target available) to detect prior probability shift |
| A7 | ✅ | Tail divergence | Measure drift specifically in distribution tails (1st/99th percentiles) where outliers live |

### B — Mitigation Strategies (Priority 2)

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| B1 | ❌ | Winsorization | Clip extreme values to configurable percentiles (e.g., 1st/99th) before other transforms |
| B2 | ⬜ | Z-score normalization | Standardize numerical features using train mean/std, apply same transform to test |
| B3 | ✅ | Quantile matching (moderate-only) | Align test quantiles to train quantiles via CDF-to-CDF mapping — moderate drift only (high-drift: catastrophic loss on full apply) |
| B4 | ⬜ | Adaptive clip percentiles | Instead of fixed clip bounds, choose percentiles based on drift severity |
| B5 | ⬜ | Ordinal encoding | For ordinal categoricals (Low/Medium/High), encode with ordered integers instead of frequency |
| B6 | ⬜ | Interaction-aware rare grouping | When grouping rare categories, consider co-occurrence with target, not just frequency |
| B7 | ⬜ | Monotone binning | Bin numerical features into buckets that maintain monotonic relationship with target |

### C — Feature Engineering (Priority 3)

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| C1 | ❌ | Difference features | For top correlated pairs, add (A − B) alongside existing ratio features — hurts generalization |
| C2 | ❌ | Polynomial interactions | Add A×B for top correlated pairs — destabilizes weight tournament, hurts generalization |
| C3 | ⬜ | Binned numericals | Discretize high-drift numericals into quantile bins to reduce sensitivity to exact values |
| C4 | ❌ | Group-level std/count | Add std and count aggregation features alongside existing group-level means |
| C5 | ❌ | Missing pattern count | Feature counting total number of missing values per row — redundant, hurts generalization |
| C6 | ⬜ | Drift magnitude features | Encode per-feature drift severity as a meta-feature for the model |

### D — Sample Weighting (Priority 4)

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| D1 | ⬜ | Inverse propensity weighting | Weight train samples by 1/P(train) from domain classifier — downweight train-specific samples |
| D2 | ⬜ | KDE-based weights | Use kernel density estimation to weight samples by test-likeness |
| D3 | ⬜ | Adaptive weight clipping | Clip extreme sample weights to prevent single-sample dominance |
| D4 | ⬜ | Soft sample selection | Instead of hard cutoff, use sigmoid-smoothed weights for borderline samples |
| D5 | ⬜ | Temporal stratified weighting | If temporal features exist, weight recent samples higher with exponential decay |

### E — Threshold & Calibration (Priority 5)

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| E1 | ✅ | Adaptive PSI thresholds | Scale PSI severity thresholds based on sample size (small samples need wider bands) |
| E2 | ⬜ | Permutation-based significance | Use permutation tests to establish feature-specific drift significance thresholds |
| E3 | ⬜ | Bootstrap severity calibration | Bootstrap confidence intervals for drift statistics to reduce false positives |

### F — Edge Cases & Robustness (Priority 6)

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| F1 | ✅ | Constant feature detection | Detect and drop features with zero or near-zero variance before drift tests |
| F2 | ✅ | Stratified CV for classification | Replaced KFold with StratifiedKFold in target encoding and weight tournament |
| F3 | ⬜ | Very high cardinality | Better handling for categoricals with >100 unique values (hash encoding, etc.) |
| F4 | ✅ | NaN-native handling | Ensure all drift tests handle NaN natively instead of dropping/filling |
| F5 | ✅ | Duplicate column detection | Detect and remove duplicate/near-duplicate columns |
| F6 | ✅ | Smoothed target encoding | Add Laplace or empirical Bayes smoothing to target encoding to reduce overfitting |
| F7 | ✅ | Zero-importance feature selection | Drop features with zero gain importance from a preliminary model |

### G — Researched Candidates (added during loop)

> New candidates discovered via research are appended here. Include a source note under each entry.

| # | Status | Candidate | Description |
|---|--------|-----------|-------------|
| G1 | ⬜ | KLIEP density ratio | Estimate train/test density ratio via KL-divergence-based kernel method — more statistically grounded than domain-classifier propensity |
| G2 | ⬜ | uLSIF / RuLSIF | Unconstrained/relative least-squares importance fitting — numerically stable density ratio estimation for sample weighting |
| G3 | ⬜ | BBSE label shift correction | Black-Box Shift Estimation: use confusion matrix on unlabelled test to correct predicted probabilities for prior probability shift |
| G4 | ⬜ | Wasserstein barycentre alignment | Align train/test marginals to their Wasserstein barycentre instead of shifting test to train — symmetric, avoids information loss |
| G5 | ⬜ | ADWIN concept drift alarm | Use ADWIN sliding-window detector on per-fold CV AU-PRC to trigger a warning when concept drift is detected across time segments |
| G6 | ⬜ | Platt scaling / isotonic calibration | Post-hoc probability calibration to correct for class-prior shift between train and test |
| G7 | ⬜ | Selective prediction (abstain on high-uncertainty) | Identify test samples where model is uncertain (low max probability) and flag rather than predict — reduces error on shifted test regions |
| G8 | ❌ | MI-based feature selection | Use `mutual_info_classif` to score features' nonlinear dependency with target, drop low-MI features — more aggressive than zero-importance filtering |
| G9 | ✅ | Native categorical handling | Use LightGBM's built-in Fisher-based categorical splits (`category` dtype) instead of target encoding — groups categories by gradient statistics at each node |
| G10 | ⬜ | Drift-aware MI hybrid gating | Combine drift severity with MI score in a 2D decision: keep high-MI features despite moderate drift, drop low-MI features even at moderate drift |
| G11 | ⬜ | Adversarial feature soft-weighting | Scale feature values by (1 - normalized_domain_importance) before model training, reducing drift-heavy feature influence without dropping them |
| G12 | ⬜ | CV permutation importance pruning | Use cross-validated permutation importance (shuffle-and-measure) to identify truly non-predictive features — more robust than single-model gain importance |

> Sources: Sugiyama et al. 2012 (KLIEP/uLSIF), Lipton et al. 2018 (BBSE), Villani 2009 (Wasserstein), Bifet & Gavaldà 2007 (ADWIN), Platt 1999, Geifman & El-Yaniv 2017 (selective prediction).
> G8: scikit-learn `mutual_info_classif` docs; Zaffalon 2002 (robust MI feature selection).
> G9: LightGBM Advanced Topics docs; Fisher 1958 optimal categorical splits.
> G10: Feature-based concept drift analysis (Neurocomputing 2024).
> G11: Kaggle adversarial validation notebooks; Carlmcbrideellis 2023; Analytics Vidhya 2023.
> G12: Altmann et al. 2010 (permutation importance); scikit-learn `permutation_importance` docs.

---

## Priority Order

1. **Pipeline crashes / bugs** — always fix first
2. **A (Detection)** — better drift detection informs everything downstream
3. **B (Mitigation)** — directly reduces drift impact
4. **C (Features)** — adds predictive signal
5. **D (Weighting)** — refines sample importance
6. **E (Thresholds)** — calibration fine-tuning
7. **F (Edge cases)** — robustness hardening

Within a category, prefer candidates that:
- Address the highest-severity drift features in the current run
- Are simplest to implement correctly
- Have the widest generalization benefit

---

## Anti-Patterns (DO NOT)

- **Overfit to public data** — if train AU-PRC jumps but you suspect it won't generalize, revert.
- **Swallow errors silently** — always log when a fallback is triggered.
- **Add complexity without measurable gain** — every change must earn its place.
- **Change multiple things at once** — one improvement per iteration so you can attribute results.
- **Modify LightGBM hyperparameters** — this violates competition rules.
- **Hardcode column names or dataset-specific logic** — the pipeline must work on any telecom churn dataset.
- **Create new source files** — add code to existing files following current patterns.
- **Skip reading source files** — always re-read in STEP 2, the code changes each iteration.

---

## Tracking

All iteration history is recorded in `CHANGELOG.md`. Before picking your next improvement in STEP 3:
- Read `CHANGELOG.md` to see what has already been tried
- Do NOT re-implement a previously reverted candidate unless you have a fundamentally different approach
- Use the cumulative metric deltas to gauge which categories are yielding the most gain

After every iteration, run STEP 7 to research and add ≥ 1 new candidate to section G (or A–F if it fits better). Update the status column in this file whenever a candidate is committed (✅) or reverted (❌).
