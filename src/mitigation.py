import numpy as np
import pandas as pd
from utils import logger

_MAX_EVAL_SAMPLE = 50_000


def _eval_subsample(train_col, test_col, max_n=_MAX_EVAL_SAMPLE, seed=42):
    """Subsample columns for fast strategy evaluation."""
    rng = np.random.RandomState(seed)
    tr, te = train_col, test_col
    if len(tr) > max_n:
        tr = tr.iloc[rng.choice(len(tr), max_n, replace=False)]
    if len(te) > max_n:
        te = te.iloc[rng.choice(len(te), max_n, replace=False)]
    return tr, te


def compute_domain_weights(
    train_oof_proba: np.ndarray, power: float = 0.5
) -> np.ndarray:
    """Convert domain classifier P(test) probabilities to sample weights.

    Higher P(test) = training row looks more like test = more important.
    Power dampening (default 0.5) prevents extreme weight ratios.

    Returns weights normalized to mean=1.0.
    """
    if train_oof_proba is None or len(train_oof_proba) == 0:
        return None

    # Clip to avoid zero weights
    proba = np.clip(train_oof_proba, 1e-6, 1.0 - 1e-6)

    # Power dampening: soften the weight distribution
    weights = proba ** power

    # Clip extreme values at 99th percentile
    cap = np.percentile(weights, 99)
    weights = np.clip(weights, None, cap)

    # Normalize to mean=1.0
    weights = weights / weights.mean()

    return weights


def compute_recency_weights(
    month_rank: pd.Series, decay: float = 0.15
) -> np.ndarray:
    """Compute recency weights from month rank using exponential decay.

    Most recent months get weight ~1.0; older months decay down.
    Returns weights normalized to mean=1.0.
    """
    if month_rank is None or len(month_rank) == 0:
        return None

    max_rank = month_rank.max()
    if max_rank == 0:
        return None

    # Exponential decay: recent rows weighted higher
    weights = np.exp(decay * (month_rank.values - max_rank))

    # Normalize to mean=1.0
    weights = weights / weights.mean()

    return weights


def detect_abbreviation_mappings(
    train: pd.DataFrame, test: pd.DataFrame, col: str
) -> dict[str, str]:
    """Auto-detect city abbreviation-to-full-name mappings from the data.

    Candidate abbreviations: 2-3 uppercase alpha strings (e.g. "LA", "NYC").
    Candidate full names: multi-word values (contain a space).
    Matches abbreviation to full name whose word-initial letters match.
    Ties broken by combined frequency.
    """
    _ABB_SAMPLE = 50_000
    tr_vals = train[col].dropna()
    te_vals = test[col].dropna()
    if len(tr_vals) > _ABB_SAMPLE:
        tr_vals = tr_vals.sample(n=_ABB_SAMPLE, random_state=42)
    if len(te_vals) > _ABB_SAMPLE:
        te_vals = te_vals.sample(n=_ABB_SAMPLE, random_state=42)
    all_values = set(tr_vals.unique()) | set(te_vals.unique())

    abbreviations = [v for v in all_values if len(v) in (2, 3) and v.isupper() and v.isalpha()]
    full_names = [v for v in all_values if " " in v]

    if not abbreviations or not full_names:
        return {}

    # Combined frequency across both datasets (subsampled)
    combined_freq = tr_vals.value_counts().add(te_vals.value_counts(), fill_value=0)

    mapping = {}
    for abbrev in abbreviations:
        matches = [
            fn for fn in full_names
            if "".join(w[0].upper() for w in fn.split()) == abbrev
        ]
        if matches:
            best = max(matches, key=lambda fn: combined_freq.get(fn, 0))
            mapping[abbrev] = best

    return mapping


def normalize_data_quality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Normalize data quality: city abbreviation mapping, lowercase, hyphen removal.

    Skips ID and target columns to preserve values like 'Yes'/'No'.
    Returns (train_clean, test_clean, norm_log) where norm_log maps column names
    to lists of normalization action labels applied.
    """
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    train_clean = train.copy(deep=False)
    test_clean = test.copy(deep=False)
    norm_log: dict[str, list[str]] = {}

    # Auto-detect abbreviation mappings on all categorical object columns
    for col in feature_types.get("categorical", []) + feature_types.get("temporal", []):
        if col in skip_cols or col not in train_clean.columns or col not in test_clean.columns:
            continue
        if train_clean[col].dtype != object:
            continue
        abbreviation_map = detect_abbreviation_mappings(train_clean, test_clean, col)
        if abbreviation_map:
            for df in [train_clean, test_clean]:
                df[col] = df[col].replace(abbreviation_map)
            logger.info(f"Auto-detected abbreviation mappings in '{col}': {abbreviation_map}")
            norm_log.setdefault(col, []).append("Abbreviations standardized")

    # Conditional lowercase + hyphen removal: only apply if it reduces drift
    from drift_detector import detect_categorical_drift
    normalized = []
    skipped_norm = []
    cat_cols = set(feature_types.get("categorical", []) + feature_types.get("temporal", []))

    # Pre-compute subsample indices for drift comparison on large datasets
    _NORM_SAMPLE = 50_000
    _tr_norm_idx = np.random.RandomState(42).choice(
        len(train_clean), min(_NORM_SAMPLE, len(train_clean)), replace=False
    ) if len(train_clean) > _NORM_SAMPLE else None
    _te_norm_idx = np.random.RandomState(42).choice(
        len(test_clean), min(_NORM_SAMPLE, len(test_clean)), replace=False
    ) if len(test_clean) > _NORM_SAMPLE else None

    for col in train_clean.columns:
        if col in skip_cols or col not in cat_cols:
            continue
        if train_clean[col].dtype == object:
            # Subsample for drift comparison
            tr_sample = train_clean[col].iloc[_tr_norm_idx] if _tr_norm_idx is not None else train_clean[col]
            te_sample = test_clean[col].iloc[_te_norm_idx] if _te_norm_idx is not None else test_clean[col]

            # Compute drift before normalization (on subsample)
            before = detect_categorical_drift(tr_sample, te_sample)
            before_jsd = before.get("jsd", 0) or 0

            # Trial normalization on subsample for comparison
            tr_trial_s = tr_sample.astype(str).str.lower().str.replace("-", " ", regex=False)
            te_trial_s = te_sample.astype(str).str.lower().str.replace("-", " ", regex=False)
            after = detect_categorical_drift(tr_trial_s, te_trial_s)
            after_jsd = after.get("jsd", 0) or 0

            if after_jsd <= before_jsd:
                # Apply to full data
                train_clean[col] = train_clean[col].astype(str).str.lower().str.replace("-", " ", regex=False)
                test_clean[col] = test_clean[col].astype(str).str.lower().str.replace("-", " ", regex=False)
                normalized.append(col)
                norm_log.setdefault(col, []).extend(["Lowercased", "Hyphens removed"])
            else:
                skipped_norm.append(col)

    if normalized:
        logger.info(f"Normalized {len(normalized)} columns: {normalized}")
    if skipped_norm:
        logger.info(f"Skipped normalization for {len(skipped_norm)} columns (would increase drift): {skipped_norm}")

    # Fuzzy match: map unseen test categories to closest train category
    # Skip for high-cardinality columns — fuzzy matching is O(unseen × train_cats)
    # and only useful for small categoricals (city names, state codes, etc.)
    import difflib
    _MAX_FUZZY_TRAIN_CATS = 200
    _MAX_FUZZY_UNSEEN = 50
    fuzzy_skipped = 0
    for col in cat_cols:
        if col in skip_cols or col not in train_clean.columns or col not in test_clean.columns:
            continue
        if train_clean[col].dtype != object:
            continue
        tr_s = train_clean[col].iloc[_tr_norm_idx].dropna() if _tr_norm_idx is not None else train_clean[col].dropna()
        te_s = test_clean[col].iloc[_te_norm_idx].dropna() if _te_norm_idx is not None else test_clean[col].dropna()
        train_cats = set(tr_s.unique())
        test_cats = set(te_s.unique())
        unseen = test_cats - train_cats
        if not unseen:
            continue
        # Skip if too many categories — fuzzy matching is only for low-cardinality typo fixes
        if len(train_cats) > _MAX_FUZZY_TRAIN_CATS or len(unseen) > _MAX_FUZZY_UNSEEN:
            fuzzy_skipped += 1
            continue
        fuzzy_map = {}
        for cat in unseen:
            matches = difflib.get_close_matches(cat, list(train_cats), n=1, cutoff=0.6)
            if matches:
                fuzzy_map[cat] = matches[0]
        if fuzzy_map:
            test_clean[col] = test_clean[col].replace(fuzzy_map)
            logger.info(f"Fuzzy matched '{col}': {fuzzy_map}")
            norm_log.setdefault(col, []).append(f"Fuzzy matched: {fuzzy_map}")
    if fuzzy_skipped:
        logger.info(f"Skipped fuzzy matching for {fuzzy_skipped} high-cardinality columns")
    return train_clean, test_clean, norm_log


# ── Strategy functions ────────────────────────────────────────────────────


def median_align(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Shift test values so that test median matches train median."""
    train_vals = pd.to_numeric(train_col, errors="coerce").dropna()
    test_vals = pd.to_numeric(test_col, errors="coerce").dropna()
    if len(train_vals) == 0 or len(test_vals) == 0:
        return train_col, test_col
    shift = train_vals.median() - test_vals.median()
    if abs(shift) < 1e-10:
        return train_col, test_col  # no shift needed
    test_aligned = test_col.copy()
    test_aligned = pd.to_numeric(test_aligned, errors="coerce") + shift
    return train_col, test_aligned


def clip_outliers(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Clip both columns to [1st, 99th] percentile of train."""
    vals = pd.to_numeric(train_col, errors="coerce").dropna().values.astype(float)
    lo, hi = np.percentile(vals, [1, 99])
    return train_col.clip(lo, hi), test_col.clip(lo, hi)


def log_transform(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series] | None:
    """Apply log1p if all values >= 0; return None to skip otherwise."""
    if (train_col.dropna() < 0).any() or (test_col.dropna() < 0).any():
        return None
    return np.log1p(train_col), np.log1p(test_col)


def quantile_bin(
    train_col: pd.Series, test_col: pd.Series, n_bins: int = 10
) -> tuple[pd.Series, pd.Series]:
    """Discretize into quantile bins defined by train edges."""
    vals = pd.to_numeric(train_col, errors="coerce").dropna().values.astype(float)
    # Adapt bins to sample size
    n_bins = min(n_bins, max(2, len(vals) // 20))
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, percentiles)
    edges = np.unique(edges)
    return (
        pd.Series(np.digitize(train_col, edges), index=train_col.index, name=train_col.name),
        pd.Series(np.digitize(test_col, edges), index=test_col.index, name=test_col.name),
    )


def group_rare_categories(
    train_col: pd.Series, test_col: pd.Series, min_freq: float = 0.01
) -> tuple[pd.Series, pd.Series]:
    """Merge categories below min_freq in train + unseen-in-train into '_other'."""
    freq = train_col.value_counts(normalize=True)
    rare_cats = set(freq[freq < min_freq].index)
    train_cats = set(train_col.dropna().unique())

    def _remap(s: pd.Series) -> pd.Series:
        return s.map(lambda v: v if pd.isna(v) else ("_other" if v in rare_cats or v not in train_cats else v))

    return _remap(train_col), _remap(test_col)


def encode_temporal_cyclical(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Parse temporal -> month -> sin/cos pair. Returns (train_sin, train_cos, test_sin, test_cos)."""
    def _to_month(s: pd.Series) -> pd.Series:
        # Title-case to restore month abbreviations after lowercase normalization
        s_title = s.astype(str).str.title()
        for fmt in ["%y-%b", "%b-%y", "%Y-%m", "%m-%Y", "%b %Y", "%Y-%b",
                    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
            parsed = pd.to_datetime(s_title, format=fmt, errors="coerce")
            if parsed.notna().sum() > len(s) * 0.5:
                return parsed.dt.month.fillna(1).astype(float)
        return pd.Series(np.ones(len(s)), index=s.index)

    train_month = _to_month(train_col)
    test_month = _to_month(test_col)

    train_sin = np.sin(2 * np.pi * train_month / 12)
    train_cos = np.cos(2 * np.pi * train_month / 12)
    test_sin = np.sin(2 * np.pi * test_month / 12)
    test_cos = np.cos(2 * np.pi * test_month / 12)

    name = train_col.name
    return (
        pd.Series(train_sin, index=train_col.index, name=f"{name}_sin"),
        pd.Series(train_cos, index=train_col.index, name=f"{name}_cos"),
        pd.Series(test_sin, index=test_col.index, name=f"{name}_sin"),
        pd.Series(test_cos, index=test_col.index, name=f"{name}_cos"),
    )


def frequency_encode_mitigation(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Replace categories with their training-set frequency (0-1).

    Converts categorical to numerical, making it amenable to numerical strategies.
    """
    freq_map = train_col.value_counts(normalize=True).to_dict()
    tr_encoded = train_col.map(freq_map).fillna(0.0)
    te_encoded = test_col.map(freq_map).fillna(0.0)
    return (
        pd.Series(tr_encoded.values, index=train_col.index, name=train_col.name),
        pd.Series(te_encoded.values, index=test_col.index, name=test_col.name),
    )


def _clip_then_log(train_col, test_col):
    tr_c, te_c = clip_outliers(train_col, test_col)
    # Shift to ensure non-negative before log
    min_val = min(tr_c.min(), te_c.min())
    if pd.isna(min_val):
        return None
    if min_val < 0:
        shift = abs(min_val) + 1e-6
        tr_c = tr_c + shift
        te_c = te_c + shift
    return log_transform(tr_c, te_c)


def quantile_match(
    train_col: pd.Series, test_col: pd.Series, n_quantiles: int = 100
) -> tuple[pd.Series, pd.Series]:
    """Align test distribution to train via CDF-to-CDF quantile mapping.

    For each test value, finds its percentile rank in the test distribution,
    then maps to the corresponding value from the training distribution.
    Training data is unchanged; test rank ordering is preserved.
    """
    train_vals = pd.to_numeric(train_col, errors="coerce").dropna().values.astype(float)
    test_numeric = pd.to_numeric(test_col, errors="coerce")
    test_vals = test_numeric.dropna().values.astype(float)

    if len(train_vals) < 10 or len(test_vals) < 10:
        return train_col, test_col

    n_q = min(n_quantiles, len(train_vals) // 2)
    probs = np.linspace(0, 1, n_q + 1)
    train_q = np.quantile(train_vals, probs)

    # Vectorized: compute test CDF rank for all values
    sorted_test = np.sort(test_vals)
    all_test = test_numeric.values.astype(float)
    ranks = np.searchsorted(sorted_test, all_test, side='right').astype(float) / len(sorted_test)
    ranks = np.clip(ranks, 0, 1)

    # Map ranks to training quantiles
    mapped = np.interp(ranks, probs, train_q)

    # Preserve NaN
    nan_mask = test_numeric.isna()
    mapped[nan_mask.values] = np.nan

    result = pd.Series(mapped, index=test_col.index, name=test_col.name)
    return train_col, result


def rank_transform(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Convert values to their percentile rank (0-1) based on training distribution.

    Distribution-agnostic: eliminates drift while preserving ordering.
    """
    train_vals = pd.to_numeric(train_col, errors="coerce").dropna().values.astype(float)
    sorted_vals = np.sort(train_vals)
    n = len(sorted_vals)

    def _to_rank(s):
        arr = pd.to_numeric(s, errors="coerce").values.astype(float)
        ranks = np.searchsorted(sorted_vals, arr, side="right") / n
        return pd.Series(np.clip(ranks, 0, 1), index=s.index, name=s.name)

    return _to_rank(train_col.fillna(train_col.median())), _to_rank(test_col.fillna(train_col.median()))


# ── Progressive mitigation loop ──────────────────────────────────────────


def mitigate_feature_progressive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    feature_type: str,
    original_severity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Try graduated strategies on a single high-drift feature.

    Returns (train, test, log_entry) where log_entry records what happened.
    """
    from drift_detector import detect_numerical_drift, detect_categorical_drift

    log_entry = {
        "column": col,
        "feature_type": feature_type,
        "original_severity": original_severity,
        "strategy_applied": "Drop Feature",
        "final_severity": original_severity,
    }

    if feature_type == "temporal":
        # Try cyclical encoding
        train_sin, train_cos, test_sin, test_cos = encode_temporal_cyclical(
            train[col], test[col]
        )
        sin_drift = detect_numerical_drift(train_sin, test_sin)
        cos_drift = detect_numerical_drift(train_cos, test_cos)
        # Accept if BOTH sin and cos are below high
        if sin_drift["severity"] != "high" and cos_drift["severity"] != "high":
            train = train.drop(columns=[col])
            test = test.drop(columns=[col])
            train[train_sin.name] = train_sin.values
            train[train_cos.name] = train_cos.values
            test[test_sin.name] = test_sin.values
            test[test_cos.name] = test_cos.values
            worst = sin_drift["severity"] if sin_drift["severity"] == "moderate" else cos_drift["severity"]
            log_entry["strategy_applied"] = "Seasonality Matching"
            log_entry["final_severity"] = worst
            return train, test, log_entry
        # Cyclical failed -> drop
        return train, test, log_entry

    if feature_type == "numerical":
        strategies = [
            ("Feature Scaling", lambda tc, sc: clip_outliers(tc, sc)),
            ("Feature Scaling (Log)", lambda tc, sc: log_transform(tc, sc)),
            ("Feature Scaling (Clip+Log)", lambda tc, sc: _clip_then_log(tc, sc)),
            ("Discretization", lambda tc, sc: quantile_bin(tc, sc)),
            ("Rank Transform", lambda tc, sc: rank_transform(tc, sc)),
        ]
        # Evaluate strategies on subsample, apply winner to full data
        eval_tr, eval_te = _eval_subsample(train[col], test[col])
        for name, fn in strategies:
            result = fn(eval_tr, eval_te)
            if result is None:
                continue  # strategy not applicable (e.g. log with negatives)
            tr_col, te_col = result
            drift = detect_numerical_drift(tr_col, te_col)
            if drift["severity"] != "high":
                full_result = fn(train[col], test[col])
                if full_result is not None:
                    train[col] = full_result[0].values
                    test[col] = full_result[1].values
                log_entry["strategy_applied"] = name
                log_entry["final_severity"] = drift["severity"]
                return train, test, log_entry
        # All strategies failed -> drop
        return train, test, log_entry

    if feature_type == "categorical":
        # Evaluate strategies on subsample, apply winner to full data
        eval_tr, eval_te = _eval_subsample(train[col], test[col])

        # Strategy 1: Group rare categories
        tr_col, te_col = group_rare_categories(eval_tr, eval_te)
        drift = detect_categorical_drift(tr_col, te_col)
        if drift["severity"] != "high":
            full_tr, full_te = group_rare_categories(train[col], test[col])
            train[col] = full_tr.values
            test[col] = full_te.values
            log_entry["strategy_applied"] = "Category Consolidation"
            log_entry["final_severity"] = drift["severity"]
            return train, test, log_entry

        # Strategy 2: Frequency encode (converts to numerical)
        tr_freq, te_freq = frequency_encode_mitigation(eval_tr, eval_te)
        drift = detect_numerical_drift(tr_freq, te_freq)
        if drift["severity"] != "high":
            full_tr, full_te = frequency_encode_mitigation(train[col], test[col])
            train[col] = full_tr.values
            test[col] = full_te.values
            log_entry["strategy_applied"] = "Frequency Encoding"
            log_entry["final_severity"] = drift["severity"]
            return train, test, log_entry

        # Strategy 3: Frequency encode + numerical strategies
        num_strategies = [
            ("Freq+Clip", lambda tc, sc: clip_outliers(tc, sc)),
            ("Freq+Rank", lambda tc, sc: rank_transform(tc, sc)),
            ("Freq+Quantile", lambda tc, sc: quantile_bin(tc, sc)),
        ]
        for name, fn in num_strategies:
            result = fn(tr_freq, te_freq)
            if result is None:
                continue
            tr_num, te_num = result
            drift = detect_numerical_drift(tr_num, te_num)
            if drift["severity"] != "high":
                full_freq_tr, full_freq_te = frequency_encode_mitigation(train[col], test[col])
                full_result = fn(full_freq_tr, full_freq_te)
                if full_result is not None:
                    train[col] = full_result[0].values
                    test[col] = full_result[1].values
                log_entry["strategy_applied"] = name
                log_entry["final_severity"] = drift["severity"]
                return train, test, log_entry

        # All categorical strategies failed -> drop
        return train, test, log_entry

    return train, test, log_entry


# ── Main entry point ─────────────────────────────────────────────────────


def apply_all_mitigations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drift_df: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict], dict[str, list[str]], dict]:
    """Run all mitigation stages in order.

    Pipeline:
    1. Normalize data quality (city abbreviations, lowercase, hyphen removal)
    2. Re-run drift detection on normalized data
    3. Progressive mitigation: try graduated strategies before dropping
    4. Multivariate shift detection (diagnostic only)

    Returns (train_mitigated, test_mitigated, drop_cols, mitigation_log, norm_log, shift_info).
    """
    from drift_detector import _analyze_single_feature, _small_sample_factor, _apply_bh_correction

    # Stage 1: Data quality normalization
    train_m, test_m, norm_log = normalize_data_quality(train, test, feature_types)

    # Stage 2: Re-check drift only for columns modified by normalization
    # (avoids re-running all 500 features when only a few changed)
    drift_df_norm = drift_df.copy()
    if norm_log and not drift_df_norm.empty:
        modified_cols = set(norm_log.keys())
        recheck_count = 0
        for col in modified_cols:
            if col not in train_m.columns or col not in test_m.columns:
                continue
            ftype = None
            for ft in ("numerical", "categorical", "temporal"):
                if col in feature_types.get(ft, []):
                    ftype = ft
                    break
            if ftype is None:
                continue
            result = _analyze_single_feature(train_m[col], test_m[col], col, ftype)
            idx = drift_df_norm.index[drift_df_norm["column"] == col]
            if len(idx) > 0:
                for key, val in result.items():
                    if key in drift_df_norm.columns:
                        drift_df_norm.loc[idx[0], key] = val
                recheck_count += 1
        if recheck_count > 0:
            ss_factor = _small_sample_factor(len(train_m), len(test_m))
            drift_df_norm = _apply_bh_correction(drift_df_norm, ss_factor)
            drift_df_norm = drift_df_norm.sort_values(
                "severity", key=lambda s: s.map({"high": 0, "moderate": 1, "low": 2, "unknown": 3})
            )
        logger.info(f"Re-checked drift for {recheck_count} normalized columns (skipped {len(drift_df) - recheck_count} unchanged)")

    # Stage 3: Progressive mitigation for high-drift features
    # Collect all high-drift features (including temporal)
    high_drift_features = []
    if not drift_df_norm.empty:
        for _, row in drift_df_norm[drift_df_norm["severity"] == "high"].iterrows():
            high_drift_features.append((row["column"], row["feature_type"], row["severity"]))

    mitigation_log = []
    drop_cols = []

    if high_drift_features:
        logger.info(f"Progressive mitigation: processing {len(high_drift_features)} high-drift features")

    for i, (col, ftype, severity) in enumerate(high_drift_features):
        if col not in train_m.columns:
            continue
        if (i + 1) % 25 == 0:
            logger.info(f"  High-drift progress: {i + 1}/{len(high_drift_features)}")
        train_m, test_m, entry = mitigate_feature_progressive(
            train_m, test_m, col, ftype, severity
        )
        mitigation_log.append(entry)
        if entry["strategy_applied"] == "Drop Feature":
            drop_cols.append(col)
        else:
            logger.info(f"  {col}: {entry['strategy_applied']} ({severity} -> {entry['final_severity']})")

    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} features after all strategies exhausted: {drop_cols}")

    # Stage 3b: Progressive mitigation for moderate-drift features
    from drift_detector import detect_numerical_drift, detect_categorical_drift

    moderate_features = []
    if not drift_df_norm.empty:
        for _, row in drift_df_norm[drift_df_norm["severity"] == "moderate"].iterrows():
            col = row["column"]
            if col not in drop_cols and col in train_m.columns:
                moderate_features.append((col, row["feature_type"], row.get("psi", 0) or 0, row.get("jsd", 0) or 0))

    # Cap moderate features to top 50 by drift magnitude for large feature sets
    _MAX_MODERATE = 50
    if len(moderate_features) > _MAX_MODERATE:
        moderate_features.sort(key=lambda x: max(x[2], x[3]), reverse=True)
        logger.info(f"Moderate mitigation: processing top {_MAX_MODERATE}/{len(moderate_features)} by drift magnitude")
        moderate_features = moderate_features[:_MAX_MODERATE]

    if moderate_features:
        logger.info(f"Progressive mitigation: processing {len(moderate_features)} moderate-drift features")

    for mi, (col, ftype, orig_psi, orig_jsd) in enumerate(moderate_features):
        if (mi + 1) % 25 == 0:
            logger.info(f"  Moderate-drift progress: {mi + 1}/{len(moderate_features)}")
        if ftype == "numerical":
            strategies = [
                ("Gentle clipping", lambda tc, sc: clip_outliers(tc, sc)),
                ("Log transform", lambda tc, sc: log_transform(tc, sc)),
                ("Clip+Log", lambda tc, sc: _clip_then_log(tc, sc)),
                ("Quantile matching", lambda tc, sc: quantile_match(tc, sc)),
                ("Rank transform", lambda tc, sc: rank_transform(tc, sc)),
            ]
            eval_tr, eval_te = _eval_subsample(train_m[col], test_m[col])
            best_strategy = None
            best_psi = orig_psi
            best_fn = None
            best_sev = None
            for name, fn in strategies:
                result = fn(eval_tr, eval_te)
                if result is None:
                    continue
                tr_col, te_col = result
                drift = detect_numerical_drift(tr_col, te_col)
                new_psi = drift.get("psi", orig_psi) or orig_psi
                if drift["severity"] == "low":
                    # Immediate accept — apply to full data
                    full_result = fn(train_m[col], test_m[col])
                    if full_result is not None:
                        train_m[col] = full_result[0].values
                        test_m[col] = full_result[1].values
                    logger.info(f"  {col}: {name} (moderate -> low)")
                    mitigation_log.append({"column": col, "feature_type": ftype,
                                           "original_severity": "moderate", "strategy_applied": name,
                                           "final_severity": "low"})
                    best_strategy = None  # already applied
                    break
                elif new_psi < best_psi:
                    best_psi = new_psi
                    best_strategy = name
                    best_fn = fn
                    best_sev = drift["severity"]
            # Apply best improvement even if still moderate
            if best_strategy and best_fn:
                full_result = best_fn(train_m[col], test_m[col])
                if full_result is not None:
                    train_m[col] = full_result[0].values
                    test_m[col] = full_result[1].values
                logger.info(f"  {col}: {best_strategy} (moderate, PSI {orig_psi:.4f} -> {best_psi:.4f})")
                mitigation_log.append({"column": col, "feature_type": ftype,
                                       "original_severity": "moderate", "strategy_applied": best_strategy,
                                       "final_severity": best_sev})

        elif ftype == "categorical":
            eval_tr, eval_te = _eval_subsample(train_m[col], test_m[col])
            cat_strategies = [
                ("Rare category grouping", lambda tc, sc: group_rare_categories(tc, sc), "cat"),
                ("Frequency encoding", lambda tc, sc: frequency_encode_mitigation(tc, sc), "num"),
            ]
            best_strategy = None
            best_jsd = orig_jsd
            best_fn = None
            best_sev = None
            for name, fn, detect_type in cat_strategies:
                tr_col, te_col = fn(eval_tr, eval_te)
                if detect_type == "cat":
                    drift = detect_categorical_drift(tr_col, te_col)
                    new_jsd = drift.get("jsd", orig_jsd) or orig_jsd
                else:
                    drift = detect_numerical_drift(tr_col, te_col)
                    new_jsd = drift.get("psi", orig_jsd) or orig_jsd
                if drift["severity"] == "low":
                    full_tr, full_te = fn(train_m[col], test_m[col])
                    train_m[col] = full_tr.values
                    test_m[col] = full_te.values
                    logger.info(f"  {col}: {name} (moderate -> low)")
                    mitigation_log.append({"column": col, "feature_type": ftype,
                                           "original_severity": "moderate", "strategy_applied": name,
                                           "final_severity": "low"})
                    best_strategy = None
                    break
                elif new_jsd < best_jsd:
                    best_jsd = new_jsd
                    best_strategy = name
                    best_fn = fn
                    best_sev = drift["severity"]
            if best_strategy and best_fn:
                full_tr, full_te = best_fn(train_m[col], test_m[col])
                train_m[col] = full_tr.values
                test_m[col] = full_te.values
                logger.info(f"  {col}: {best_strategy} (moderate, JSD {orig_jsd:.4f} -> {best_jsd:.4f})")
                mitigation_log.append({"column": col, "feature_type": ftype,
                                       "original_severity": "moderate", "strategy_applied": best_strategy,
                                       "final_severity": best_sev})

    # Stage 4: Multivariate shift detection (diagnostic only, no weighting)
    from drift_detector import detect_multivariate_shift

    shift_info = detect_multivariate_shift(train_m, test_m, feature_types, extra_drop_cols=drop_cols)

    if shift_info["shift_detected"]:
        logger.info(
            f"Multivariate shift detected (AUC={shift_info['auc']:.4f}), "
            f"diagnostic only — no sample reweighting applied"
        )
    else:
        logger.info(
            f"No significant multivariate shift (AUC={shift_info['auc']:.4f}), "
            f"using uniform weights"
        )

    return train_m, test_m, drop_cols, mitigation_log, norm_log, shift_info
