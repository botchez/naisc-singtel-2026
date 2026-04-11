import numpy as np
import polars as pl
from polars_compat import to_numeric_coerce, is_numeric_col
from utils import logger

_MAX_EVAL_SAMPLE = 100_000


def _eval_subsample(train_col, test_col, max_n=_MAX_EVAL_SAMPLE, seed=42):
    """Subsample columns for fast strategy evaluation."""
    tr, te = train_col, test_col
    if len(tr) > max_n:
        tr = tr.sample(n=max_n, seed=seed)
    if len(te) > max_n:
        te = te.sample(n=max_n, seed=seed)
    return tr, te




def detect_abbreviation_mappings(
    train: pl.DataFrame, test: pl.DataFrame, col: str
) -> dict[str, str]:
    """Auto-detect city abbreviation-to-full-name mappings from the data.

    Candidate abbreviations: 2-3 uppercase alpha strings (e.g. "LA", "NYC").
    Candidate full names: multi-word values (contain a space).
    Matches abbreviation to full name whose word-initial letters match.
    Ties broken by combined frequency.
    """
    _ABB_SAMPLE = 100_000
    tr_vals = train[col].drop_nulls()
    te_vals = test[col].drop_nulls()
    if len(tr_vals) > _ABB_SAMPLE:
        tr_vals = tr_vals.sample(n=_ABB_SAMPLE, seed=42)
    if len(te_vals) > _ABB_SAMPLE:
        te_vals = te_vals.sample(n=_ABB_SAMPLE, seed=42)
    all_values = set(tr_vals.unique().to_list()) | set(te_vals.unique().to_list())

    abbreviations = [v for v in all_values if isinstance(v, str) and len(v) in (2, 3) and v.isupper() and v.isalpha()]
    full_names = [v for v in all_values if isinstance(v, str) and " " in v]

    if not abbreviations or not full_names:
        return {}

    # Combined frequency across both datasets (subsampled)
    tr_vc = tr_vals.value_counts()
    te_vc = te_vals.value_counts()
    # Build a combined frequency dict
    tr_freq = dict(zip(tr_vc.get_column(col).to_list(), tr_vc.get_column("count").to_list()))
    te_freq = dict(zip(te_vc.get_column(col).to_list(), te_vc.get_column("count").to_list()))
    combined_freq = {}
    for k in set(tr_freq) | set(te_freq):
        combined_freq[k] = tr_freq.get(k, 0) + te_freq.get(k, 0)

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
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_types: dict[str, list[str]],
):
    """Normalize data quality: city abbreviation mapping, lowercase, hyphen removal.

    Skips ID and target columns to preserve values like 'Yes'/'No'.
    Returns (train_clean, test_clean, norm_log) where norm_log maps column names
    to lists of normalization action labels applied.
    """
    train_clean = train.clone()
    test_clean = test.clone()

    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    norm_log: dict[str, list[str]] = {}

    # Auto-detect abbreviation mappings on categorical columns (NOT temporal)
    temporal_cols = set(feature_types.get("temporal", []))
    for col in feature_types.get("categorical", []):
        if col in skip_cols or col not in train_clean.columns or col not in test_clean.columns:
            continue
        if train_clean[col].dtype not in (pl.Utf8, pl.String):
            continue
        abbreviation_map = detect_abbreviation_mappings(train_clean, test_clean, col)
        if abbreviation_map:
            for old_val, new_val in abbreviation_map.items():
                train_clean = train_clean.with_columns(
                    pl.when(pl.col(col) == old_val).then(pl.lit(new_val)).otherwise(pl.col(col)).alias(col)
                )
                test_clean = test_clean.with_columns(
                    pl.when(pl.col(col) == old_val).then(pl.lit(new_val)).otherwise(pl.col(col)).alias(col)
                )
            logger.info(f"Auto-detected abbreviation mappings in '{col}': {abbreviation_map}")
            norm_log.setdefault(col, []).append("Abbreviations standardized")

    # Conditional lowercase + hyphen removal: only apply if it reduces drift
    from drift_detector import detect_categorical_drift
    normalized = []
    skipped_norm = []
    cat_cols = set(feature_types.get("categorical", []))  # exclude temporal from normalization/fuzzy

    # Pre-compute subsample indices for drift comparison on large datasets
    _NORM_SAMPLE = 100_000
    rng = np.random.RandomState(42)
    _tr_norm_idx = rng.choice(
        len(train_clean), min(_NORM_SAMPLE, len(train_clean)), replace=False
    ) if len(train_clean) > _NORM_SAMPLE else None
    _te_norm_idx = rng.choice(
        len(test_clean), min(_NORM_SAMPLE, len(test_clean)), replace=False
    ) if len(test_clean) > _NORM_SAMPLE else None

    for col in train_clean.columns:
        if col in skip_cols or col not in cat_cols:
            continue
        if train_clean[col].dtype in (pl.Utf8, pl.String):
            # Subsample for drift comparison
            tr_sample = train_clean[col].gather(_tr_norm_idx.tolist()) if _tr_norm_idx is not None else train_clean[col]
            te_sample = test_clean[col].gather(_te_norm_idx.tolist()) if _te_norm_idx is not None else test_clean[col]

            # Compute drift before normalization (on subsample)
            before = detect_categorical_drift(tr_sample, te_sample)
            before_jsd = before.get("jsd", 0) or 0

            # Trial normalization on subsample for comparison
            tr_trial_s = tr_sample.cast(pl.Utf8).str.to_lowercase().str.replace_all("-", " ", literal=True)
            te_trial_s = te_sample.cast(pl.Utf8).str.to_lowercase().str.replace_all("-", " ", literal=True)
            after = detect_categorical_drift(tr_trial_s, te_trial_s)
            after_jsd = after.get("jsd", 0) or 0

            if after_jsd <= before_jsd:
                # Apply to full data
                train_clean = train_clean.with_columns(
                    pl.col(col).cast(pl.Utf8).str.to_lowercase().str.replace_all("-", " ", literal=True).alias(col)
                )
                test_clean = test_clean.with_columns(
                    pl.col(col).cast(pl.Utf8).str.to_lowercase().str.replace_all("-", " ", literal=True).alias(col)
                )
                normalized.append(col)
                norm_log.setdefault(col, []).extend(["Lowercased", "Hyphens removed"])
            else:
                skipped_norm.append(col)

    if normalized:
        logger.info(f"Normalized {len(normalized)} columns: {normalized}")
    if skipped_norm:
        logger.info(f"Skipped normalization for {len(skipped_norm)} columns (would increase drift): {skipped_norm}")

    # Fuzzy match: map unseen test categories to closest train category
    # Skip for high-cardinality columns -- fuzzy matching is O(unseen x train_cats)
    # and only useful for small categoricals (city names, state codes, etc.)
    import difflib
    _MAX_FUZZY_TRAIN_CATS = 200
    _MAX_FUZZY_UNSEEN = 50
    fuzzy_skipped = 0
    for col in cat_cols:
        if col in skip_cols or col not in train_clean.columns or col not in test_clean.columns:
            continue
        if train_clean[col].dtype not in (pl.Utf8, pl.String):
            continue
        tr_s = train_clean[col].gather(_tr_norm_idx.tolist()).drop_nulls() if _tr_norm_idx is not None else train_clean[col].drop_nulls()
        te_s = test_clean[col].gather(_te_norm_idx.tolist()).drop_nulls() if _te_norm_idx is not None else test_clean[col].drop_nulls()
        train_cats = set(tr_s.unique().to_list())
        test_cats = set(te_s.unique().to_list())
        unseen = test_cats - train_cats
        if not unseen:
            continue
        # Skip if too many categories -- fuzzy matching is only for low-cardinality typo fixes
        if len(train_cats) > _MAX_FUZZY_TRAIN_CATS or len(unseen) > _MAX_FUZZY_UNSEEN:
            fuzzy_skipped += 1
            continue
        fuzzy_map = {}
        train_cats_list = list(train_cats)
        for cat in unseen:
            # Try strict cutoff first, then relaxed
            matches = difflib.get_close_matches(cat, train_cats_list, n=1, cutoff=0.6)
            if not matches:
                matches = difflib.get_close_matches(cat, train_cats_list, n=1, cutoff=0.45)
            if matches:
                fuzzy_map[cat] = matches[0]
        if fuzzy_map:
            # Apply fuzzy map via chained when/then
            expr = pl.col(col)
            for old_val, new_val in fuzzy_map.items():
                expr = pl.when(pl.col(col) == old_val).then(pl.lit(new_val)).otherwise(expr)
            test_clean = test_clean.with_columns(expr.alias(col))
            logger.info(f"Fuzzy matched '{col}': {fuzzy_map}")
            norm_log.setdefault(col, []).append(f"Fuzzy matched: {fuzzy_map}")
    if fuzzy_skipped:
        logger.info(f"Skipped fuzzy matching for {fuzzy_skipped} high-cardinality columns")

    return train_clean, test_clean, norm_log


# ── Strategy functions ────────────────────────────────────────────────────


def clip_outliers(
    train_col, test_col
):
    """Clip both columns to [1st, 99th] percentile of train."""
    vals = to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float)
    lo, hi = np.percentile(vals, [1, 99])
    return train_col.clip(lo, hi), test_col.clip(lo, hi)


def log_transform(
    train_col, test_col
):
    """Apply log1p if all values >= 0; return None to skip otherwise."""
    tr_clean = to_numeric_coerce(train_col).drop_nulls()
    te_clean = to_numeric_coerce(test_col).drop_nulls()
    if (tr_clean < 0).any() or (te_clean < 0).any():
        return None
    # Use numpy for log1p, then wrap back as polars Series
    tr_numeric = to_numeric_coerce(train_col)
    te_numeric = to_numeric_coerce(test_col)
    tr_arr = tr_numeric.to_numpy()
    te_arr = te_numeric.to_numpy()
    return (
        pl.Series(name=train_col.name, values=np.log1p(tr_arr)),
        pl.Series(name=test_col.name, values=np.log1p(te_arr)),
    )


def quantile_bin(
    train_col, test_col, n_bins: int = 10
):
    """Discretize into quantile bins defined by train edges. Nulls stay null."""
    vals = to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float)
    # Adapt bins to sample size
    n_bins = min(n_bins, max(2, len(vals) // 20))
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, percentiles)
    edges = np.unique(edges)
    tr_arr = to_numeric_coerce(train_col).to_numpy().astype(float)
    te_arr = to_numeric_coerce(test_col).to_numpy().astype(float)
    # np.digitize silently returns 0 for NaN (merging nulls with the smallest
    # bin). Mask nulls, digitize the rest, then write NaN back.
    tr_nan = np.isnan(tr_arr)
    te_nan = np.isnan(te_arr)
    tr_bins = np.digitize(np.where(tr_nan, 0.0, tr_arr), edges).astype(np.float64)
    te_bins = np.digitize(np.where(te_nan, 0.0, te_arr), edges).astype(np.float64)
    tr_bins[tr_nan] = np.nan
    te_bins[te_nan] = np.nan
    return (
        pl.Series(name=train_col.name, values=tr_bins),
        pl.Series(name=test_col.name, values=te_bins),
    )


def group_rare_categories(
    train_col, test_col, min_freq: float = 0.01
):
    """Merge categories below min_freq in train + unseen-in-train into '_other'."""

    # Compute frequency of each category in train
    vc = train_col.drop_nulls().value_counts()
    total = len(train_col.drop_nulls())
    if total == 0:
        return train_col, test_col
    # vc columns: [train_col.name, "count"]
    col_name = train_col.name
    freq_df = vc.with_columns((pl.col("count") / total).alias("freq"))
    rare_cats_df = freq_df.filter(pl.col("freq") < min_freq)
    rare_cats = set(rare_cats_df[col_name].to_list())
    train_cats = set(train_col.drop_nulls().unique().to_list())

    def _remap(s: pl.Series) -> pl.Series:
        # Null stays null; rare or unseen -> "_other"; rest stays
        rare_list = list(rare_cats)
        train_list = list(train_cats)
        result = s.to_frame().select(
            pl.when(pl.col(s.name).is_null()).then(pl.lit(None))
            .when(pl.col(s.name).is_in(rare_list) | ~pl.col(s.name).is_in(train_list))
            .then(pl.lit("_other"))
            .otherwise(pl.col(s.name))
            .alias(s.name)
        )[s.name]
        return result

    return _remap(train_col), _remap(test_col)


def frequency_encode_mitigation(
    train_col, test_col
):
    """Replace categories with their training-set frequency (0-1).

    Converts categorical to numerical, making it amenable to numerical strategies.
    """

    # Build frequency map from train
    vc = train_col.drop_nulls().value_counts()
    total = len(train_col.drop_nulls())
    if total == 0:
        return (
            pl.Series(name=train_col.name, values=np.zeros(len(train_col))),
            pl.Series(name=test_col.name, values=np.zeros(len(test_col))),
        )
    col_name = train_col.name
    freq_map = dict(zip(
        vc[col_name].to_list(),
        (vc["count"].cast(pl.Float64) / total).to_list()
    ))

    # Map values; unseen categories get 0.0 (vectorized via polars)
    tr_series = train_col.replace_strict(freq_map, default=0.0, return_dtype=pl.Float64).fill_null(0.0)
    te_series = test_col.replace_strict(freq_map, default=0.0, return_dtype=pl.Float64).fill_null(0.0)

    return (
        tr_series.alias(train_col.name),
        te_series.alias(test_col.name),
    )


def robust_scale(
    train_col, test_col
):
    """Scale using median and IQR from train. Distribution-robust alternative to z-score."""
    train_vals = to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float)
    median = np.median(train_vals)
    q1, q3 = np.percentile(train_vals, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-10:
        return train_col, test_col
    tr_arr = (to_numeric_coerce(train_col).to_numpy() - median) / iqr
    te_arr = (to_numeric_coerce(test_col).to_numpy() - median) / iqr
    return (
        pl.Series(name=train_col.name, values=tr_arr),
        pl.Series(name=test_col.name, values=te_arr),
    )


def _clip_then_log(train_col, test_col):
    tr_c, te_c = clip_outliers(train_col, test_col)
    # Shift to ensure non-negative before log
    tr_min = tr_c.min()
    te_min = te_c.min()
    min_val = min(
        tr_min if tr_min is not None else float('inf'),
        te_min if te_min is not None else float('inf')
    )
    if min_val is None or min_val != min_val:  # NaN check
        return None
    if min_val == float('inf'):
        return None
    if min_val < 0:
        shift = abs(min_val) + 1e-6
        tr_c = tr_c + shift
        te_c = te_c + shift
    return log_transform(tr_c, te_c)


def quantile_match(
    train_col, test_col, n_quantiles: int = 100
):
    """Align test distribution to train via CDF-to-CDF quantile mapping.

    For each test value, finds its percentile rank in the test distribution,
    then maps to the corresponding value from the training distribution.
    Training data is unchanged; test rank ordering is preserved.
    """
    train_vals = to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float)
    test_numeric = to_numeric_coerce(test_col)
    test_vals = test_numeric.drop_nulls().to_numpy().astype(float)

    if len(train_vals) < 10 or len(test_vals) < 10:
        return train_col, test_col

    n_q = min(n_quantiles, len(train_vals) // 2)
    probs = np.linspace(0, 1, n_q + 1)
    train_q = np.quantile(train_vals, probs)

    # Vectorized: compute test CDF rank for all values
    sorted_test = np.sort(test_vals)
    all_test = test_numeric.to_numpy().astype(float)
    ranks = np.searchsorted(sorted_test, all_test, side='right').astype(float) / len(sorted_test)
    ranks = np.clip(ranks, 0, 1)

    # Map ranks to training quantiles
    mapped = np.interp(ranks, probs, train_q)

    # Preserve NaN
    nan_mask = test_numeric.is_null().to_numpy()
    mapped[nan_mask] = np.nan

    result = pl.Series(name=test_col.name, values=mapped)
    return train_col, result


def rank_transform(
    train_col, test_col
):
    """Convert values to their percentile rank (0-1) based on training distribution.

    Distribution-agnostic: eliminates drift while preserving ordering.
    """
    train_vals = to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float)
    sorted_vals = np.sort(train_vals)
    n = len(sorted_vals)

    # Fill nulls with median for rank computation
    train_median = float(np.median(train_vals)) if len(train_vals) > 0 else 0.0

    def _to_rank(s: pl.Series) -> pl.Series:
        arr = to_numeric_coerce(s.fill_null(train_median)).to_numpy().astype(float)
        ranks = np.searchsorted(sorted_vals, arr, side="right") / n
        return pl.Series(name=s.name, values=np.clip(ranks, 0, 1))

    return _to_rank(train_col), _to_rank(test_col)


# ── Progressive mitigation loop ──────────────────────────────────────────


def mitigate_feature_progressive(
    train: pl.DataFrame,
    test: pl.DataFrame,
    col: str,
    feature_type: str,
    original_severity: str,
):
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

    if feature_type == "numerical":
        strategies = [
            ("Feature Scaling", lambda tc, sc: clip_outliers(tc, sc)),
            ("Feature Scaling (Log)", lambda tc, sc: log_transform(tc, sc)),
            ("Feature Scaling (Clip+Log)", lambda tc, sc: _clip_then_log(tc, sc)),
            ("Robust Scaling", lambda tc, sc: robust_scale(tc, sc)),
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
                    train = train.with_columns(full_result[0].alias(col))
                    test = test.with_columns(full_result[1].alias(col))
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
            train = train.with_columns(full_tr.alias(col))
            test = test.with_columns(full_te.alias(col))
            log_entry["strategy_applied"] = "Category Consolidation"
            log_entry["final_severity"] = drift["severity"]
            return train, test, log_entry

        # Strategy 2: Frequency encode (converts to numerical)
        tr_freq, te_freq = frequency_encode_mitigation(eval_tr, eval_te)
        drift = detect_numerical_drift(tr_freq, te_freq)
        if drift["severity"] != "high":
            full_tr, full_te = frequency_encode_mitigation(train[col], test[col])
            train = train.with_columns(full_tr.alias(col))
            test = test.with_columns(full_te.alias(col))
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
                    train = train.with_columns(full_result[0].alias(col))
                    test = test.with_columns(full_result[1].alias(col))
                log_entry["strategy_applied"] = name
                log_entry["final_severity"] = drift["severity"]
                return train, test, log_entry

        # All categorical strategies failed -> drop
        return train, test, log_entry

    return train, test, log_entry


# ── Main entry point ─────────────────────────────────────────────────────


def apply_all_mitigations(
    train: pl.DataFrame,
    test: pl.DataFrame,
    drift_df: pl.DataFrame,
    feature_types: dict[str, list[str]],
):
    """Run all mitigation stages in order.

    Pipeline:
    1. Normalize data quality (city abbreviations, lowercase, hyphen removal)
    2. Re-run drift detection on normalized data
    3. Progressive mitigation: try graduated strategies before dropping

    Returns (train_mitigated, test_mitigated, drop_cols, mitigation_log, norm_log).
    """
    from drift_detector import _analyze_single_feature, _small_sample_factor, _apply_bh_correction

    # Stage 1: Data quality normalization
    train_m, test_m, norm_log = normalize_data_quality(train, test, feature_types)

    # Stage 2: Re-check drift only for columns modified by normalization
    # (avoids re-running all 500 features when only a few changed)
    drift_df_norm = drift_df.clone()
    if norm_log and len(drift_df_norm) > 0:
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
            # Find the row index for this column
            col_mask = drift_df_norm["column"] == col
            match_indices = [i for i, v in enumerate(col_mask.to_list()) if v]
            if match_indices:
                idx = match_indices[0]
                # Update each field from the result
                for key, val in result.items():
                    if key in drift_df_norm.columns:
                        # Build a new column with the updated value at idx
                        old_col = drift_df_norm[key]
                        values = old_col.to_list()
                        values[idx] = val
                        drift_df_norm = drift_df_norm.with_columns(
                            pl.Series(name=key, values=values, dtype=old_col.dtype if not isinstance(val, str) else pl.Utf8)
                        )
                recheck_count += 1
        if recheck_count > 0:
            ss_factor = _small_sample_factor(len(train_m), len(test_m))
            drift_df_norm = _apply_bh_correction(drift_df_norm, ss_factor)
            # Sort by severity
            severity_order = {"high": 0, "moderate": 1, "low": 2, "unknown": 3}
            drift_df_norm = drift_df_norm.with_columns(
                pl.col("severity").replace_strict(severity_order, default=3).alias("_sev_order")
            ).sort("_sev_order").drop("_sev_order")
        logger.info(f"Re-checked drift for {recheck_count} normalized columns (skipped {len(drift_df) - recheck_count} unchanged)")

    # Stage 3: Progressive mitigation for high-drift features
    # Collect all high-drift features (including temporal)
    high_drift_features = []
    if len(drift_df_norm) > 0:
        high_mask = drift_df_norm["severity"] == "high"
        high_rows = drift_df_norm.filter(high_mask)
        for i in range(len(high_rows)):
            high_drift_features.append((
                high_rows["column"][i],
                high_rows["feature_type"][i],
                high_rows["severity"][i],
            ))

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
    if len(drift_df_norm) > 0:
        mod_mask = drift_df_norm["severity"] == "moderate"
        mod_rows = drift_df_norm.filter(mod_mask)
        for i in range(len(mod_rows)):
            col = mod_rows["column"][i]
            if col not in drop_cols and col in train_m.columns:
                psi_val = mod_rows["psi"][i] if "psi" in mod_rows.columns else 0
                jsd_val = mod_rows["jsd"][i] if "jsd" in mod_rows.columns else 0
                moderate_features.append((
                    col,
                    mod_rows["feature_type"][i],
                    psi_val if psi_val is not None else 0,
                    jsd_val if jsd_val is not None else 0,
                ))

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
                ("Robust scaling", lambda tc, sc: robust_scale(tc, sc)),
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
                    # Immediate accept -- apply to full data
                    full_result = fn(train_m[col], test_m[col])
                    if full_result is not None:
                        train_m = train_m.with_columns(full_result[0].alias(col))
                        test_m = test_m.with_columns(full_result[1].alias(col))
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
                    train_m = train_m.with_columns(full_result[0].alias(col))
                    test_m = test_m.with_columns(full_result[1].alias(col))
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
                    train_m = train_m.with_columns(full_tr.alias(col))
                    test_m = test_m.with_columns(full_te.alias(col))
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
                train_m = train_m.with_columns(full_tr.alias(col))
                test_m = test_m.with_columns(full_te.alias(col))
                logger.info(f"  {col}: {best_strategy} (moderate, JSD {orig_jsd:.4f} -> {best_jsd:.4f})")
                mitigation_log.append({"column": col, "feature_type": ftype,
                                       "original_severity": "moderate", "strategy_applied": best_strategy,
                                       "final_severity": best_sev})

    return train_m, test_m, drop_cols, mitigation_log, norm_log
