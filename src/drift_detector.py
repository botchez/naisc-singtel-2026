import polars as pl
import numpy as np
from scipy import stats
from polars_compat import to_numeric_coerce

# Maximum sample size for statistical tests. Beyond this, additional rows
# add negligible statistical power but cost O(n log n) compute.  100K is
# enough for KS, PSI, Chi², and Wasserstein at any practical significance level.
_MAX_DRIFT_SAMPLE = 100_000


def _subsample_array(arr: np.ndarray, max_n: int = _MAX_DRIFT_SAMPLE, seed: int = 42) -> np.ndarray:
    """Randomly subsample an array if it exceeds max_n."""
    if len(arr) <= max_n:
        return arr
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


def _subsample_series(s: pl.Series, max_n: int = _MAX_DRIFT_SAMPLE, seed: int = 42) -> pl.Series:
    """Randomly subsample a Series if it exceeds max_n."""
    if len(s) <= max_n:
        return s
    return s.sample(n=max_n, seed=seed)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    # Adapt bins to sample size to avoid sparse bins
    n_valid = len(expected[~np.isnan(expected)])
    bins = min(bins, max(2, n_valid // 20))
    # Use shared bin edges from expected
    breakpoints = np.linspace(0, 100, bins + 1)
    edges = np.percentile(expected[~np.isnan(expected)], breakpoints)
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0

    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)

    # Avoid division by zero
    expected_pct = (expected_counts + 1e-6) / (expected_counts.sum() + 1e-6 * len(expected_counts))
    actual_pct = (actual_counts + 1e-6) / (actual_counts.sum() + 1e-6 * len(actual_counts))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_cramers_v(chi2_stat: float, n: int, k: int) -> float:
    """Cramer's V: sample-size-normalized chi-squared effect size.

    V ranges from 0 (identical distributions) to 1 (complete separation).
    Unlike chi2 p-value, V is independent of sample size.
    """
    if n == 0 or k <= 1:
        return 0.0
    return float(np.sqrt(chi2_stat / (n * (k - 1))))


def compute_jsd(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """Jensen-Shannon divergence between two categorical distributions."""
    p = (p_counts + 1e-10) / (p_counts.sum() + 1e-10 * len(p_counts))
    q = (q_counts + 1e-10) / (q_counts.sum() + 1e-10 * len(q_counts))
    m = 0.5 * (p + q)
    jsd = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
    return float(jsd)


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for multiple testing."""
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    raw = pvals[order]
    adjusted = raw * n / np.arange(1, n + 1)
    # Enforce monotonicity (cumulative minimum from the end)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(n)
    result[order] = adjusted
    return result


def _missingness_severity(train_null_pct: float, test_null_pct: float) -> str:
    """Classify severity of missingness drift between train and test."""
    null_diff = abs(test_null_pct - train_null_pct)
    null_ratio = test_null_pct / max(train_null_pct, 0.001)
    if null_diff > 0.20 or (null_ratio > 5 and test_null_pct > 0.05):
        return "high"
    if null_diff > 0.05 or (null_ratio > 2 and test_null_pct > 0.02):
        return "moderate"
    return "low"


_SEVERITY_RANK = {"high": 2, "moderate": 1, "low": 0, "unknown": -1}


def _escalate_severity(base: str, missingness: str) -> str:
    """Return the worse of two severity levels."""
    if _SEVERITY_RANK.get(missingness, -1) > _SEVERITY_RANK.get(base, -1):
        return missingness
    return base


def detect_numerical_drift(
    train_col: pl.Series, test_col: pl.Series
) -> dict:
    """Run drift tests on a single numerical feature."""
    train_null_pct = float(train_col.null_count() / len(train_col))
    test_null_pct = float(test_col.null_count() / len(test_col))
    miss_sev = _missingness_severity(train_null_pct, test_null_pct)

    train_vals = to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float)
    test_vals = to_numeric_coerce(test_col).drop_nulls().to_numpy().astype(float)

    if len(train_vals) == 0 or len(test_vals) == 0:
        return {"ks_stat": None, "ks_pval": None, "psi": None, "wasserstein": None,
                "wasserstein_norm": None, "train_null_pct": round(train_null_pct, 4),
                "test_null_pct": round(test_null_pct, 4), "severity": _escalate_severity("unknown", miss_sev)}

    # Subsample for statistical tests -- 50K rows gives equivalent power
    train_sample = _subsample_array(train_vals)
    test_sample = _subsample_array(test_vals)

    ks_stat, ks_pval = stats.ks_2samp(train_sample, test_sample)
    psi = compute_psi(train_sample, test_sample)
    wasserstein = stats.wasserstein_distance(train_sample, test_sample)

    # Normalized Wasserstein (scale-invariant, similar to Cohen's d)
    std_train = np.std(train_sample)
    wasserstein_norm = wasserstein / (std_train + 1e-8)

    # IQR shift detection: spread changes not captured by location-based tests
    train_q1, train_q3 = np.percentile(train_sample, [25, 75])
    test_q1, test_q3 = np.percentile(test_sample, [25, 75])
    train_iqr = train_q3 - train_q1
    test_iqr = test_q3 - test_q1
    iqr_ratio = test_iqr / (train_iqr + 1e-8)

    # Tail divergence: extreme percentile shifts not captured by histogram-based PSI
    train_p1, train_p99 = np.percentile(train_sample, [1, 99])
    test_p1, test_p99 = np.percentile(test_sample, [1, 99])
    tail_low = abs(test_p1 - train_p1) / (std_train + 1e-8)
    tail_high = abs(test_p99 - train_p99) / (std_train + 1e-8)
    tail_div = max(tail_low, tail_high)

    # Use original full sizes for small-sample factor (not subsampled sizes)
    ss_factor = _small_sample_factor(len(train_vals), len(test_vals))
    severity = _numerical_severity(ks_pval, psi, wasserstein_norm, ks_stat, iqr_ratio, tail_div, ss_factor)
    severity = _escalate_severity(severity, miss_sev)

    return {
        "ks_stat": round(ks_stat, 4),
        "ks_pval": round(ks_pval, 6),
        "psi": round(psi, 4),
        "wasserstein": round(wasserstein, 4),
        "wasserstein_norm": round(wasserstein_norm, 4),
        "iqr_ratio": round(iqr_ratio, 4),
        "tail_divergence": round(tail_div, 4),
        "train_null_pct": round(train_null_pct, 4),
        "test_null_pct": round(test_null_pct, 4),
        "severity": severity,
    }


def detect_categorical_drift(
    train_col: pl.Series, test_col: pl.Series
) -> dict:
    """Run drift tests on a single categorical feature."""
    train_null_pct = float(train_col.null_count() / len(train_col))
    test_null_pct = float(test_col.null_count() / len(test_col))
    miss_sev = _missingness_severity(train_null_pct, test_null_pct)

    # Subsample for chi-squared / JSD -- preserves distribution proportions
    train_sub = _subsample_series(train_col.drop_nulls())
    test_sub = _subsample_series(test_col.drop_nulls())

    all_categories = sorted(set(train_sub.unique().to_list()) | set(test_sub.unique().to_list()), key=str)

    if len(all_categories) == 0:
        return {"chi2_stat": None, "chi2_pval": None, "jsd": None, "unseen_in_test": [],
                "train_null_pct": round(train_null_pct, 4), "test_null_pct": round(test_null_pct, 4),
                "severity": _escalate_severity("unknown", miss_sev)}

    # Build aligned count arrays via polars join
    vc_train = train_sub.value_counts()
    all_cats_df = pl.DataFrame({train_sub.name: all_categories})
    merged_train = all_cats_df.join(vc_train, on=train_sub.name, how="left").fill_null(0)
    train_counts = merged_train["count"].to_numpy().astype(float)

    vc_test = test_sub.value_counts()
    all_cats_df_test = pl.DataFrame({test_sub.name: all_categories})
    merged_test = all_cats_df_test.join(vc_test, on=test_sub.name, how="left").fill_null(0)
    test_counts = merged_test["count"].to_numpy().astype(float)

    # JSD on full distribution (always reliable)
    jsd = compute_jsd(train_counts, test_counts)

    # Chi-squared: for high-cardinality columns, group rare categories into _other
    # to avoid sparse contingency tables inflating chi2
    n_categories = len(all_categories)
    if n_categories > 50:
        # Keep top-K by combined frequency, merge rest into _other
        combined = train_counts + test_counts
        top_k = 30
        top_idx = np.argsort(combined)[-top_k:]
        other_train = train_counts.sum() - train_counts[top_idx].sum()
        other_test = test_counts.sum() - test_counts[top_idx].sum()
        chi2_train = np.append(train_counts[top_idx], other_train)
        chi2_test = np.append(test_counts[top_idx], other_test)
        contingency = np.array([chi2_train, chi2_test])
    else:
        contingency = np.array([train_counts, test_counts])

    # Only run chi2 if we have sufficient data
    if contingency.sum() > 0 and np.all(contingency.sum(axis=0) > 0):
        chi2_stat, chi2_pval, _, _ = stats.chi2_contingency(contingency)
    else:
        chi2_stat, chi2_pval = 0.0, 1.0

    # Cramer's V: sample-size-normalized effect size
    n_total = int(contingency.sum())
    k_cats = min(contingency.shape[0], contingency.shape[1])
    cramers_v = compute_cramers_v(chi2_stat, n_total, k_cats)

    # Unseen categories in test
    train_cats = set(train_sub.unique().to_list())
    test_cats = set(test_sub.unique().to_list())
    unseen_in_test = sorted(test_cats - train_cats, key=str)

    # Use original full sizes for small-sample factor
    n_tr = len(train_col.drop_nulls())
    n_te = len(test_col.drop_nulls())
    ss_factor = _small_sample_factor(n_tr, n_te)
    severity = _categorical_severity(chi2_pval, jsd, cramers_v, ss_factor)
    severity = _escalate_severity(severity, miss_sev)

    return {
        "chi2_stat": round(chi2_stat, 4),
        "chi2_pval": round(chi2_pval, 6),
        "jsd": round(jsd, 4),
        "cramers_v": round(cramers_v, 4),
        "unseen_in_test": unseen_in_test,
        "train_null_pct": round(train_null_pct, 4),
        "test_null_pct": round(test_null_pct, 4),
        "severity": severity,
    }


def _small_sample_factor(n_train: int, n_test: int, ref_size: int = 5000) -> float:
    """Compute threshold scaling factor for small-sample robustness.

    For large samples (harmonic mean >= ref_size), returns 1.0 (no change).
    For small samples, returns >1.0 to widen PSI/JSD severity thresholds,
    reducing false-positive drift detections caused by noisy histogram estimates.
    """
    n_eff = 2 * n_train * n_test / (n_train + n_test + 1e-8)
    if n_eff >= ref_size:
        return 1.0
    return (ref_size / n_eff) ** 0.5


def _numerical_severity(ks_pval: float, psi: float, wasserstein_norm: float = 0.0,
                        ks_stat: float = 0.0, iqr_ratio: float = 1.0,
                        tail_div: float = 0.0, ss_factor: float = 1.0) -> str:
    """Classify drift severity for numerical features.

    Uses PSI, KS p-value, KS D-statistic (effect size), normalized Wasserstein
    distance, IQR ratio (test_iqr / train_iqr), and tail divergence (max shift
    in 1st/99th percentiles normalized by training std).

    ss_factor: small-sample scaling factor for PSI thresholds (>1.0 widens bands).
    """
    if psi >= 0.25 * ss_factor:
        return "high"
    if psi >= 0.15 * ss_factor and ks_pval < 0.001:
        return "high"
    # Large distributional shift confirmed by both Wasserstein and PSI
    if wasserstein_norm >= 1.0 and psi >= 0.1 * ss_factor:
        return "high"
    # Large KS effect size combined with moderate PSI
    if ks_stat >= 0.15 and psi >= 0.1 * ss_factor:
        return "high"
    # IQR spread change > 2x (or < 0.5x) combined with moderate PSI
    if (iqr_ratio > 2.0 or iqr_ratio < 0.5) and psi >= 0.1 * ss_factor:
        return "high"
    # Tail divergence: extreme percentile shift (>2 std) with any PSI signal
    if tail_div > 2.0 and psi >= 0.05 * ss_factor:
        return "high"
    if psi >= 0.1 * ss_factor or (ks_pval < 0.05 and ks_stat >= 0.05) or (wasserstein_norm >= 0.3 and psi >= 0.05 * ss_factor):
        return "moderate"
    if tail_div > 1.0 and psi >= 0.03 * ss_factor:
        return "moderate"
    return "low"


def _categorical_severity(chi2_pval: float, jsd: float, cramers_v: float = 0.0,
                          ss_factor: float = 1.0) -> str:
    """Classify drift severity for categorical features.

    Uses JSD, Chi2 p-value, and Cramer's V (sample-size-normalized effect size).
    ss_factor: small-sample scaling factor for JSD thresholds (>1.0 widens bands).
    """
    if jsd >= 0.1 * ss_factor:
        return "high"
    if cramers_v >= 0.3 and jsd >= 0.05 * ss_factor:
        return "high"
    if jsd >= 0.05 * ss_factor or chi2_pval < 0.001 or cramers_v >= 0.15:
        return "moderate"
    if chi2_pval < 0.05:
        return "moderate"
    return "low"


def generate_drift_description(
    train_col: pl.Series, test_col: pl.Series, feature_type: str, drift_metrics: dict
) -> str:
    """Generate a qualitative natural-language description of the drift observed."""
    parts = []

    if feature_type == "numerical":
        train_vals = _subsample_array(to_numeric_coerce(train_col).drop_nulls().to_numpy().astype(float))
        test_vals = _subsample_array(to_numeric_coerce(test_col).drop_nulls().to_numpy().astype(float))
        if len(train_vals) == 0 or len(test_vals) == 0:
            return "Insufficient data for drift characterization"

        # Skewness comparison
        train_skew = stats.skew(train_vals)
        test_skew = stats.skew(test_vals)
        skew_diff = test_skew - train_skew
        if abs(skew_diff) > 0.3:
            direction = "left" if skew_diff < 0 else "right"
            parts.append(f"greater {direction}-skewness in test set compared to training set")

        # Mean shift
        train_mean, test_mean = np.mean(train_vals), np.mean(test_vals)
        train_std = np.std(train_vals)
        if train_std > 0:
            mean_shift = (test_mean - train_mean) / train_std
            if abs(mean_shift) > 0.15:
                direction = "higher" if mean_shift > 0 else "lower"
                parts.append(f"mean shifts {direction} in test set")

        # Range expansion / contraction
        train_range = np.ptp(train_vals)
        test_range = np.ptp(test_vals)
        if train_range > 0:
            range_ratio = test_range / train_range
            if range_ratio > 1.3:
                parts.append("feature ranges explode in test set")
            elif range_ratio < 0.7:
                parts.append("feature range contracts in test set")

        # Variance change
        test_std = np.std(test_vals)
        if train_std > 0:
            var_ratio = test_std / train_std
            if var_ratio > 1.3 and "range" not in " ".join(parts):
                parts.append("variance increases in test set")
            elif var_ratio < 0.7:
                parts.append("variance decreases in test set")

    elif feature_type in ("categorical", "temporal"):
        unseen = drift_metrics.get("unseen_in_test", "")
        if unseen:
            parts.append("new set of categorical values in test set not seen in training set")

        # Dominant category shift
        train_mode = train_col.mode()
        test_mode = test_col.mode()
        if len(train_mode) > 0 and len(test_mode) > 0 and train_mode[0] != test_mode[0]:
            parts.append(f"dominant category changes from '{train_mode[0]}' to '{test_mode[0]}'")

        jsd = drift_metrics.get("jsd", 0) or 0
        if jsd >= 0.05 and not parts:
            parts.append("category proportions shift between train and test")

        if feature_type == "temporal":
            # Check if temporal distribution shifts
            if parts:
                parts = [p.replace("categorical values", "time periods") for p in parts]
            elif jsd >= 0.01:
                parts.append("temporal distribution shifts between train and test")

    if not parts:
        severity = drift_metrics.get("severity", "low")
        if severity == "high":
            parts.append("significant distributional shift detected")
        elif severity == "moderate":
            parts.append("moderate distributional shift detected")
        else:
            parts.append("minor distributional difference")

    desc = "Feature demonstrates " + "; ".join(parts)
    return desc[0].upper() + desc[1:]


def _apply_bh_correction(drift_df: pl.DataFrame, ss_factor: float = 1.0) -> pl.DataFrame:
    """Apply Benjamini-Hochberg FDR correction across all p-values and re-classify severity."""
    # Collect all p-values (KS and Chi2) for joint correction
    n = len(drift_df)
    all_pvals = []
    pval_info = []  # (row_index, pval_col_name)

    has_ks = "ks_pval" in drift_df.columns
    has_chi2 = "chi2_pval" in drift_df.columns

    for i in range(n):
        ft = drift_df["feature_type"][i]
        if ft == "numerical" and has_ks:
            ks_p = drift_df["ks_pval"][i]
            if ks_p is not None:
                all_pvals.append(ks_p)
                pval_info.append((i, "ks_pval"))
        elif ft in ("categorical", "temporal") and has_chi2:
            chi2_p = drift_df["chi2_pval"][i]
            if chi2_p is not None:
                all_pvals.append(chi2_p)
                pval_info.append((i, "chi2_pval"))

    if not all_pvals:
        return drift_df

    corrected = _bh_correct(np.array(all_pvals))

    # Build corrected p-value columns
    ks_bh = [None] * n
    chi2_bh = [None] * n
    for (idx, pval_col), corr_p in zip(pval_info, corrected):
        if pval_col == "ks_pval":
            ks_bh[idx] = round(corr_p, 6)
        else:
            chi2_bh[idx] = round(corr_p, 6)

    drift_df = drift_df.with_columns([
        pl.Series("ks_pval_bh", ks_bh, dtype=pl.Float64),
        pl.Series("chi2_pval_bh", chi2_bh, dtype=pl.Float64),
    ])

    # Re-classify severity using corrected p-values with consistent ss_factor
    # Also reapply missingness escalation (which initial detection applied but
    # would otherwise be lost when severity is overwritten here)
    new_severities = []
    for i in range(n):
        row_ft = drift_df["feature_type"][i]
        train_null = drift_df["train_null_pct"][i] or 0.0
        test_null = drift_df["test_null_pct"][i] or 0.0
        miss_sev = _missingness_severity(train_null, test_null)

        if row_ft == "numerical":
            ks_p = ks_bh[i] if ks_bh[i] is not None else ((drift_df["ks_pval"][i] if has_ks else None) or 1.0)
            psi = (drift_df["psi"][i] if "psi" in drift_df.columns else None) or 0.0
            w_norm = (drift_df["wasserstein_norm"][i] if "wasserstein_norm" in drift_df.columns else None) or 0.0
            ks_d = (drift_df["ks_stat"][i] if "ks_stat" in drift_df.columns else None) or 0.0
            iqr_r = (drift_df["iqr_ratio"][i] if "iqr_ratio" in drift_df.columns else None) or 1.0
            t_div = (drift_df["tail_divergence"][i] if "tail_divergence" in drift_df.columns else None) or 0.0
            sev = _numerical_severity(ks_p, psi, w_norm, ks_d, iqr_r, t_div, ss_factor)
            new_severities.append(_escalate_severity(sev, miss_sev))
        elif row_ft in ("categorical", "temporal"):
            chi2_p = chi2_bh[i] if chi2_bh[i] is not None else ((drift_df["chi2_pval"][i] if has_chi2 else None) or 1.0)
            jsd = (drift_df["jsd"][i] if "jsd" in drift_df.columns else None) or 0.0
            cv = (drift_df["cramers_v"][i] if "cramers_v" in drift_df.columns else None) or 0.0
            cv = cv or 0.0
            sev = _categorical_severity(chi2_p, jsd, cv, ss_factor)
            new_severities.append(_escalate_severity(sev, miss_sev))
        else:
            new_severities.append(drift_df["severity"][i])

    drift_df = drift_df.with_columns(pl.Series("severity", new_severities))
    return drift_df


def _analyze_single_feature(train_col, test_col, col_name, feature_type):
    """Analyze drift for a single feature (used for parallel execution)."""
    if feature_type == "numerical":
        drift = detect_numerical_drift(train_col, test_col)
        desc = generate_drift_description(train_col, test_col, "numerical", drift)
        return {"column": col_name, "feature_type": "numerical", "drift_description": desc, **drift}
    else:
        drift = detect_categorical_drift(train_col, test_col)
        desc = generate_drift_description(train_col, test_col, feature_type, drift)
        drift["unseen_in_test"] = ", ".join(drift["unseen_in_test"]) if drift["unseen_in_test"] else ""
        return {"column": col_name, "feature_type": feature_type, "drift_description": desc, **drift}


def run_drift_analysis(
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_types: dict[str, list[str]],
) -> pl.DataFrame:
    """Run drift detection on all features. Returns a summary DataFrame.

    Applies Benjamini-Hochberg FDR correction across all p-values to
    control for multiple testing, then re-classifies severity.
    Uses parallel execution for datasets with many features.
    """
    from joblib import Parallel, delayed

    # Build list of (col, type) pairs to analyze
    tasks = []
    for col in feature_types["numerical"]:
        if col in train.columns and col in test.columns:
            tasks.append((col, "numerical"))
    for col in feature_types["categorical"]:
        if col in train.columns and col in test.columns:
            tasks.append((col, "categorical"))
    # Temporal columns (e.g. "25-Jan") are timestamps, not features —
    # they *should* differ between train/test. Skip drift detection.

    # Parallelize when there are many features (>20)
    if len(tasks) > 20:
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_analyze_single_feature)(train[col], test[col], col, ftype)
            for col, ftype in tasks
        )
    else:
        results = [
            _analyze_single_feature(train[col], test[col], col, ftype)
            for col, ftype in tasks
        ]

    if not results:
        return pl.DataFrame()

    drift_df = pl.DataFrame(results)
    if len(drift_df) == 0:
        return drift_df

    # Compute dataset-level ss_factor for consistent threshold scaling
    ss_factor = _small_sample_factor(len(train), len(test))

    # Apply BH FDR correction and re-classify severity
    drift_df = _apply_bh_correction(drift_df, ss_factor)

    # Sort by severity: high first, then moderate, low, unknown
    severity_order = {"high": 0, "moderate": 1, "low": 2, "unknown": 3}
    drift_df = drift_df.sort(
        pl.col("severity").replace_strict(severity_order, default=4)
    )

    return drift_df


