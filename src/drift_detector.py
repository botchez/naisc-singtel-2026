import pandas as pd
import numpy as np
from scipy import stats

# Maximum sample size for statistical tests. Beyond this, additional rows
# add negligible statistical power but cost O(n log n) compute.  50K is
# enough for KS, PSI, Chi², and Wasserstein at any practical significance level.
_MAX_DRIFT_SAMPLE = 50_000


def _subsample_array(arr: np.ndarray, max_n: int = _MAX_DRIFT_SAMPLE, seed: int = 42) -> np.ndarray:
    """Randomly subsample an array if it exceeds max_n."""
    if len(arr) <= max_n:
        return arr
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(arr), size=max_n, replace=False)
    return arr[idx]


def _subsample_series(s: pd.Series, max_n: int = _MAX_DRIFT_SAMPLE, seed: int = 42) -> pd.Series:
    """Randomly subsample a Series if it exceeds max_n."""
    if len(s) <= max_n:
        return s
    return s.sample(n=max_n, random_state=seed)


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
    """Cramér's V: sample-size-normalized chi-squared effect size.

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
    train_col: pd.Series, test_col: pd.Series
) -> dict:
    """Run drift tests on a single numerical feature."""
    train_null_pct = float(train_col.isna().mean())
    test_null_pct = float(test_col.isna().mean())
    miss_sev = _missingness_severity(train_null_pct, test_null_pct)

    train_vals = pd.to_numeric(train_col, errors="coerce").dropna().values.astype(float)
    test_vals = pd.to_numeric(test_col, errors="coerce").dropna().values.astype(float)

    if len(train_vals) == 0 or len(test_vals) == 0:
        return {"ks_stat": None, "ks_pval": None, "psi": None, "wasserstein": None,
                "wasserstein_norm": None, "train_null_pct": round(train_null_pct, 4),
                "test_null_pct": round(test_null_pct, 4), "severity": _escalate_severity("unknown", miss_sev)}

    # Subsample for statistical tests — 50K rows gives equivalent power
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
    train_col: pd.Series, test_col: pd.Series
) -> dict:
    """Run drift tests on a single categorical feature."""
    train_null_pct = float(train_col.isna().mean())
    test_null_pct = float(test_col.isna().mean())
    miss_sev = _missingness_severity(train_null_pct, test_null_pct)

    # Subsample for chi-squared / JSD — preserves distribution proportions
    train_sub = _subsample_series(train_col.dropna())
    test_sub = _subsample_series(test_col.dropna())

    all_categories = sorted(set(train_sub.unique()) | set(test_sub.unique()), key=str)

    if len(all_categories) == 0:
        return {"chi2_stat": None, "chi2_pval": None, "jsd": None, "unseen_in_test": [],
                "train_null_pct": round(train_null_pct, 4), "test_null_pct": round(test_null_pct, 4),
                "severity": _escalate_severity("unknown", miss_sev)}

    train_counts = train_sub.value_counts().reindex(all_categories, fill_value=0).values.astype(float)
    test_counts = test_sub.value_counts().reindex(all_categories, fill_value=0).values.astype(float)

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

    # Cramér's V: sample-size-normalized effect size
    n_total = int(contingency.sum())
    k_cats = min(contingency.shape[0], contingency.shape[1])
    cramers_v = compute_cramers_v(chi2_stat, n_total, k_cats)

    # Unseen categories in test
    train_cats = set(train_sub.unique())
    test_cats = set(test_sub.unique())
    unseen_in_test = sorted(test_cats - train_cats, key=str)

    # Use original full sizes for small-sample factor
    n_tr = len(train_col.dropna())
    n_te = len(test_col.dropna())
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

    Uses JSD, Chi² p-value, and Cramér's V (sample-size-normalized effect size).
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
    train_col: pd.Series, test_col: pd.Series, feature_type: str, drift_metrics: dict
) -> str:
    """Generate a qualitative natural-language description of the drift observed."""
    parts = []

    if feature_type == "numerical":
        train_vals = _subsample_array(pd.to_numeric(train_col, errors="coerce").dropna().values.astype(float))
        test_vals = _subsample_array(pd.to_numeric(test_col, errors="coerce").dropna().values.astype(float))
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
        if len(train_mode) > 0 and len(test_mode) > 0 and train_mode.iloc[0] != test_mode.iloc[0]:
            parts.append(f"dominant category changes from '{train_mode.iloc[0]}' to '{test_mode.iloc[0]}'")

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


def _apply_bh_correction(drift_df: pd.DataFrame, ss_factor: float = 1.0) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction across all p-values and re-classify severity."""
    # Collect all p-values (KS and Chi2) for joint correction
    all_pvals = []
    pval_info = []  # (row_idx, pval_col_name)

    for idx, row in drift_df.iterrows():
        if row["feature_type"] == "numerical" and pd.notna(row.get("ks_pval")):
            all_pvals.append(row["ks_pval"])
            pval_info.append((idx, "ks_pval"))
        elif row["feature_type"] in ("categorical", "temporal") and pd.notna(row.get("chi2_pval")):
            all_pvals.append(row["chi2_pval"])
            pval_info.append((idx, "chi2_pval"))

    if not all_pvals:
        return drift_df

    corrected = _bh_correct(np.array(all_pvals))

    # Store corrected p-values alongside originals
    if "ks_pval_bh" not in drift_df.columns:
        drift_df["ks_pval_bh"] = np.nan
    if "chi2_pval_bh" not in drift_df.columns:
        drift_df["chi2_pval_bh"] = np.nan

    for (idx, pval_col), corr_p in zip(pval_info, corrected):
        drift_df.loc[idx, f"{pval_col}_bh"] = round(corr_p, 6)

    # Re-classify severity using corrected p-values with consistent ss_factor
    # Also reapply missingness escalation (which initial detection applied but
    # would otherwise be lost when severity is overwritten here)
    for idx, row in drift_df.iterrows():
        train_null = row.get("train_null_pct") or 0.0
        test_null = row.get("test_null_pct") or 0.0
        miss_sev = _missingness_severity(train_null, test_null)

        if row["feature_type"] == "numerical":
            ks_p = row.get("ks_pval_bh", row.get("ks_pval", 1.0))
            if pd.isna(ks_p):
                ks_p = row.get("ks_pval", 1.0)
            psi = row.get("psi") or 0.0
            w_norm = row.get("wasserstein_norm") or 0.0
            ks_d = row.get("ks_stat") or 0.0
            iqr_r = row.get("iqr_ratio") or 1.0
            t_div = row.get("tail_divergence") or 0.0
            sev = _numerical_severity(ks_p, psi, w_norm, ks_d, iqr_r, t_div, ss_factor)
            drift_df.loc[idx, "severity"] = _escalate_severity(sev, miss_sev)
        elif row["feature_type"] in ("categorical", "temporal"):
            chi2_p = row.get("chi2_pval_bh", row.get("chi2_pval", 1.0))
            if pd.isna(chi2_p):
                chi2_p = row.get("chi2_pval", 1.0)
            jsd = row.get("jsd") or 0.0
            cv = row.get("cramers_v") or 0.0
            sev = _categorical_severity(chi2_p, jsd, cv, ss_factor)
            drift_df.loc[idx, "severity"] = _escalate_severity(sev, miss_sev)

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
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> pd.DataFrame:
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
    for col in feature_types["temporal"]:
        if col in train.columns and col in test.columns:
            tasks.append((col, "temporal"))

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

    drift_df = pd.DataFrame(results)
    if drift_df.empty:
        return drift_df

    # Compute dataset-level ss_factor for consistent threshold scaling
    ss_factor = _small_sample_factor(len(train), len(test))

    # Apply BH FDR correction and re-classify severity
    drift_df = _apply_bh_correction(drift_df, ss_factor)

    drift_df = drift_df.sort_values("severity", key=lambda s: s.map({"high": 0, "moderate": 1, "low": 2, "unknown": 3}))

    return drift_df


def detect_multivariate_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    extra_drop_cols: list[str] | None = None,
    n_splits: int = 3,
) -> dict:
    """Classifier-based multivariate covariate shift detection.

    Trains a LightGBM domain classifier to distinguish train vs test samples.
    AUC near 0.5 means no shift; AUC near 1.0 means strong shift.

    Returns dict with 'auc', 'fold_aucs', 'shift_detected'.
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import LabelEncoder

    drop_cols = feature_types.get("id", []) + feature_types.get("target", [])
    if extra_drop_cols:
        drop_cols = drop_cols + [c for c in extra_drop_cols if c not in drop_cols]
    feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]

    if not feature_cols:
        return {"auc": 0.5, "fold_aucs": [], "shift_detected": False, "train_oof_proba": None}

    # Cap features for domain classifier efficiency (LightGBM cost is O(n*m*trees))
    _MAX_DOMAIN_FEATURES = 100
    if len(feature_cols) > _MAX_DOMAIN_FEATURES:
        _orig_n_feats = len(feature_cols)
        _var_sample = train[feature_cols].sample(
            n=min(50_000, len(train)), random_state=42
        ) if len(train) > 50_000 else train[feature_cols]
        _variances = _var_sample.apply(pd.to_numeric, errors="coerce").var().sort_values(ascending=False)
        feature_cols = _variances.head(_MAX_DOMAIN_FEATURES).index.tolist()
        from utils import logger as _logger
        _logger.info(f"Domain classifier: using top {_MAX_DOMAIN_FEATURES}/{_orig_n_feats} features by variance")

    X_tr = train[feature_cols].copy()
    X_te = test[feature_cols].copy()

    # Subsample for domain classifier when datasets are large
    # 100K per split is sufficient for shift detection
    _MAX_DOMAIN_SAMPLE = 100_000
    full_train_n = len(X_tr)
    if len(X_tr) > _MAX_DOMAIN_SAMPLE:
        X_tr = X_tr.sample(n=_MAX_DOMAIN_SAMPLE, random_state=42).reset_index(drop=True)
    if len(X_te) > _MAX_DOMAIN_SAMPLE:
        X_te = X_te.sample(n=_MAX_DOMAIN_SAMPLE, random_state=42).reset_index(drop=True)

    # Simple preprocessing: encode objects via dict mapping, fill NaN
    _cat_maps = {}  # Save mappings for reuse on full train data
    _fill_vals = {}
    for col in feature_cols:
        if X_tr[col].dtype == object:
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)], ignore_index=True)
            cat_map = {c: i for i, c in enumerate(combined.unique())}
            X_tr[col] = X_tr[col].astype(str).map(cat_map).astype(int)
            X_te[col] = X_te[col].astype(str).map(cat_map).astype(int)
            _cat_maps[col] = cat_map
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            fill = X_tr[col].median()
            if pd.isna(fill):
                fill = 0
            _fill_vals[col] = fill
            X_tr[col] = X_tr[col].fillna(fill)
            X_te[col] = X_te[col].fillna(fill)
        else:
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(0)
            X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(0)

    X_all = pd.concat([X_tr, X_te], ignore_index=True)
    y_domain = np.array([0] * len(X_tr) + [1] * len(X_te))

    # Adapt fold count for small datasets
    from utils import safe_n_splits
    n_splits = safe_n_splits(len(X_all), desired=n_splits, min_per_fold=20)

    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_all))
    fold_aucs = []
    importance_acc = np.zeros(len(feature_cols))

    for fold_train_idx, fold_val_idx in skf.split(X_all, y_domain):
        clf = LGBMClassifier(
            verbosity=-1,
            objective="binary",
            is_unbalance=True,
            random_state=42,
            importance_type="gain",
            n_estimators=50,  # Reduced: domain classifier is diagnostic
        )
        clf.fit(X_all.iloc[fold_train_idx], y_domain[fold_train_idx])
        oof_proba[fold_val_idx] = clf.predict_proba(X_all.iloc[fold_val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_domain[fold_val_idx], oof_proba[fold_val_idx]))
        importance_acc += clf.feature_importances_

    from utils import logger
    auc = roc_auc_score(y_domain, oof_proba)
    logger.info(f"Domain classifier fold AUCs: {[round(a, 4) for a in fold_aucs]}, overall={round(auc, 4)}")

    # Log top features driving multivariate shift
    avg_importance = importance_acc / n_splits
    top_k = min(5, len(feature_cols))
    top_idx = np.argsort(avg_importance)[-top_k:][::-1]
    top_features = [(feature_cols[i], round(float(avg_importance[i]), 1)) for i in top_idx]
    logger.info(f"Top domain-shift features (gain): {top_features}")

    # Extract per-training-row P(test) for sample weighting
    # If training data was subsampled for the domain classifier, we need
    # propensity scores for ALL training rows (used for sample weighting).
    # Use the last fold's classifier to score the full training set.
    if full_train_n > _MAX_DOMAIN_SAMPLE:
        X_tr_full = train[feature_cols].copy()
        for col in feature_cols:
            if col in _cat_maps:
                cat_map = _cat_maps[col]
                default_code = len(cat_map)
                X_tr_full[col] = X_tr_full[col].astype(str).map(cat_map).fillna(default_code).astype(int)
            elif pd.api.types.is_numeric_dtype(X_tr_full[col]):
                fill = _fill_vals.get(col, 0)
                X_tr_full[col] = X_tr_full[col].fillna(fill)
            else:
                X_tr_full[col] = pd.to_numeric(X_tr_full[col], errors="coerce").fillna(0)
        train_oof_proba = clf.predict_proba(X_tr_full)[:, 1]
    else:
        train_oof_proba = oof_proba[:len(X_tr)]

    return {
        "auc": round(auc, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "shift_detected": auc > 0.65,
        "train_oof_proba": train_oof_proba,
        "top_shift_features": top_features,
    }


def detect_label_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> dict | None:
    """Detect prior probability shift in the target between train and test.

    Compares target class distribution (churn rate) between datasets.
    Returns None if test target is not available.
    """
    from utils import logger

    target_cols = feature_types.get("target", [])
    target_col = target_cols[0] if target_cols else None
    if target_col is None or target_col not in train.columns:
        return None
    if target_col not in test.columns or test[target_col].isna().all():
        logger.info("Label shift detection skipped: no target in test")
        return None

    try:
        from model import _encode_target
        y_train = _encode_target(train[target_col])
        y_test = _encode_target(test[target_col])

        train_rate = float(y_train.mean())
        test_rate = float(y_test.mean())
        rate_diff = abs(test_rate - train_rate)

        # Proportion test (chi-squared)
        n_train, n_test = len(y_train), len(y_test)
        contingency = np.array([
            [int(y_train.sum()), n_train - int(y_train.sum())],
            [int(y_test.sum()), n_test - int(y_test.sum())],
        ])
        chi2_stat, chi2_pval, _, _ = stats.chi2_contingency(contingency)

        # Classify severity
        if rate_diff > 0.10 and chi2_pval < 0.001:
            severity = "high"
        elif rate_diff > 0.03 or chi2_pval < 0.05:
            severity = "moderate"
        else:
            severity = "low"

        logger.info(
            f"Label shift: train churn={train_rate:.4f}, test churn={test_rate:.4f}, "
            f"diff={rate_diff:.4f}, severity={severity}"
        )

        return {
            "train_rate": round(train_rate, 4),
            "test_rate": round(test_rate, 4),
            "rate_diff": round(rate_diff, 4),
            "chi2_stat": round(chi2_stat, 4),
            "chi2_pval": round(chi2_pval, 6),
            "severity": severity,
        }
    except Exception as e:
        logger.warning(f"Label shift detection failed: {e}")
        return None


def detect_intra_training_drift(
    train: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Detect temporal drift within training data by comparing early vs late months.

    Splits training data into early (first half of months) and late (second half)
    based on the temporal column, then runs drift detection between the two halves.

    Returns (intra_drift_df, month_series) where month_series is the parsed month
    ordering per row (or None if no temporal column).
    """
    from utils import logger

    temporal_cols = feature_types.get("temporal", [])
    temporal_col = None
    for col in temporal_cols:
        if col in train.columns:
            temporal_col = col
            break

    if temporal_col is None:
        logger.info("No temporal column found; skipping intra-training drift detection")
        return pd.DataFrame(), None

    # Parse temporal column to month ordering
    raw = train[temporal_col].astype(str).str.title()
    month_series = None
    for fmt in ["%y-%b", "%b-%y", "%Y-%m", "%m-%Y", "%b %Y", "%Y-%b",
                "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if parsed.notna().sum() > len(train) * 0.5:
            month_series = parsed.dt.to_period("M").astype(str)
            break

    if month_series is None:
        logger.info(f"Could not parse temporal column '{temporal_col}'; skipping intra-training drift")
        return pd.DataFrame(), None

    # Sort unique months and split into early vs late halves
    unique_months = sorted(month_series.dropna().unique())
    if len(unique_months) < 2:
        logger.info("Only one month in training data; skipping intra-training drift")
        return pd.DataFrame(), None

    mid = len(unique_months) // 2
    early_months = set(unique_months[:mid])
    late_months = set(unique_months[mid:])

    early_mask = month_series.isin(early_months)
    late_mask = month_series.isin(late_months)
    train_early = train[early_mask]
    train_late = train[late_mask]

    # Subsample for intra-training drift when dataset is large
    _MAX_INTRA_SAMPLE = 100_000
    if len(train_early) > _MAX_INTRA_SAMPLE:
        train_early = train_early.sample(n=_MAX_INTRA_SAMPLE, random_state=42)
    if len(train_late) > _MAX_INTRA_SAMPLE:
        train_late = train_late.sample(n=_MAX_INTRA_SAMPLE, random_state=42)

    logger.info(
        f"Intra-training split: early={len(train_early)} rows ({len(early_months)} months), "
        f"late={len(train_late)} rows ({len(late_months)} months)"
    )

    # Run drift detection between early and late halves
    results = []
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []) + temporal_cols)

    for col in feature_types["numerical"]:
        if col in skip_cols or col not in train.columns:
            continue
        drift = detect_numerical_drift(train_early[col], train_late[col])
        if drift["severity"] in ("high", "moderate"):
            desc = generate_drift_description(train_early[col], train_late[col], "numerical", drift)
            results.append({"column": col, "feature_type": "numerical", "drift_description": desc, **drift})

    for col in feature_types["categorical"]:
        if col in skip_cols or col not in train.columns:
            continue
        drift = detect_categorical_drift(train_early[col], train_late[col])
        if drift["severity"] in ("high", "moderate"):
            desc = generate_drift_description(train_early[col], train_late[col], "categorical", drift)
            drift["unseen_in_test"] = ", ".join(drift["unseen_in_test"]) if drift["unseen_in_test"] else ""
            results.append({"column": col, "feature_type": "categorical", "drift_description": desc, **drift})

    intra_drift_df = pd.DataFrame(results)
    if not intra_drift_df.empty:
        logger.info(f"Intra-training drift: {len(intra_drift_df)} features with high/moderate temporal drift")
    else:
        logger.info("No significant intra-training drift detected")

    # Return month_series as parsed month rank for recency weighting
    month_rank = pd.Series(0, index=train.index)
    month_to_rank = {m: i for i, m in enumerate(unique_months)}
    month_rank = month_series.map(month_to_rank).fillna(0).astype(int)

    return intra_drift_df, month_rank


def print_drift_summary(drift_df: pd.DataFrame) -> None:
    """Print drift detection summary to console."""
    print("\n" + "=" * 70)
    print("DATA DRIFT DETECTION SUMMARY")
    print("=" * 70)

    if drift_df.empty:
        print("No features analysed.")
        return

    for severity in ["high", "moderate", "low"]:
        subset = drift_df[drift_df["severity"] == severity]
        if subset.empty:
            continue
        print(f"\n--- {severity.upper()} drift ({len(subset)} features) ---")
        for _, row in subset.iterrows():
            col = row["column"]
            ftype = row["feature_type"]
            if ftype == "numerical":
                ks_p = row.get('ks_pval_bh', row['ks_pval'])
                ks_p = row['ks_pval'] if pd.isna(ks_p) else ks_p
                w_norm = row.get('wasserstein_norm', 0)
                w_norm = 0 if pd.isna(w_norm) else w_norm
                print(f"  {col:30s}  KS={row['ks_stat']:.4f} (p={ks_p:.4f} BH)  PSI={row['psi']:.4f}  W*={w_norm:.4f}")
            else:
                cv = row.get('cramers_v', 0) or 0
                unseen = f"  unseen=[{row['unseen_in_test']}]" if row.get("unseen_in_test") else ""
                print(f"  {col:30s}  Chi2={row['chi2_stat']:.4f} (p={row['chi2_pval']:.4f})  JSD={row['jsd']:.4f}  V={cv:.4f}{unseen}")

    print("=" * 70)
