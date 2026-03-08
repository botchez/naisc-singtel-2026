import pandas as pd
import numpy as np
from scipy import stats

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
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


def detect_numerical_drift(
    train_col: pd.Series, test_col: pd.Series
) -> dict:
    """Run drift tests on a single numerical feature."""
    train_vals = train_col.dropna().values.astype(float)
    test_vals = test_col.dropna().values.astype(float)

    if len(train_vals) == 0 or len(test_vals) == 0:
        return {"ks_stat": None, "ks_pval": None, "psi": None, "wasserstein": None, "wasserstein_norm": None, "severity": "unknown"}

    ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
    psi = compute_psi(train_vals, test_vals)
    wasserstein = stats.wasserstein_distance(train_vals, test_vals)

    # Normalized Wasserstein (scale-invariant, similar to Cohen's d)
    std_train = np.std(train_vals)
    wasserstein_norm = wasserstein / (std_train + 1e-8)

    severity = _numerical_severity(ks_pval, psi, wasserstein_norm)

    return {
        "ks_stat": round(ks_stat, 4),
        "ks_pval": round(ks_pval, 6),
        "psi": round(psi, 4),
        "wasserstein": round(wasserstein, 4),
        "wasserstein_norm": round(wasserstein_norm, 4),
        "severity": severity,
    }


def detect_categorical_drift(
    train_col: pd.Series, test_col: pd.Series
) -> dict:
    """Run drift tests on a single categorical feature."""
    all_categories = sorted(set(train_col.dropna().unique()) | set(test_col.dropna().unique()))

    if len(all_categories) == 0:
        return {"chi2_stat": None, "chi2_pval": None, "jsd": None, "unseen_in_test": [], "severity": "unknown"}

    train_counts = train_col.value_counts().reindex(all_categories, fill_value=0).values.astype(float)
    test_counts = test_col.value_counts().reindex(all_categories, fill_value=0).values.astype(float)

    # Chi-squared test on proportions
    # Create contingency-style table
    contingency = np.array([train_counts, test_counts])
    # Only run chi2 if we have sufficient data
    if contingency.sum() > 0 and np.all(contingency.sum(axis=0) > 0):
        chi2_stat, chi2_pval, _, _ = stats.chi2_contingency(contingency)
    else:
        chi2_stat, chi2_pval = 0.0, 1.0

    jsd = compute_jsd(train_counts, test_counts)

    # Unseen categories in test
    train_cats = set(train_col.dropna().unique())
    test_cats = set(test_col.dropna().unique())
    unseen_in_test = sorted(test_cats - train_cats)

    severity = _categorical_severity(chi2_pval, jsd)

    return {
        "chi2_stat": round(chi2_stat, 4),
        "chi2_pval": round(chi2_pval, 6),
        "jsd": round(jsd, 4),
        "unseen_in_test": unseen_in_test,
        "severity": severity,
    }


def _numerical_severity(ks_pval: float, psi: float, wasserstein_norm: float = 0.0) -> str:
    """Classify drift severity for numerical features.

    Uses PSI, KS p-value, and normalized Wasserstein distance.
    """
    if psi >= 0.25:
        return "high"
    if psi >= 0.15 and ks_pval < 0.001:
        return "high"
    # Large distributional shift confirmed by both Wasserstein and PSI
    if wasserstein_norm >= 1.0 and psi >= 0.1:
        return "high"
    if psi >= 0.1 or ks_pval < 0.05 or (wasserstein_norm >= 0.3 and psi >= 0.05):
        return "moderate"
    return "low"


def _categorical_severity(chi2_pval: float, jsd: float) -> str:
    """Classify drift severity for categorical features."""
    if jsd >= 0.1:
        return "high"
    if jsd >= 0.05 or chi2_pval < 0.001:
        return "moderate"
    if chi2_pval < 0.05:
        return "moderate"
    return "low"


def _apply_bh_correction(drift_df: pd.DataFrame) -> pd.DataFrame:
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

    # Re-classify severity using corrected p-values
    for idx, row in drift_df.iterrows():
        if row["feature_type"] == "numerical":
            ks_p = row.get("ks_pval_bh", row.get("ks_pval", 1.0))
            if pd.isna(ks_p):
                ks_p = row.get("ks_pval", 1.0)
            psi = row.get("psi") or 0.0
            w_norm = row.get("wasserstein_norm") or 0.0
            drift_df.loc[idx, "severity"] = _numerical_severity(ks_p, psi, w_norm)
        elif row["feature_type"] in ("categorical", "temporal"):
            chi2_p = row.get("chi2_pval_bh", row.get("chi2_pval", 1.0))
            if pd.isna(chi2_p):
                chi2_p = row.get("chi2_pval", 1.0)
            jsd = row.get("jsd") or 0.0
            drift_df.loc[idx, "severity"] = _categorical_severity(chi2_p, jsd)

    return drift_df


def run_drift_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> pd.DataFrame:
    """Run drift detection on all features. Returns a summary DataFrame.

    Applies Benjamini-Hochberg FDR correction across all p-values to
    control for multiple testing, then re-classifies severity.
    """
    results = []

    for col in feature_types["numerical"]:
        if col not in train.columns or col not in test.columns:
            continue
        drift = detect_numerical_drift(train[col], test[col])
        results.append({"column": col, "feature_type": "numerical", **drift})

    for col in feature_types["categorical"]:
        if col not in train.columns or col not in test.columns:
            continue
        drift = detect_categorical_drift(train[col], test[col])
        # Convert list to string for display
        drift["unseen_in_test"] = ", ".join(drift["unseen_in_test"]) if drift["unseen_in_test"] else ""
        results.append({"column": col, "feature_type": "categorical", **drift})

    for col in feature_types["temporal"]:
        if col not in train.columns or col not in test.columns:
            continue
        drift = detect_categorical_drift(train[col], test[col])
        drift["unseen_in_test"] = ", ".join(drift["unseen_in_test"]) if drift["unseen_in_test"] else ""
        results.append({"column": col, "feature_type": "temporal", **drift})

    drift_df = pd.DataFrame(results)
    if drift_df.empty:
        return drift_df

    # Apply BH FDR correction and re-classify severity
    drift_df = _apply_bh_correction(drift_df)

    drift_df = drift_df.sort_values("severity", key=lambda s: s.map({"high": 0, "moderate": 1, "low": 2, "unknown": 3}))

    return drift_df


def detect_multivariate_shift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    extra_drop_cols: list[str] | None = None,
    n_splits: int = 5,
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
        return {"auc": 0.5, "fold_aucs": [], "shift_detected": False}

    X_tr = train[feature_cols].copy()
    X_te = test[feature_cols].copy()

    # Simple preprocessing: label encode objects, fill NaN
    for col in feature_cols:
        if X_tr[col].dtype == object:
            le = LabelEncoder()
            combined = pd.concat([X_tr[col].astype(str), X_te[col].astype(str)], ignore_index=True)
            le.fit(combined)
            X_tr[col] = le.transform(X_tr[col].astype(str))
            X_te[col] = le.transform(X_te[col].astype(str))
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            fill = X_tr[col].median()
            if pd.isna(fill):
                fill = 0
            X_tr[col] = X_tr[col].fillna(fill)
            X_te[col] = X_te[col].fillna(fill)
        else:
            X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce").fillna(0)
            X_te[col] = pd.to_numeric(X_te[col], errors="coerce").fillna(0)

    X_all = pd.concat([X_tr, X_te], ignore_index=True)
    y_domain = np.array([0] * len(X_tr) + [1] * len(X_te))

    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_proba = np.zeros(len(X_all))
    fold_aucs = []

    for fold_train_idx, fold_val_idx in skf.split(X_all, y_domain):
        clf = LGBMClassifier(
            verbosity=-1,
            objective="binary",
            is_unbalance=True,
            random_state=42,
            importance_type="gain",
        )
        clf.fit(X_all.iloc[fold_train_idx], y_domain[fold_train_idx])
        oof_proba[fold_val_idx] = clf.predict_proba(X_all.iloc[fold_val_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_domain[fold_val_idx], oof_proba[fold_val_idx]))

    from utils import logger
    auc = roc_auc_score(y_domain, oof_proba)
    logger.info(f"Domain classifier fold AUCs: {[round(a, 4) for a in fold_aucs]}, overall={round(auc, 4)}")

    return {
        "auc": round(auc, 4),
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "shift_detected": auc > 0.65,
    }


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
                unseen = f"  unseen=[{row['unseen_in_test']}]" if row.get("unseen_in_test") else ""
                print(f"  {col:30s}  Chi2={row['chi2_stat']:.4f} (p={row['chi2_pval']:.4f})  JSD={row['jsd']:.4f}{unseen}")

    print("=" * 70)
