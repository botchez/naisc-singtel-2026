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


def detect_numerical_drift(
    train_col: pd.Series, test_col: pd.Series
) -> dict:
    """Run drift tests on a single numerical feature."""
    train_vals = train_col.dropna().values.astype(float)
    test_vals = test_col.dropna().values.astype(float)

    if len(train_vals) == 0 or len(test_vals) == 0:
        return {"ks_stat": None, "ks_pval": None, "psi": None, "wasserstein": None, "severity": "unknown"}

    ks_stat, ks_pval = stats.ks_2samp(train_vals, test_vals)
    psi = compute_psi(train_vals, test_vals)
    wasserstein = stats.wasserstein_distance(train_vals, test_vals)

    severity = _numerical_severity(ks_pval, psi)

    return {
        "ks_stat": round(ks_stat, 4),
        "ks_pval": round(ks_pval, 6),
        "psi": round(psi, 4),
        "wasserstein": round(wasserstein, 4),
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


def _numerical_severity(ks_pval: float, psi: float) -> str:
    """Classify drift severity for numerical features."""
    if psi >= 0.25:
        return "high"
    if psi >= 0.15 and ks_pval < 0.001:
        return "high"
    if psi >= 0.1 or ks_pval < 0.05:
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


def run_drift_analysis(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> pd.DataFrame:
    """Run drift detection on all features. Returns a summary DataFrame."""
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
    if not drift_df.empty:
        drift_df = drift_df.sort_values("severity", key=lambda s: s.map({"high": 0, "moderate": 1, "low": 2, "unknown": 3}))

    return drift_df


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
                print(f"  {col:30s}  KS={row['ks_stat']:.4f} (p={row['ks_pval']:.4f})  PSI={row['psi']:.4f}  Wasserstein={row['wasserstein']:.4f}")
            else:
                unseen = f"  unseen=[{row['unseen_in_test']}]" if row.get("unseen_in_test") else ""
                print(f"  {col:30s}  Chi2={row['chi2_stat']:.4f} (p={row['chi2_pval']:.4f})  JSD={row['jsd']:.4f}{unseen}")

    print("=" * 70)
