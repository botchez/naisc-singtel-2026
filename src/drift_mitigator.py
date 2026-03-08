import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from utils import logger


def _get_feature_importances(
    train: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> dict[str, float]:
    """Train a quick model to get feature importance scores."""
    target_col = feature_types.get("target", ["ChurnStatus"])[0]
    id_col = feature_types.get("id", ["CustomerID"])[0]

    if target_col not in train.columns:
        return {}

    skip = {id_col, target_col} | set(feature_types.get("temporal", []))
    feature_cols = [c for c in train.columns if c not in skip]
    X = train[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == object or not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X[col] = X[col].fillna(X[col].median())

    y = (train[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})).astype(int)

    clf = LGBMClassifier(
        n_estimators=100, max_depth=4, verbosity=-1, random_state=42,
        importance_type="gain",
    )
    clf.fit(X, y)

    importances = dict(zip(X.columns, clf.feature_importances_))
    # Normalize to 0-1
    max_imp = max(importances.values()) if importances else 1.0
    if max_imp > 0:
        importances = {k: v / max_imp for k, v in importances.items()}
    return importances


def _classify_action(severity: str, importance: float) -> str:
    """Determine action based on drift severity and feature importance.

    Returns: 'keep', 'transform', or 'drop'
    """
    high_importance = importance >= 0.1

    if severity == "high":
        return "transform" if high_importance else "drop"
    if severity == "moderate":
        return "transform" if high_importance else "keep"
    return "keep"


def _quantile_normalize(
    train_col: pd.Series,
    test_col: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Quantile-normalize a numerical column to reduce distribution shift.

    Fits on train, transforms both. Maps to uniform [0,1] distribution
    so train and test have comparable scales regardless of shift.
    """
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(train_col)),
        output_distribution="uniform",
        random_state=42,
    )
    train_vals = train_col.fillna(train_col.median()).values.reshape(-1, 1)
    test_vals = test_col.fillna(train_col.median()).values.reshape(-1, 1)

    qt.fit(train_vals)
    return (
        pd.Series(qt.transform(train_vals).ravel(), index=train_col.index, name=train_col.name),
        pd.Series(qt.transform(test_vals).ravel(), index=test_col.index, name=test_col.name),
    )


def apply_feature_mitigations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drift_df: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply tiered feature-level mitigation based on drift severity x importance.

    Actions:
      - keep:      no change
      - transform:  quantile-normalize numericals, frequency-encode high-card categoricals
      - drop:       remove feature entirely

    Returns (train_mitigated, test_mitigated, action_log).
    """
    if drift_df.empty:
        action_log = pd.DataFrame(columns=["column", "severity", "importance", "action"])
        return train, test, action_log

    train_out = train.copy()
    test_out = test.copy()

    importances = _get_feature_importances(train, feature_types)

    actions = []
    skip_cols = set(
        feature_types.get("id", [])
        + feature_types.get("target", [])
        + feature_types.get("temporal", [])
    )

    for _, row in drift_df.iterrows():
        col = row["column"]
        severity = row["severity"]
        ftype = row.get("feature_type", "unknown")

        if col in skip_cols:
            continue
        if col not in train_out.columns:
            continue

        imp = importances.get(col, 0.0)
        action = _classify_action(severity, imp)

        if action == "drop":
            train_out.drop(columns=[col], inplace=True)
            test_out.drop(columns=[col], inplace=True)

        elif action == "transform":
            if ftype == "numerical" and pd.api.types.is_numeric_dtype(train_out[col]):
                train_out[col], test_out[col] = _quantile_normalize(
                    train_out[col], test_out[col]
                )
            elif ftype == "categorical":
                # Frequency-encode to stabilize against unseen/shifted categories
                freq_map = train[col].value_counts(normalize=True).to_dict()
                train_out[col] = train_out[col].map(freq_map).fillna(0).astype(float)
                test_out[col] = test_out[col].map(freq_map).fillna(0).astype(float)

        actions.append({
            "column": col,
            "type": ftype,
            "severity": severity,
            "importance": round(imp, 4),
            "action": action,
        })

    action_log = pd.DataFrame(actions)

    n_keep = sum(1 for a in actions if a["action"] == "keep")
    n_transform = sum(1 for a in actions if a["action"] == "transform")
    n_drop = sum(1 for a in actions if a["action"] == "drop")
    logger.info(f"Feature mitigation: {n_keep} keep, {n_transform} transform, {n_drop} drop")

    return train_out, test_out, action_log


def print_mitigation_summary(action_log: pd.DataFrame) -> None:
    """Print feature-level mitigation actions to console."""
    print("\n" + "=" * 70)
    print("FEATURE-LEVEL DRIFT MITIGATION")
    print("=" * 70)

    if action_log.empty:
        print("  No mitigation actions taken.")
        print("=" * 70)
        return

    print(f"  {'Column':<30s} {'Severity':<10s} {'Importance':>10s} {'Action':<12s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")

    for action in ["drop", "transform", "keep"]:
        subset = action_log[action_log["action"] == action]
        for _, row in subset.iterrows():
            print(f"  {row['column']:<30s} {row['severity']:<10s} "
                  f"{row['importance']:>10.4f} {row['action']:<12s}")

    counts = action_log["action"].value_counts()
    print(f"\n  Summary: {counts.get('keep', 0)} kept, "
          f"{counts.get('transform', 0)} transformed, "
          f"{counts.get('drop', 0)} dropped")
    print("=" * 70)
