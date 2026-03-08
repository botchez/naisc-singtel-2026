import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from utils import logger


def _frequency_encode(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
) -> tuple[pd.Series, pd.Series]:
    """Frequency-encode a column based on training set frequencies."""
    freq_map = train[col].value_counts(normalize=True).to_dict()
    return (
        train[col].map(freq_map).fillna(0).astype(float),
        test[col].map(freq_map).fillna(0).astype(float),
    )


def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    drift_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate engineered features dynamically. No hardcoded column names.

    Skips high-drift columns to avoid generating noisy features from
    columns that differ significantly between train and test.

    Returns enriched copies of train and test with new columns added.
    """
    train_out = train.copy()
    test_out = test.copy()

    skip_cols = set(
        feature_types.get("id", [])
        + feature_types.get("target", [])
        + feature_types.get("temporal", [])
    )

    # Exclude high-drift columns from feature engineering
    high_drift_cols = set()
    if drift_df is not None and not drift_df.empty:
        high_drift_cols = set(
            drift_df.loc[drift_df["severity"] == "high", "column"].tolist()
        )
        skip_cols |= high_drift_cols

    num_cols = [c for c in feature_types.get("numerical", []) if c in train.columns and c not in skip_cols]
    cat_cols = [c for c in feature_types.get("categorical", []) if c in train.columns and c not in skip_cols]

    new_features = []

    # 0. Cyclical encoding for temporal columns (sin/cos instead of 1-12)
    for col in feature_types.get("temporal", []):
        if col not in train.columns:
            continue
        for df_in, df_out in [(train, train_out), (test, test_out)]:
            months = pd.to_datetime(df_in[col], format="%y-%b", errors="coerce").dt.month
            # Also try reverse format
            if months.isna().all():
                months = pd.to_datetime(df_in[col], format="%b-%y", errors="coerce").dt.month
            months = months.fillna(6).astype(float)  # default to mid-year
            df_out[f"{col}_sin"] = np.sin(2 * np.pi * months / 12)
            df_out[f"{col}_cos"] = np.cos(2 * np.pi * months / 12)
        new_features.extend([f"{col}_sin", f"{col}_cos"])

    # 1. Frequency encoding for high-cardinality categoricals
    for col in cat_cols:
        nunique = train[col].nunique()
        if nunique > 10:
            fname = f"{col}_freq"
            train_out[fname], test_out[fname] = _frequency_encode(train, test, col)
            new_features.append(fname)

    # 2. Missing value indicators
    for col in train.columns:
        if col in skip_cols:
            continue
        null_rate = train[col].isna().mean()
        if null_rate > 0.01:
            fname = f"{col}_missing"
            train_out[fname] = train[col].isna().astype(int)
            test_out[fname] = test[col].isna().astype(int)
            new_features.append(fname)

    # 3. Ratio features between numerical pairs (limited to avoid explosion)
    if len(num_cols) >= 2:
        # Select top pairs by absolute correlation with target if available
        target_col = feature_types.get("target", [None])[0] if feature_types.get("target") else None
        ratio_pairs = _select_ratio_pairs(train, num_cols, target_col, max_pairs=15)

        for col_a, col_b in ratio_pairs:
            fname = f"{col_a}_div_{col_b}"
            for df in [train_out, test_out]:
                denominator = df[col_b].fillna(0).replace(0, 1)
                df[fname] = df[col_a].fillna(0) / denominator
            new_features.append(fname)

    # 4. Interaction features: top categorical pairs -> frequency-encoded
    if len(cat_cols) >= 2:
        interaction_pairs = _select_interaction_pairs(cat_cols, max_pairs=10)
        for col_a, col_b in interaction_pairs:
            fname = f"{col_a}_x_{col_b}"
            train_out[fname] = train[col_a].astype(str) + "_" + train[col_b].astype(str)
            test_out[fname] = test[col_a].astype(str) + "_" + test[col_b].astype(str)
            # Frequency-encode the interaction
            freq_fname = f"{fname}_freq"
            train_out[freq_fname], test_out[freq_fname] = _frequency_encode(train_out, test_out, fname)
            # Drop the raw interaction string, keep only frequency
            train_out.drop(columns=[fname], inplace=True)
            test_out.drop(columns=[fname], inplace=True)
            new_features.append(freq_fname)

    # 5. Quantile-based binning for numericals
    for col in num_cols:
        fname = f"{col}_qbin"
        try:
            train_out[fname], bin_edges = pd.qcut(
                train[col], q=5, labels=False, retbins=True, duplicates="drop"
            )
            test_out[fname] = pd.cut(
                test[col], bins=bin_edges, labels=False, include_lowest=True
            )
            # Fill NaN bins with median bin
            median_bin = train_out[fname].median()
            train_out[fname] = train_out[fname].fillna(median_bin).astype(float)
            test_out[fname] = test_out[fname].fillna(median_bin).astype(float)
            new_features.append(fname)
        except (ValueError, IndexError):
            # Not enough unique values to bin
            if fname in train_out.columns:
                train_out.drop(columns=[fname], inplace=True)
            if fname in test_out.columns:
                test_out.drop(columns=[fname], inplace=True)

    # 6. Group aggregation stats: mean of numericals within categorical groups
    agg_count = 0
    max_agg = 20
    for cat_col in cat_cols:
        if agg_count >= max_agg:
            break
        if train[cat_col].nunique() < 2 or train[cat_col].nunique() > 100:
            continue
        for num_col in num_cols:
            if agg_count >= max_agg:
                break
            fname_mean = f"{num_col}_by_{cat_col}_mean"
            group_stats = train.groupby(cat_col)[num_col].mean().to_dict()
            global_mean = train[num_col].mean()
            train_out[fname_mean] = train[cat_col].map(group_stats).fillna(global_mean)
            test_out[fname_mean] = test[cat_col].map(group_stats).fillna(global_mean)
            new_features.append(fname_mean)
            agg_count += 1

    logger.info(f"Feature engineering: generated {len(new_features)} candidate features")

    # Filter: keep only engineered features that have non-trivial importance
    if new_features:
        train_out, test_out, kept, dropped = _filter_by_importance(
            train_out, test_out, feature_types, new_features
        )
        logger.info(f"Feature selection: kept {len(kept)}, dropped {len(dropped)} "
                    f"(total: {train_out.shape[1]} columns)")

    return train_out, test_out


def _select_ratio_pairs(
    train: pd.DataFrame,
    num_cols: list[str],
    target_col: str | None,
    max_pairs: int = 15,
) -> list[tuple[str, str]]:
    """Select numerical column pairs for ratio features.

    Prioritizes pairs where both columns correlate with the target.
    Falls back to variance-based selection if no target.
    """
    if target_col and target_col in train.columns:
        try:
            target_numeric = pd.to_numeric(
                train[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0}),
                errors="coerce"
            )
            correlations = {}
            for col in num_cols:
                corr = train[col].corr(target_numeric)
                if pd.notna(corr):
                    correlations[col] = abs(corr)
            # Sort by correlation, take top columns
            sorted_cols = sorted(correlations, key=correlations.get, reverse=True)
            top_cols = sorted_cols[:8]
        except Exception:
            top_cols = num_cols[:8]
    else:
        top_cols = num_cols[:8]

    pairs = []
    for i, col_a in enumerate(top_cols):
        for col_b in top_cols[i + 1:]:
            pairs.append((col_a, col_b))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _select_interaction_pairs(
    cat_cols: list[str],
    max_pairs: int = 10,
) -> list[tuple[str, str]]:
    """Select categorical column pairs for interaction features."""
    pairs = []
    for i, col_a in enumerate(cat_cols):
        for col_b in cat_cols[i + 1:]:
            pairs.append((col_a, col_b))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _filter_by_importance(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    new_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Train a quick LightGBM and drop engineered features with zero importance.

    Only filters the newly added features — original columns are always kept.
    """
    target_col = feature_types.get("target", ["ChurnStatus"])[0]
    id_col = feature_types.get("id", ["CustomerID"])[0]

    if target_col not in train.columns:
        return train, test, new_features, []

    # Prepare a quick dataset
    drop = [id_col, target_col] + feature_types.get("temporal", [])
    feature_cols = [c for c in train.columns if c not in drop]
    X = train[feature_cols].copy()

    # Encode any remaining object columns
    for col in X.columns:
        if X[col].dtype == object or not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        X[col] = X[col].fillna(X[col].median())

    y = (train[target_col].map({"Yes": 1, "No": 0, 1: 1, 0: 0})).astype(int)

    clf = LGBMClassifier(
        n_estimators=50, max_depth=4, verbosity=-1, random_state=42,
        importance_type="gain",
    )
    clf.fit(X, y)

    importances = dict(zip(X.columns, clf.feature_importances_))

    # Only filter engineered features; keep any with importance > 0
    kept = []
    dropped = []
    for f in new_features:
        if f in importances and importances[f] > 0:
            kept.append(f)
        else:
            dropped.append(f)
            if f in train.columns:
                train = train.drop(columns=[f])
            if f in test.columns:
                test = test.drop(columns=[f])

    return train, test, kept, dropped
