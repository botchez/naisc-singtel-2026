import numpy as np
import pandas as pd
from utils import logger


def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate dynamic features. No hardcoded column names.

    Features added:
    1. Missing value indicators for columns with >1% nulls
    2. Frequency encoding for high-cardinality categoricals (>10 unique)
    3. Ratio features between correlated numerical pairs

    Returns (train_enriched, test_enriched) with aligned columns.
    """
    train = train.copy()
    test = test.copy()
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    new_features = []

    # 1. Missing value indicators
    for col in train.columns:
        if col in skip_cols:
            continue
        null_pct = train[col].isna().mean()
        if null_pct > 0.01 and col in test.columns:
            fname = f"{col}_is_null"
            train[fname] = train[col].isna().astype(int)
            test[fname] = test[col].isna().astype(int)
            new_features.append(fname)

    if new_features:
        logger.info(f"Added {len(new_features)} missing indicator features")

    # 2. Frequency encoding for high-cardinality categoricals
    cat_cols = [c for c in feature_types.get("categorical", [])
                if c not in skip_cols and c in train.columns and c in test.columns]
    freq_features = []
    for col in cat_cols:
        n_unique = train[col].nunique()
        if n_unique > 10:
            # Frequency map from training data
            freq_map = train[col].value_counts(normalize=True).to_dict()
            fname = f"{col}_freq"
            train[fname] = train[col].map(freq_map).fillna(0.0)
            test[fname] = test[col].map(freq_map).fillna(0.0)
            freq_features.append(fname)

    if freq_features:
        logger.info(f"Added {len(freq_features)} frequency encoding features: {freq_features}")

    # 3. Ratio features between correlated numerical pairs
    num_cols = [c for c in feature_types.get("numerical", [])
                if c not in skip_cols and c in train.columns and c in test.columns]

    if len(num_cols) >= 2:
        # Compute correlation matrix on training data (numeric only)
        num_train = train[num_cols].apply(pd.to_numeric, errors="coerce")
        corr_matrix = num_train.corr().abs()

        # Find top correlated pairs (above threshold, not same column)
        pairs = []
        for i, col_a in enumerate(num_cols):
            for col_b in num_cols[i + 1:]:
                c = corr_matrix.loc[col_a, col_b]
                if pd.notna(c) and c > 0.3:
                    pairs.append((col_a, col_b, c))

        # Sort by correlation, take top 5
        pairs.sort(key=lambda x: x[2], reverse=True)
        ratio_features = []
        for col_a, col_b, _ in pairs[:5]:
            fname = f"{col_a}_div_{col_b}"
            a_train = pd.to_numeric(train[col_a], errors="coerce").fillna(0)
            b_train = pd.to_numeric(train[col_b], errors="coerce").fillna(0)
            a_test = pd.to_numeric(test[col_a], errors="coerce").fillna(0)
            b_test = pd.to_numeric(test[col_b], errors="coerce").fillna(0)
            train[fname] = a_train / (b_train + 1)
            test[fname] = a_test / (b_test + 1)
            ratio_features.append(fname)

        if ratio_features:
            logger.info(f"Added {len(ratio_features)} ratio features: {ratio_features}")

    # Ensure train and test have identical columns
    common_cols = sorted(set(train.columns) & set(test.columns))
    train = train[common_cols]
    test = test[common_cols]

    return train, test
