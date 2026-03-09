import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import joblib
from utils import logger


def _encode_target(series: pd.Series) -> pd.Series:
    """Auto-detect and encode binary target to 0/1.

    Handles: "Yes"/"No", 1/0, True/False, "1"/"0", "yes"/"no", etc.
    """
    if series.isna().all():
        return series.fillna(0).astype(int)

    vals = series.dropna().unique()

    # Already numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            return series.fillna(0).astype(int)

    # Boolean
    if series.dtype == bool:
        return series.astype(int)

    # String/object: map common positive labels to 1
    positive_labels = {"yes", "true", "1", "churn", "churned", "positive"}
    str_vals = {str(v).lower().strip() for v in vals}

    if str_vals & positive_labels:
        pos = str_vals & positive_labels
        return series.apply(lambda x: 1 if str(x).lower().strip() in pos else 0)

    # Fallback for binary: assume minority class is positive
    vc = series.value_counts()
    if len(vc) == 2:
        minority = vc.idxmin()
        logger.warning(f"Auto-encoding target: '{minority}' -> 1 (minority class)")
        return (series == minority).astype(int)

    # Last resort: try numeric conversion
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _parse_temporal_month(series: pd.Series) -> pd.Series:
    """Parse temporal column to month number, trying multiple formats."""
    # Title-case to restore month abbreviations after lowercase normalization
    s_title = series.astype(str).str.title()
    for fmt in ["%y-%b", "%b-%y", "%Y-%m", "%m-%Y", "%b %Y", "%Y-%b",
                "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]:
        parsed = pd.to_datetime(s_title, format=fmt, errors="coerce")
        if parsed.notna().sum() > len(series) * 0.5:
            return parsed.dt.month.fillna(0).astype(int)
    return pd.Series(np.zeros(len(series), dtype=int), index=series.index)


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    extra_drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Minimal preprocessing: encode categoricals, drop IDs, split X/y.

    Returns (X_train, y_train, X_test, customer_ids).
    """
    target_col = feature_types["target"][0] if feature_types["target"] else None
    id_col = feature_types["id"][0] if feature_types["id"] else None

    # Extract customer IDs from test
    if id_col and id_col in test.columns:
        customer_ids = test[id_col].tolist()
    else:
        customer_ids = list(range(len(test)))
        logger.warning("ID column not found in test; using row indices")

    # Encode target
    if target_col and target_col in train.columns:
        y_train = _encode_target(train[target_col])
    else:
        logger.warning("No target column found; using zeros")
        y_train = pd.Series(np.zeros(len(train), dtype=int))

    # Columns to use as features (must be in both train and test)
    drop_cols = feature_types["id"] + feature_types["target"]
    if extra_drop_cols:
        drop_cols = drop_cols + [c for c in extra_drop_cols if c not in drop_cols]
    feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # Coerce mismatched dtypes between train and test
    for col in feature_cols:
        if X_train[col].dtype != X_test[col].dtype:
            if pd.api.types.is_numeric_dtype(X_train[col]) or pd.api.types.is_numeric_dtype(X_test[col]):
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    # Convert temporal columns to month number (1-12)
    for col in feature_types["temporal"]:
        if col in feature_cols:
            X_train[col] = _parse_temporal_month(X_train[col])
            X_test[col] = _parse_temporal_month(X_test[col])

    # Target encoding for categorical columns (k-fold to avoid leakage)
    cat_cols = [c for c in feature_cols if X_train[c].dtype == object]
    global_mean = y_train.mean()
    from utils import safe_n_splits
    kf = StratifiedKFold(n_splits=safe_n_splits(len(X_train)), shuffle=True, random_state=42)

    smoothing_m = 10  # Bayesian smoothing: shrink low-count category means toward global mean

    for col in cat_cols:
        raw_train = X_train[col].astype(str)
        raw_test = X_test[col].astype(str)

        # Full training set smoothed means (for test encoding)
        col_counts = y_train.groupby(raw_train).count()
        col_means_raw = y_train.groupby(raw_train).mean()
        col_means = (col_counts * col_means_raw + smoothing_m * global_mean) / (col_counts + smoothing_m)

        # K-fold target encoding for train (avoids leakage)
        train_encoded = pd.Series(np.full(len(X_train), np.nan), index=X_train.index)
        for train_idx, val_idx in kf.split(X_train, y_train):
            fold_y = y_train.iloc[train_idx]
            fold_raw = raw_train.iloc[train_idx]
            fold_counts = fold_y.groupby(fold_raw).count()
            fold_means_raw = fold_y.groupby(fold_raw).mean()
            fold_means = (fold_counts * fold_means_raw + smoothing_m * global_mean) / (fold_counts + smoothing_m)
            train_encoded.iloc[val_idx] = raw_train.iloc[val_idx].map(fold_means)
        X_train[col] = train_encoded.fillna(global_mean)

        # Test: use full training set smoothed means
        X_test[col] = raw_test.map(col_means).fillna(global_mean)

    # Fill remaining NaN: let LightGBM handle numeric NaN natively (learns optimal
    # split direction), only fill non-numeric columns with mode
    for col in X_train.columns:
        if X_train[col].isna().any() or X_test[col].isna().any():
            if not pd.api.types.is_numeric_dtype(X_train[col]):
                mode = X_train[col].mode()
                fill_val = mode.iloc[0] if not mode.empty else 0
                X_train[col] = X_train[col].fillna(fill_val)
                X_test[col] = X_test[col].fillna(fill_val)

    logger.info(f"Features prepared: {X_train.shape[1]} columns")
    return X_train, y_train, X_test, customer_ids


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> LGBMClassifier:
    """Train LightGBM with fixed hyperparameters."""
    model = LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type="gain",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    tag = "with sample weights" if sample_weight is not None else "uniform weights"
    logger.info(f"Model trained ({tag})")
    return model


def evaluate_and_save(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    customer_ids: list[str],
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    tag: str = "",
) -> dict[str, float]:
    """Predict, compute AU-PRC, save model and predictions."""
    suffix = f"_{tag}" if tag else ""

    # Train AU-PRC
    train_proba = model.predict_proba(X_train)[:, 1]
    train_auprc = average_precision_score(y_train, train_proba)

    # Test AU-PRC (only if target column exists in test)
    target_col = feature_types["target"][0] if feature_types["target"] else None
    test_proba = model.predict_proba(X_test)[:, 1]
    test_auprc = None
    if target_col and target_col in test.columns and test[target_col].notna().any():
        y_test = _encode_target(test[target_col])
        test_auprc = average_precision_score(y_test, test_proba)

    # Save prediction and model
    id_col = feature_types["id"][0] if feature_types["id"] else "ID"
    pred_df = pd.DataFrame({id_col: customer_ids, "probability_score": test_proba})
    pred_df.to_csv(f"prediction{suffix}.csv", index=False)
    logger.info(f"Saved prediction{suffix}.csv")

    joblib.dump(model, f"model{suffix}.joblib")
    logger.info(f"Saved model{suffix}.joblib")

    metrics = {"train_auprc": train_auprc}
    if test_auprc is not None:
        metrics["test_auprc"] = test_auprc

    return metrics


def print_model_performance_summary(
    model_results: list[tuple[str, dict[str, float]]],
) -> None:
    """Print model performance table comparing all model variants with deltas."""
    CW = (18, 16, 16, 16)
    total = sum(CW) + len(CW) - 1
    title = "MODEL PERFORMANCE SUMMARY"

    def h(l, m, r):
        return f"{l}{'═'*CW[0]}{m}{'═'*CW[1]}{m}{'═'*CW[2]}{m}{'═'*CW[3]}{r}"

    def col_row(c1, c2, c3, c4):
        return f"║ {c1:<{CW[0]-2}} ║ {c2:<{CW[1]-2}} ║ {c3:<{CW[2]-2}} ║ {c4:<{CW[3]-2}} ║"

    # Extract baseline (first) and mitigated (second) results
    baseline = model_results[0][1] if len(model_results) > 0 else {}
    mitigated = model_results[1][1] if len(model_results) > 1 else {}

    metric_rows = [
        ("AU-PRC (train)", "train_auprc"),
        ("AU-PRC (test)", "test_auprc"),
    ]

    print()
    print(f"╔{'═' * total}╗")
    print(f"║{title:^{total}}║")
    print(h("╠", "╦", "╣"))
    print(col_row("Metric", "Baseline", "Mitigated", "Delta"))
    print(h("╠", "╬", "╣"))
    for label, key in metric_rows:
        b = baseline.get(key)
        m = mitigated.get(key)
        b_v = f"{b:.4f}" if b is not None else "N/A"
        m_v = f"{m:.4f}" if m is not None else "N/A"
        if b is not None and m is not None:
            delta_v = f"{m - b:+.4f}"
        else:
            delta_v = "N/A"
        print(col_row(label, b_v, m_v, delta_v))
    print(h("╚", "╩", "╝"))
