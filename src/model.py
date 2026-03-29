import numpy as np
import polars as pl
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import joblib
from utils import logger
from polars_compat import (
    to_numeric_coerce, is_numeric_col, to_pandas_for_lgbm,
    ensure_pl_df as _ensure_pl_df, ensure_pl_series as _ensure_pl_series,
)


def _encode_target(series) -> pl.Series:
    """Auto-detect and encode binary target to 0/1.

    Handles: "Yes"/"No", 1/0, True/False, "1"/"0", "yes"/"no", etc.
    Accepts pandas Series or polars Series; always returns polars Series.
    """
    series = _ensure_pl_series(series)

    if series.is_null().all():
        return series.fill_null(0).cast(pl.Int64)

    vals = series.drop_nulls().unique()

    # Already numeric 0/1
    if is_numeric_col(series):
        vals_set = set(vals.to_list())
        if vals_set.issubset({0, 1, 0.0, 1.0}):
            return series.fill_null(0).cast(pl.Int64)

    # Boolean
    if series.dtype == pl.Boolean:
        return series.cast(pl.Int64)

    # String/object: map common positive labels to 1
    positive_labels = {"yes", "true", "1", "churn", "churned", "positive"}
    str_vals = {str(v).lower().strip() for v in vals.to_list()}

    if str_vals & positive_labels:
        pos = list(str_vals & positive_labels)
        target = series.cast(pl.Utf8).str.to_lowercase().str.strip_chars()
        result = target.to_frame("_t").select(
            pl.when(pl.col("_t").is_in(pos)).then(1).otherwise(0).cast(pl.Int64).alias("_t")
        ).to_series()
        return result.rename(series.name)

    # Fallback for binary: assume minority class is positive
    vc = series.value_counts()
    if len(vc) == 2:
        # value_counts returns a DataFrame with columns [series_name, "count"]
        col_name = vc.columns[0]
        minority_row = vc.sort("count").row(0)
        minority = minority_row[0]
        logger.warning(f"Auto-encoding target: '{minority}' -> 1 (minority class)")
        result = series.to_frame("_t").select(
            pl.when(pl.col("_t") == minority).then(1).otherwise(0).cast(pl.Int64).alias("_t")
        ).to_series()
        return result.rename(series.name)

    # Last resort: try numeric conversion
    return to_numeric_coerce(series).fill_null(0).cast(pl.Int64)



def prepare_features(
    train,
    test,
    feature_types: dict[str, list[str]],
    extra_drop_cols: list[str] | None = None,
) -> tuple:
    """Minimal preprocessing: encode categoricals, drop IDs, split X/y.

    Accepts pandas or polars DataFrames. Returns pandas if input was pandas
    (for compatibility with main.py), polars otherwise.

    Returns (X_train, y_train, X_test, customer_ids).
    """
    import pandas as pd
    _input_was_pandas = isinstance(train, pd.DataFrame)
    train = _ensure_pl_df(train)
    test = _ensure_pl_df(test)

    target_col = feature_types["target"][0] if feature_types["target"] else None
    id_col = feature_types["id"][0] if feature_types["id"] else None

    # Extract customer IDs from test
    if id_col and id_col in test.columns:
        customer_ids = test[id_col].to_list()
    else:
        customer_ids = list(range(len(test)))
        logger.warning("ID column not found in test; using row indices")

    # Encode target
    if target_col and target_col in train.columns:
        y_train = _encode_target(train[target_col])
    else:
        logger.warning("No target column found; using zeros")
        y_train = pl.Series("target", np.zeros(len(train), dtype=np.int64))

    # Columns to use as features (must be in both train and test)
    # Temporal columns are timestamps (e.g. "25-Jan"), not predictive features —
    # their signal is captured via recency sample weights instead.
    drop_cols = feature_types["id"] + feature_types["target"] + feature_types.get("temporal", [])
    if extra_drop_cols:
        drop_cols = drop_cols + [c for c in extra_drop_cols if c not in drop_cols]
    feature_cols = [c for c in train.columns if c not in drop_cols and c in test.columns]

    X_train = train.select(feature_cols)
    X_test = test.select(feature_cols)

    # Coerce columns classified as numerical but stored as non-numeric (e.g., "TotalCharges" with blanks)
    for col in feature_types["numerical"]:
        if col in feature_cols and not is_numeric_col(X_train[col]):
            X_train = X_train.with_columns(to_numeric_coerce(X_train[col]).alias(col))
            X_test = X_test.with_columns(to_numeric_coerce(X_test[col]).alias(col))

    # Coerce mismatched dtypes between train and test
    for col in feature_cols:
        if X_train[col].dtype != X_test[col].dtype:
            if is_numeric_col(X_train[col]) or is_numeric_col(X_test[col]):
                X_train = X_train.with_columns(to_numeric_coerce(X_train[col]).alias(col))
                X_test = X_test.with_columns(to_numeric_coerce(X_test[col]).alias(col))

    # Target encoding for categorical columns (k-fold to avoid leakage)
    # High-cardinality categoricals (>10 unique) use LightGBM native categorical splits
    cat_cols = [c for c in feature_cols if X_train[c].dtype in (pl.Utf8, pl.String)]
    global_mean = float(y_train.mean())
    from utils import safe_n_splits
    _single_class = y_train.n_unique() < 2
    if _single_class:
        logger.warning("Single-class target detected — using global mean for all target encoding")
    kf = StratifiedKFold(n_splits=safe_n_splits(len(X_train)), shuffle=True, random_state=42) if not _single_class else None

    smoothing_m = 10  # Bayesian smoothing: shrink low-count category means toward global mean

    # Split categoricals by cardinality
    native_cat_cols = []  # high-cardinality -> LightGBM native
    te_cat_cols = []      # low-cardinality -> target encoding
    for col in cat_cols:
        n_unique = X_train[col].n_unique()
        if n_unique > 10:
            native_cat_cols.append(col)
        else:
            te_cat_cols.append(col)

    if native_cat_cols:
        logger.info(f"Native categorical encoding for {len(native_cat_cols)} high-cardinality cols: {native_cat_cols}")

    # Store native cat cols at module level so train_model can set category dtype
    global _native_cat_cols
    _native_cat_cols = list(native_cat_cols)

    # Native categorical: integer label encode (keep as Int64 for LightGBM compatibility)
    for col in native_cat_cols:
        try:
            raw_train = X_train[col].cast(pl.Utf8)
            raw_test = X_test[col].cast(pl.Utf8)
            # Build label mapping from train categories (exclude None)
            train_cats = [c for c in raw_train.drop_nulls().unique().to_list()]
            cat_to_int = {cat: i for i, cat in enumerate(train_cats)}
            # Encode train — nulls stay null (LightGBM handles natively)
            encoded_tr = raw_train.replace_strict(cat_to_int, default=None, return_dtype=pl.Int64)
            X_train = X_train.with_columns(encoded_tr.alias(col))
            # Encode test — unseen categories become null
            encoded_te = raw_test.replace_strict(cat_to_int, default=None, return_dtype=pl.Int64)
            X_test = X_test.with_columns(encoded_te.alias(col))
        except Exception as e:
            logger.warning(f"Native categorical encoding failed for {col}, falling back to target encoding: {e}")
            te_cat_cols.append(col)

    # Subsample for target encoding fold computation on large datasets
    _MAX_TE_SAMPLE = 200_000
    _te_use_sample = len(X_train) > _MAX_TE_SAMPLE
    _te_sample_idx = None
    if _te_use_sample:
        from utils import stratified_subsample_idx
        _te_sample_idx = stratified_subsample_idx(
            len(X_train), _MAX_TE_SAMPLE,
            month_values=None, seed=42,
        )
        logger.info(f"Target encoding: using stratified {_MAX_TE_SAMPLE}/{len(X_train)} sample for fold means")

    y_train_np = y_train.to_numpy()

    for col in te_cat_cols:
        raw_train = X_train[col].cast(pl.Utf8)
        raw_test = X_test[col].cast(pl.Utf8)

        # Single-class target: skip k-fold, use global mean for all categories
        if _single_class:
            X_train = X_train.with_columns(pl.lit(global_mean).alias(col))
            X_test = X_test.with_columns(pl.lit(global_mean).alias(col))
            continue

        # Smoothed means (computed on sample if large, full data otherwise)
        if _te_use_sample:
            sample_raw_list = raw_train.gather(_te_sample_idx.tolist()).to_list()
            sample_y_list = y_train_np[_te_sample_idx]
        else:
            sample_raw_list = raw_train.to_list()
            sample_y_list = y_train_np

        # Compute col_means using a temporary polars DataFrame
        _temp = pl.DataFrame({"raw": sample_raw_list, "y": sample_y_list})
        _agg = _temp.group_by("raw").agg([
            pl.col("y").count().alias("cnt"),
            pl.col("y").mean().alias("mn"),
        ])
        col_means = {}
        for row in _agg.iter_rows(named=True):
            cnt, mn = row["cnt"], row["mn"]
            col_means[row["raw"]] = (cnt * mn + smoothing_m * global_mean) / (cnt + smoothing_m)

        # K-fold target encoding for train (avoids leakage)
        encoded = np.full(len(X_train), np.nan)

        if _te_use_sample:
            # Compute fold means on subsample, map onto ALL rows
            n_folds = safe_n_splits(len(_te_sample_idx))
            fold_assignments = np.arange(len(X_train)) % n_folds

            raw_train_list = raw_train.to_list()

            for fold_id in range(n_folds):
                # Means from other folds' subsample rows
                other_mask = fold_assignments[_te_sample_idx] != fold_id
                fold_raw_list = [sample_raw_list[i] for i, m in enumerate(other_mask) if m]
                fold_y_list = sample_y_list[other_mask]

                _ft = pl.DataFrame({"raw": fold_raw_list, "y": fold_y_list})
                _fa = _ft.group_by("raw").agg([
                    pl.col("y").count().alias("cnt"),
                    pl.col("y").mean().alias("mn"),
                ])
                fold_means = {}
                for row in _fa.iter_rows(named=True):
                    fc, fm = row["cnt"], row["mn"]
                    fold_means[row["raw"]] = (fc * fm + smoothing_m * global_mean) / (fc + smoothing_m)

                # Map to all full-dataset rows in this fold
                this_fold_mask = fold_assignments == fold_id
                this_fold_indices = np.where(this_fold_mask)[0]
                mapped = [fold_means.get(raw_train_list[i], np.nan) for i in this_fold_indices]
                encoded[this_fold_indices] = mapped
        else:
            raw_train_list = raw_train.to_list()
            for train_idx, val_idx in kf.split(np.arange(len(X_train)), y_train_np):
                fold_raw_list = [raw_train_list[i] for i in train_idx]
                fold_y_arr = y_train_np[train_idx]

                _ft = pl.DataFrame({"raw": fold_raw_list, "y": fold_y_arr})
                _fa = _ft.group_by("raw").agg([
                    pl.col("y").count().alias("cnt"),
                    pl.col("y").mean().alias("mn"),
                ])
                fold_means = {}
                for row in _fa.iter_rows(named=True):
                    fc, fm = row["cnt"], row["mn"]
                    fold_means[row["raw"]] = (fc * fm + smoothing_m * global_mean) / (fc + smoothing_m)

                mapped = [fold_means.get(raw_train_list[i], np.nan) for i in val_idx]
                encoded[val_idx] = mapped

        # Fill NaN with global mean
        nan_mask = np.isnan(encoded)
        encoded[nan_mask] = global_mean
        X_train = X_train.with_columns(pl.Series(col, encoded))

        # Test: use smoothed means
        test_raw_list = raw_test.to_list()
        test_encoded = np.array([col_means.get(v, global_mean) for v in test_raw_list])
        X_test = X_test.with_columns(pl.Series(col, test_encoded))

    # Fill remaining NaN: let LightGBM handle numeric NaN natively (learns optimal
    # split direction), only fill non-numeric columns with mode
    for col in X_train.columns:
        if X_train[col].is_null().any() or X_test[col].is_null().any():
            if not is_numeric_col(X_train[col]):
                mode_vals = X_train[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 0
                X_train = X_train.with_columns(X_train[col].fill_null(fill_val).alias(col))
                X_test = X_test.with_columns(X_test[col].fill_null(fill_val).alias(col))

    logger.info(f"Features prepared: {X_train.shape[1]} columns")

    if _input_was_pandas:
        return X_train.to_pandas(), y_train.to_pandas(), X_test.to_pandas(), customer_ids
    return X_train, y_train, X_test, customer_ids


def _to_pandas_with_categoricals(df: pl.DataFrame, cat_cols: list[str] | None = None):
    """Convert polars DataFrame to pandas, preserving categorical dtype for LightGBM."""
    import pandas as pd
    pdf = to_pandas_for_lgbm(df)
    if cat_cols:
        for col in cat_cols:
            if col in pdf.columns:
                pdf[col] = pdf[col].astype("category")
    return pdf


# Module-level storage for categorical columns detected during prepare_features
_native_cat_cols: list[str] = []


def train_model(
    X_train,
    y_train,
    sample_weight: np.ndarray | None = None,
) -> LGBMClassifier:
    """Train LightGBM with fixed hyperparameters. (DO NOT TOUCH!!!)"""
    model = LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type="gain",
    )
    # LightGBM needs pandas/numpy — convert polars if needed
    import pandas as pd
    if isinstance(X_train, pl.DataFrame):
        X_fit = _to_pandas_with_categoricals(X_train, _native_cat_cols)
    elif isinstance(X_train, pd.DataFrame):
        X_fit = X_train
    else:
        X_fit = X_train

    if isinstance(y_train, pl.Series):
        y_fit = y_train.to_numpy()
    elif isinstance(y_train, pd.Series):
        y_fit = y_train.to_numpy()
    else:
        y_fit = y_train

    model.fit(X_fit, y_fit, sample_weight=sample_weight)
    tag = "with sample weights" if sample_weight is not None else "uniform weights"
    logger.info(f"Model trained ({tag})")
    return model


def evaluate_and_save(
    model: LGBMClassifier,
    X_train,
    y_train,
    X_test,
    customer_ids: list[str],
    test,
    feature_types: dict[str, list[str]],
    tag: str = "",
) -> dict[str, float]:
    """Predict, compute AU-PRC, save model and predictions.

    Accepts pandas or polars DataFrames for X_train/X_test/test and
    pandas or polars Series for y_train.
    """
    import pandas as pd
    suffix = f"_{tag}" if tag else ""

    # Convert for LightGBM predict_proba (preserve category dtypes)
    if isinstance(X_train, pl.DataFrame):
        X_train_lgbm = _to_pandas_with_categoricals(X_train, _native_cat_cols)
    elif isinstance(X_train, pd.DataFrame):
        X_train_lgbm = X_train
    else:
        X_train_lgbm = X_train

    if isinstance(X_test, pl.DataFrame):
        X_test_lgbm = _to_pandas_with_categoricals(X_test, _native_cat_cols)
    elif isinstance(X_test, pd.DataFrame):
        X_test_lgbm = X_test
    else:
        X_test_lgbm = X_test

    # Convert y_train for sklearn metrics
    if isinstance(y_train, pl.Series):
        y_train_np = y_train.to_numpy()
    elif isinstance(y_train, pd.Series):
        y_train_np = y_train.to_numpy()
    else:
        y_train_np = np.asarray(y_train)

    # Ensure test is accessible (convert to polars for uniform column access)
    test_pl = _ensure_pl_df(test)

    # Train AU-PRC
    train_proba = model.predict_proba(X_train_lgbm)[:, 1]
    train_auprc = average_precision_score(y_train_np, train_proba)

    # Test AU-PRC (only if target column exists in test)
    target_col = feature_types["target"][0] if feature_types["target"] else None
    test_proba = model.predict_proba(X_test_lgbm)[:, 1]
    test_auprc = None
    if target_col and target_col in test_pl.columns and test_pl[target_col].is_not_null().any():
        # Guard against dummy test labels (all same value) in hidden eval
        if test_pl[target_col].drop_nulls().n_unique() >= 2:
            y_test = _encode_target(test_pl[target_col])
            test_auprc = average_precision_score(y_test.to_numpy(), test_proba)

    # Save prediction and model
    id_col = feature_types["id"][0] if feature_types["id"] else "ID"
    pred_df = pl.DataFrame({id_col: customer_ids, "probability_score": test_proba})
    pred_df.write_csv(f"prediction{suffix}.csv")
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
    """Print model performance table comparing baseline vs mitigated."""
    CW_metric = 18
    CW_val = 16
    n_models = len(model_results)
    # Column widths: metric + one value column per model
    total = CW_metric + CW_val * n_models + n_models
    title = "MODEL PERFORMANCE SUMMARY"

    metric_rows = [
        ("AU-PRC (train)", "train_auprc"),
        ("AU-PRC (test)", "test_auprc"),
    ]

    def h_line(left, right):
        line = left + "═" * CW_metric
        for _ in range(n_models):
            line += "╬" + "═" * CW_val
        line += right
        return line

    print()
    print(f"╔{'═' * total}╗")
    print(f"║{title:^{total}}║")
    print(h_line("╠", "╣"))
    # Header row
    header = f"║ {'Metric':<{CW_metric-2}} "
    for name, _ in model_results:
        header += f"║ {name:<{CW_val-2}} "
    header += "║"
    print(header)
    print(h_line("╠", "╣"))
    # Data rows
    for label, key in metric_rows:
        row = f"║ {label:<{CW_metric-2}} "
        for _, metrics in model_results:
            v = metrics.get(key)
            row += f"║ {v:<{CW_val-2}.4f} " if v is not None else f"║ {'N/A':<{CW_val-2}} "
        row += "║"
        print(row)
    print(h_line("╚", "╝"))
