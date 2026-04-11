import numpy as np
import polars as pl
from polars_compat import to_numeric_coerce, is_numeric_col
from utils import logger


def _pairwise_corr_matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    """Compute pairwise-complete correlation matrix (matching pandas .corr() behavior).

    For each pair of columns, uses only rows where BOTH values are non-null.
    Returns an absolute correlation matrix as a numpy array.
    """
    n = len(cols)
    corr = np.eye(n)
    # Extract columns as numpy arrays (float64)
    arrays = {}
    for c in cols:
        arrays[c] = df[c].cast(pl.Float64, strict=False).to_numpy()

    for i in range(n):
        a = arrays[cols[i]]
        for j in range(i + 1, n):
            b = arrays[cols[j]]
            # Pairwise complete: use only rows where both are non-NaN
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 3:
                corr[i, j] = corr[j, i] = 0.0
                continue
            a_valid, b_valid = a[mask], b[mask]
            a_std = a_valid.std()
            b_std = b_valid.std()
            if a_std == 0 or b_std == 0:
                corr[i, j] = corr[j, i] = 0.0
                continue
            r = np.corrcoef(a_valid, b_valid)[0, 1]
            corr[i, j] = corr[j, i] = abs(r) if not np.isnan(r) else 0.0
    return corr


def engineer_features(
    train: pl.DataFrame,
    test: pl.DataFrame,
    feature_types: dict[str, list[str]],
    drop_cols: list[str] | None = None,
) -> tuple:
    """Generate dynamic features. No hardcoded column names.

    Features added:
    1. Missing value indicators for columns with >1% nulls
    2. Frequency encoding for high-cardinality categoricals (>10 unique)
    3. Ratio features between correlated numerical pairs

    Returns (train_enriched, test_enriched, new_feature_names) with aligned columns.
    """

    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    if drop_cols:
        skip_cols.update(drop_cols)
    new_features = []
    existing_cols = set(train.columns) | set(test.columns)  # guard against name collisions

    # 1. Missing value indicators
    _null_train_parts = {}
    _null_test_parts = {}
    for col in train.columns:
        if col in skip_cols:
            continue
        null_pct = train[col].null_count() / len(train)
        if null_pct > 0.01 and col in test.columns:
            fname = f"{col}_is_null"
            if fname in existing_cols:
                continue
            _null_train_parts[fname] = train[col].is_null().cast(pl.Int32)
            _null_test_parts[fname] = test[col].is_null().cast(pl.Int32)
            new_features.append(fname)
            existing_cols.add(fname)

    if _null_train_parts:
        train = train.hstack(pl.DataFrame(_null_train_parts))
        test = test.hstack(pl.DataFrame(_null_test_parts))
        logger.info(f"Added {len(new_features)} missing indicator features")

    # 1b. Datetime feature extraction
    _DATE_FORMATS = [
        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%Y%m%d", "%b %d, %Y", "%d %b %Y",
    ]
    dt_cols = [c for c in feature_types.get("datetime", [])
               if c not in skip_cols and c in train.columns and c in test.columns]
    dt_features_added = 0
    for col in dt_cols:
        # Parse to datetime
        parsed_train = None
        parsed_test = None
        raw_train = train[col].cast(pl.Utf8)
        raw_test = test[col].cast(pl.Utf8)
        for fmt in _DATE_FORMATS:
            try:
                pt = raw_train.str.to_datetime(fmt, strict=False)
                if pt.is_not_null().sum() > len(train) * 0.5:
                    parsed_train = pt
                    parsed_test = raw_test.str.to_datetime(fmt, strict=False)
                    break
            except Exception:
                continue
        if parsed_train is None:
            continue

        # Find reference date from both datasets for days_since
        all_min = min(
            parsed_train.drop_nulls().min() or parsed_test.drop_nulls().min(),
            parsed_test.drop_nulls().min() or parsed_train.drop_nulls().min(),
        )

        # Extract features
        dt_parts_train = {}
        dt_parts_test = {}
        for name, expr_train, expr_test in [
            (f"{col}_month", parsed_train.dt.month(), parsed_test.dt.month()),
            (f"{col}_dayofweek", parsed_train.dt.weekday(), parsed_test.dt.weekday()),
            (f"{col}_day", parsed_train.dt.day(), parsed_test.dt.day()),
            (f"{col}_days_since", (parsed_train - all_min).dt.total_days(), (parsed_test - all_min).dt.total_days()),
        ]:
            if name not in existing_cols:
                dt_parts_train[name] = expr_train.cast(pl.Float64).fill_null(0)
                dt_parts_test[name] = expr_test.cast(pl.Float64).fill_null(0)
                new_features.append(name)
                existing_cols.add(name)
                dt_features_added += 1

        if dt_parts_train:
            train = train.hstack(pl.DataFrame(dt_parts_train))
            test = test.hstack(pl.DataFrame(dt_parts_test))
            # Drop the original string column
            train = train.drop(col)
            test = test.drop(col)

    if dt_features_added:
        logger.info(f"Added {dt_features_added} datetime features from {len(dt_cols)} date columns")

    # 2. Frequency encoding for high-cardinality categoricals
    cat_cols = [c for c in feature_types.get("categorical", [])
                if c not in skip_cols and c in train.columns and c in test.columns]
    freq_features = []
    _freq_train_parts = {}
    _freq_test_parts = {}
    for col in cat_cols:
        n_unique = train[col].n_unique()
        if n_unique > 10:
            vc = train[col].value_counts()
            freq_map = dict(zip(vc[col].to_list(), (vc["count"] / len(train)).to_list()))
            fname = f"{col}_freq"
            if fname in existing_cols:
                continue
            _freq_train_parts[fname] = train[col].replace_strict(freq_map, default=0.0, return_dtype=pl.Float64).fill_null(0.0)
            _freq_test_parts[fname] = test[col].replace_strict(freq_map, default=0.0, return_dtype=pl.Float64).fill_null(0.0)
            freq_features.append(fname)
            existing_cols.add(fname)

    if _freq_train_parts:
        train = train.hstack(pl.DataFrame(_freq_train_parts))
        test = test.hstack(pl.DataFrame(_freq_test_parts))
        logger.info(f"Added {len(freq_features)} frequency encoding features: {freq_features}")

    # 3. Ratio features between correlated numerical pairs
    num_cols = [c for c in feature_types.get("numerical", [])
                if c not in skip_cols and c in train.columns and c in test.columns]

    ratio_features = []
    if len(num_cols) >= 2:
        _MAX_CORR_COLS = 50
        corr_cols = num_cols
        if len(num_cols) > _MAX_CORR_COLS:
            variances = {}
            for c in num_cols:
                v = to_numeric_coerce(train[c]).var()
                variances[c] = v if v is not None else 0.0
            corr_cols = sorted(variances, key=variances.get, reverse=True)[:_MAX_CORR_COLS]
            logger.info(f"Ratio features: using top {_MAX_CORR_COLS}/{len(num_cols)} cols by variance for correlation")

        # Build numeric data for correlation (pairwise-complete, matching pandas .corr())
        corr_train = train.select([to_numeric_coerce(train[c]).alias(c) for c in corr_cols])
        if len(corr_train) > 100_000:
            corr_train = corr_train.sample(n=100_000, seed=42)
        corr_matrix = _pairwise_corr_matrix(corr_train, corr_cols)
        col_idx = {c: i for i, c in enumerate(corr_cols)}

        pairs = []
        for i, col_a in enumerate(corr_cols):
            for col_b in corr_cols[i + 1:]:
                c = corr_matrix[col_idx[col_a], col_idx[col_b]]
                if c > 0.3:
                    pairs.append((col_a, col_b, c))

        pairs.sort(key=lambda x: x[2], reverse=True)
        _ratio_train_parts = {}
        _ratio_test_parts = {}
        for col_a, col_b, _ in pairs[:5]:
            fname = f"{col_a}_div_{col_b}"
            if fname in existing_cols:
                continue
            a_train = to_numeric_coerce(train[col_a]).fill_null(0)
            b_train = to_numeric_coerce(train[col_b]).fill_null(0)
            a_test = to_numeric_coerce(test[col_a]).fill_null(0)
            b_test = to_numeric_coerce(test[col_b]).fill_null(0)
            _ratio_train_parts[fname] = (a_train / (b_train + 1)).alias(fname)
            _ratio_test_parts[fname] = (a_test / (b_test + 1)).alias(fname)
            ratio_features.append(fname)
            existing_cols.add(fname)

        if _ratio_train_parts:
            train = train.hstack(pl.DataFrame(list(_ratio_train_parts.values())))
            test = test.hstack(pl.DataFrame(list(_ratio_test_parts.values())))
            logger.info(f"Added {len(ratio_features)} ratio features: {ratio_features}")

    # 4. Aggregation features: group-level stats (mean) per categorical
    agg_cat_cols = [c for c in cat_cols if 3 <= train[c].n_unique() <= 50]
    agg_num_cols = num_cols[:10]  # limit to avoid feature explosion

    agg_features = []
    _agg_train_parts = {}
    _agg_test_parts = {}
    if agg_cat_cols and agg_num_cols:
        for cat_col in agg_cat_cols[:3]:  # top 3 categoricals by cardinality
            for num_col in agg_num_cols[:5]:  # top 5 numerical
                fname = f"{num_col}_mean_by_{cat_col}"
                if fname in existing_cols:
                    continue
                num_series = to_numeric_coerce(train[num_col])
                global_mean = num_series.mean()
                if global_mean is None:
                    global_mean = 0.0

                # Train: window function for group mean
                group_means = train.with_columns(
                    num_series.alias("__num_tmp")
                ).select(
                    pl.col("__num_tmp").mean().over(cat_col)
                ).to_series().fill_null(global_mean).alias(fname)
                _agg_train_parts[fname] = group_means

                # Test: map using training group means
                agg = train.with_columns(
                    num_series.alias(num_col)
                ).group_by(cat_col).agg(pl.col(num_col).mean())
                means_map = dict(zip(agg[cat_col].to_list(), agg[num_col].to_list()))
                # Replace None values in means_map with global_mean
                means_map = {k: (v if v is not None else global_mean) for k, v in means_map.items()}
                test_mapped = test[cat_col].replace_strict(means_map, default=global_mean, return_dtype=pl.Float64).alias(fname)
                _agg_test_parts[fname] = test_mapped
                agg_features.append(fname)
                existing_cols.add(fname)

    if _agg_train_parts:
        train = train.hstack(pl.DataFrame(list(_agg_train_parts.values())))
        test = test.hstack(pl.DataFrame(list(_agg_test_parts.values())))
        logger.info(f"Added {len(agg_features)} aggregation features")

    # Collect all new feature names
    all_new_features = new_features + freq_features + ratio_features + agg_features

    # 5. Duplicate column detection: remove near-duplicate features (|corr| > 0.98)
    try:
        num_feature_cols = [c for c in train.columns if c not in skip_cols
                           and c in test.columns and is_numeric_col(train[c])]
        # Cap columns for duplicate detection to avoid O(m^2) correlation matrix
        _MAX_DEDUP_COLS = 200
        if len(num_feature_cols) > _MAX_DEDUP_COLS:
            _orig_dedup = len(num_feature_cols)
            _var_sample = train.select(num_feature_cols)
            if len(train) > 100_000:
                _var_sample = _var_sample.sample(n=100_000, seed=42)
            # Compute variance per column and pick top ones
            var_dict = {}
            for c in num_feature_cols:
                v = _var_sample[c].cast(pl.Float64, strict=False).var()
                var_dict[c] = v if v is not None else 0.0
            num_feature_cols = sorted(var_dict, key=var_dict.get, reverse=True)[:_MAX_DEDUP_COLS]
            logger.info(f"Duplicate detection: using top {_MAX_DEDUP_COLS}/{_orig_dedup} cols by variance")
        if len(num_feature_cols) >= 2:
            # Subsample rows for correlation on large datasets
            corr_data = train.select(num_feature_cols)
            if len(corr_data) > 100_000:
                corr_data = corr_data.sample(n=100_000, seed=42)
            corr = _pairwise_corr_matrix(corr_data, num_feature_cols)
            col_idx = {c: i for i, c in enumerate(num_feature_cols)}
            # Find pairs with correlation > 0.98
            drop_dupes = set()
            engineered_set = set(all_new_features)
            for i, col_a in enumerate(num_feature_cols):
                if col_a in drop_dupes:
                    continue
                for col_b in num_feature_cols[i + 1:]:
                    if col_b in drop_dupes:
                        continue
                    if corr[col_idx[col_a], col_idx[col_b]] > 0.98:
                        # Prefer dropping the engineered feature over original
                        if col_b in engineered_set and col_a not in engineered_set:
                            drop_dupes.add(col_b)
                        elif col_a in engineered_set and col_b not in engineered_set:
                            drop_dupes.add(col_a)
                        else:
                            # Both engineered or both original — drop the second
                            drop_dupes.add(col_b)
            if drop_dupes:
                train = train.drop([c for c in drop_dupes if c in train.columns])
                test = test.drop([c for c in drop_dupes if c in test.columns])
                all_new_features = [f for f in all_new_features if f not in drop_dupes]
                logger.info(f"Dropped {len(drop_dupes)} near-duplicate features: {sorted(drop_dupes)}")
    except Exception as e:
        logger.warning(f"Duplicate column detection skipped: {e}")

    # Ensure train and test have identical feature columns
    # Preserve train-only columns (target, ID) that aren't expected in test
    common_cols = sorted(set(train.columns) & set(test.columns))
    train_only_keep = [c for c in train.columns if c in skip_cols and c not in test.columns]
    train = train.select(common_cols + train_only_keep)
    test = test.select(common_cols)

    # Filter to only new features that survived column alignment
    all_new_features = [f for f in all_new_features if f in common_cols]

    return train, test, all_new_features
