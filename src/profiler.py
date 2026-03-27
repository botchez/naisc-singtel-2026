import polars as pl
import numpy as np
from utils import logger
from polars_compat import is_numeric_col, to_numeric_coerce


def _downcast_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast numeric columns to reduce memory usage.

    float64 -> float32, int64 -> int32/int16/int8 where safe.
    Essential for handling 10M+ row datasets within memory limits.
    """
    cast_exprs = []
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == pl.Float64:
            cast_exprs.append(pl.col(col).cast(pl.Float32))
        elif col_dtype == pl.Int64:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min is not None and col_max is not None:
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    cast_exprs.append(pl.col(col).cast(pl.Int8))
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    cast_exprs.append(pl.col(col).cast(pl.Int16))
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    cast_exprs.append(pl.col(col).cast(pl.Int32))
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df


def load_data(train_path: str, test_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load train and test CSVs with memory optimization.

    Applies dtype downcasting to reduce memory footprint for large datasets
    (e.g., 10M rows x 500 features).
    """
    train = pl.read_csv(train_path, infer_schema_length=10000)
    test = pl.read_csv(test_path, infer_schema_length=10000)

    # Polars rejects duplicate column names on read, but handle edge cases
    # where columns were auto-renamed with _duplicated_N suffix
    for df_ref, label in [(train, "train"), (test, "test")]:
        dup_cols = [c for c in df_ref.columns if "_duplicated_" in c]
        if dup_cols:
            logger.warning(f"Found {len(dup_cols)} auto-renamed duplicate columns in {label}: {dup_cols}")

    mem_before = (train.estimated_size() + test.estimated_size()) / 1e6
    train = _downcast_dataframe(train)
    test = _downcast_dataframe(test)
    mem_after = (train.estimated_size() + test.estimated_size()) / 1e6

    logger.info(f"Loaded train: {train.shape}, test: {test.shape} "
                f"(memory: {mem_before:.1f}MB -> {mem_after:.1f}MB)")
    return train, test


def _normalize_col_name(name: str) -> str:
    """Normalize column name for fuzzy comparison."""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")



def align_train_test_columns(
    train: pl.DataFrame,
    test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Align columns between train and test using fuzzy name matching.

    Renames test columns to match train column names when normalized forms match.
    Logs warnings for columns present in only one dataset.
    """
    train_cols = set(train.columns)
    test_cols = set(test.columns)

    common_exact = train_cols & test_cols
    train_only = train_cols - common_exact
    test_only = test_cols - common_exact

    rename_map = {}
    if train_only and test_only:
        train_norm = {_normalize_col_name(c): c for c in train_only}
        for test_col in list(test_only):
            norm = _normalize_col_name(test_col)
            if norm in train_norm:
                rename_map[test_col] = train_norm[norm]

    test_aligned = test.rename(rename_map) if rename_map else test

    if rename_map:
        logger.info(f"Fuzzy-matched columns (test->train): {rename_map}")

    matched_train = set(rename_map.values())
    matched_test = set(rename_map.keys())
    remaining_train_only = train_only - matched_train
    remaining_test_only = test_only - matched_test

    if remaining_train_only:
        logger.warning(f"Columns only in train: {sorted(remaining_train_only)}")
    if remaining_test_only:
        logger.warning(f"Columns only in test: {sorted(remaining_test_only)}")

    return train, test_aligned


def detect_feature_types(
    df: pl.DataFrame,
    id_col: str = "CustomerID",
    target_col: str = "ChurnStatus",
    temporal_col: str = "Month",
    test_df: pl.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Detect feature types. Columns id_col/target_col/temporal_col are
    matched by exact name. When test_df is provided, only classifies
    columns present in both datasets (except id/target).
    """
    if id_col not in df.columns:
        logger.warning(f"ID column '{id_col}' not found")
        id_col = None
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found")
        target_col = None
    if temporal_col not in df.columns:
        logger.info(f"Temporal column '{temporal_col}' not found")
        temporal_col = None

    # Determine which columns to classify as features (must be in both datasets)
    cols_to_classify = set(df.columns)
    if test_df is not None:
        common = set(df.columns) & set(test_df.columns)
        train_only_features = set(df.columns) - common
        # ID/target may be train-only (target often absent from test)
        if id_col:
            train_only_features.discard(id_col)
        if target_col:
            train_only_features.discard(target_col)
        if train_only_features:
            logger.warning(
                f"Skipping {len(train_only_features)} train-only columns "
                f"(not in test): {sorted(train_only_features)}"
            )
        cols_to_classify = common | ({id_col} if id_col and id_col in df.columns else set()) \
                                  | ({target_col} if target_col and target_col in df.columns else set())

    types: dict[str, list[str]] = {
        "numerical": [],
        "categorical": [],
        "temporal": [],
        "id": [],
        "target": [],
    }

    # Subsample for expensive per-column stats on large datasets
    _PROFILE_SAMPLE_SIZE = 100_000
    if len(df) > _PROFILE_SAMPLE_SIZE:
        df_sample = df.sample(n=_PROFILE_SAMPLE_SIZE, seed=42)
        logger.info(f"Feature type detection: using {_PROFILE_SAMPLE_SIZE}/{len(df)} row sample")
    else:
        df_sample = df

    for col in df.columns:
        if col not in cols_to_classify:
            continue

        if col == id_col:
            types["id"].append(col)
            continue
        if col == target_col:
            types["target"].append(col)
            continue

        # Skip all-null columns
        if df_sample[col].is_null().all():
            logger.warning(f"Skipping all-null column: '{col}'")
            continue

        # Skip zero-variance columns (only 1 unique non-null value)
        if df_sample[col].n_unique() <= 1:
            logger.warning(f"Skipping zero-variance column: '{col}'")
            continue

        # Skip near-zero-variance columns (>99% same value) -- non-informative
        try:
            vc = df_sample[col].value_counts().sort("count", descending=True)
            top_freq = vc["count"][0] / len(df_sample)
            if top_freq > 0.99:
                logger.warning(f"Skipping near-zero-variance column: '{col}' ({top_freq:.1%} dominant)")
                continue
        except Exception:
            pass

        # Known temporal column (exact match)
        if temporal_col and col == temporal_col:
            types["temporal"].append(col)
            continue

        # Numeric coercion: if a non-numeric column is mostly parseable as numbers,
        # treat it as numerical (e.g., TotalCharges with a few blank strings)
        if not is_numeric_col(df_sample[col]) and _looks_numeric(df_sample[col]):
            logger.info(f"Coercing '{col}' to numeric (mostly numeric values with unparseable entries)")
            types["numerical"].append(col)
            continue

        if is_numeric_col(df_sample[col]):
            types["numerical"].append(col)
        else:
            types["categorical"].append(col)

    return types


def _looks_numeric(series: pl.Series, threshold: float = 0.8) -> bool:
    """Heuristic: check if a non-numeric column is mostly parseable as numbers."""
    if len(series) > 100_000:
        series = series.sample(n=100_000, seed=42)
    coerced = to_numeric_coerce(series)
    parse_rate = coerced.is_not_null().sum() / max(len(series), 1)
    return parse_rate >= threshold




def profile_data(
    df: pl.DataFrame, feature_types: dict[str, list[str]], label: str = "dataset"
) -> pl.DataFrame:
    """Generate and print a profiling summary for the dataset."""
    _PROFILE_SAMPLE_SIZE = 100_000
    df_prof = df.sample(n=_PROFILE_SAMPLE_SIZE, seed=42) if len(df) > _PROFILE_SAMPLE_SIZE else df
    rows = []
    df_cols = set(df.columns)
    for category, cols in feature_types.items():
        for col in cols:
            if col not in df_cols:
                continue
            non_null = df[col].drop_nulls()
            sample_val = non_null[0] if len(non_null) > 0 else None
            rows.append(
                {
                    "column": col,
                    "type": category,
                    "dtype": str(df[col].dtype),
                    "n_unique": df_prof[col].n_unique(),
                    "null_pct": df_prof[col].null_count() / len(df_prof) * 100,
                    "sample_value": sample_val,
                }
            )
    profile = pl.DataFrame(rows)

    # Print table
    col_w = max(max((len(r["column"]) for r in rows), default=6), 6)
    type_w = max(max((len(r["type"]) for r in rows), default=4), 4)
    dtype_w = max(max((len(r["dtype"]) for r in rows), default=5), 5)
    sample_w = max(max((len(str(r["sample_value"])) for r in rows), default=6), 6)
    sample_w = min(sample_w, 30)  # cap width

    header = (f"  {'Column':<{col_w}}  {'Type':<{type_w}}  {'Dtype':<{dtype_w}}"
              f"  {'Uniq':>5}  {'Null%':>6}  {'Sample':<{sample_w}}")
    sep = "  " + "-" * (col_w + type_w + dtype_w + sample_w + 22)
    print(f"\n  [{label.upper()}] {len(rows)} columns "
          f"({len(feature_types['numerical'])} num, "
          f"{len(feature_types['categorical'])} cat, "
          f"{len(feature_types['temporal'])} temporal)")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        sv = str(r["sample_value"])
        if len(sv) > sample_w:
            sv = sv[:sample_w - 2] + ".."
        print(f"  {r['column']:<{col_w}}  {r['type']:<{type_w}}  {r['dtype']:<{dtype_w}}"
              f"  {r['n_unique']:>5}  {r['null_pct']:>5.1f}%  {sv:<{sample_w}}")
    print(sep)
    print()

    return profile
