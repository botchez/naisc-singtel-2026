import pandas as pd
import numpy as np
from utils import logger


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage.

    float64 -> float32, int64 -> int32/int16/int8 where safe.
    Essential for handling 10M+ row datasets within memory limits.
    """
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif col_dtype == np.int64:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    return df


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs with memory optimization.

    Applies dtype downcasting to reduce memory footprint for large datasets
    (e.g., 10M rows × 500 features).
    """
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    # Handle duplicate column names (suffix with _dup1, _dup2, etc.)
    for df, label in [(train, "train"), (test, "test")]:
        dupes = df.columns[df.columns.duplicated()]
        if len(dupes) > 0:
            new_cols = []
            seen = {}
            for c in df.columns:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_dup{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            df.columns = new_cols
            logger.warning(f"Renamed {len(dupes)} duplicate columns in {label}")

    mem_before = (train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 1e6
    train = _downcast_dataframe(train)
    test = _downcast_dataframe(test)
    mem_after = (train.memory_usage(deep=True).sum() + test.memory_usage(deep=True).sum()) / 1e6

    logger.info(f"Loaded train: {train.shape}, test: {test.shape} "
                f"(memory: {mem_before:.1f}MB -> {mem_after:.1f}MB)")
    return train, test


def _normalize_col_name(name: str) -> str:
    """Normalize column name for fuzzy comparison."""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")



def align_train_test_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    test_aligned = test.rename(columns=rename_map) if rename_map else test

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
    df: pd.DataFrame,
    id_col: str = "CustomerID",
    target_col: str = "ChurnStatus",
    temporal_col: str = "Month",
    test_df: pd.DataFrame | None = None,
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
        df_sample = df.sample(n=_PROFILE_SAMPLE_SIZE, random_state=42)
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
        if df_sample[col].isna().all():
            logger.warning(f"Skipping all-null column: '{col}'")
            continue

        # Skip zero-variance columns (only 1 unique non-null value)
        if df_sample[col].nunique() <= 1:
            logger.warning(f"Skipping zero-variance column: '{col}'")
            continue

        # Skip near-zero-variance columns (>99% same value) — non-informative
        try:
            top_freq = df_sample[col].value_counts(normalize=True).iloc[0]
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
        if not pd.api.types.is_numeric_dtype(df_sample[col]) and _looks_numeric(df_sample[col]):
            logger.info(f"Coercing '{col}' to numeric (mostly numeric values with unparseable entries)")
            types["numerical"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(df_sample[col]):
            types["numerical"].append(col)
        else:
            types["categorical"].append(col)

    return types


def _looks_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """Heuristic: check if a non-numeric column is mostly parseable as numbers."""
    if len(series) > 100_000:
        series = series.sample(n=100_000, random_state=42)
    coerced = pd.to_numeric(series, errors="coerce")
    parse_rate = coerced.notna().sum() / max(len(series), 1)
    return parse_rate >= threshold




def profile_data(
    df: pd.DataFrame, feature_types: dict[str, list[str]], label: str = "dataset"
) -> pd.DataFrame:
    """Generate and print a profiling summary for the dataset."""
    _PROFILE_SAMPLE_SIZE = 100_000
    df_prof = df.sample(n=_PROFILE_SAMPLE_SIZE, random_state=42) if len(df) > _PROFILE_SAMPLE_SIZE else df
    rows = []
    df_cols = set(df.columns)
    for category, cols in feature_types.items():
        for col in cols:
            if col not in df_cols:
                continue
            sample_val = df[col].dropna().iloc[0] if df[col].notna().any() else None
            rows.append(
                {
                    "column": col,
                    "type": category,
                    "dtype": str(df[col].dtype),
                    "n_unique": df_prof[col].nunique(),
                    "null_pct": df_prof[col].isna().mean() * 100,
                    "sample_value": sample_val,
                }
            )
    profile = pd.DataFrame(rows)

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
