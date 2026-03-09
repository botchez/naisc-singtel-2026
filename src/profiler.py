import pandas as pd
import numpy as np
from utils import logger


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info(f"Loaded train: {train.shape}, test: {test.shape}")
    return train, test


def _normalize_col_name(name: str) -> str:
    """Normalize column name for fuzzy comparison."""
    return name.lower().replace("_", "").replace(" ", "").replace("-", "")


_ID_PATTERNS = [
    "customerid", "custid", "userid", "accountid", "subscriberid",
    "clientid", "memberid", "patientid", "transactionid", "orderid",
    "recordid", "sampleid", "subjectid", "caseid",
]

_TARGET_PATTERNS = [
    "churnstatus", "churn", "target", "label", "class", "y",
    "ischurn", "churned", "outcome", "result", "status",
    "default", "fraud", "isfraud", "fraudulent",
    "diagnosis", "response", "flag",
]


def _auto_detect_id_column(df: pd.DataFrame) -> str | None:
    """Detect ID column by name patterns and uniqueness heuristic."""
    norm_map = {_normalize_col_name(c): c for c in df.columns}

    for pattern in _ID_PATTERNS:
        if pattern in norm_map:
            return norm_map[pattern]

    for norm, orig in norm_map.items():
        if norm.endswith("id") and df[orig].nunique() / max(len(df), 1) > 0.9:
            return orig

    for norm, orig in norm_map.items():
        if "id" in norm and df[orig].nunique() / max(len(df), 1) > 0.9:
            return orig

    return None


def _auto_detect_target_column(df: pd.DataFrame) -> str | None:
    """Detect target column by name patterns and binary heuristic."""
    norm_map = {_normalize_col_name(c): c for c in df.columns}

    for pattern in _TARGET_PATTERNS:
        if pattern in norm_map:
            return norm_map[pattern]

    # Fallback: look for binary columns (exactly 2 unique values) with suggestive names
    for norm, orig in norm_map.items():
        if any(kw in norm for kw in ("target", "label", "class", "outcome", "status", "flag", "response")):
            return orig

    return None


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

    test_aligned = test.rename(columns=rename_map) if rename_map else test.copy()

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

    return train.copy(), test_aligned


def detect_feature_types(
    df: pd.DataFrame,
    id_col: str = "CustomerID",
    target_col: str = "ChurnStatus",
    test_df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Auto-detect feature types with robust column detection.

    If id_col/target_col aren't found by exact name, tries fuzzy matching
    and heuristic detection. When test_df is provided, only classifies
    columns present in both datasets (except id/target).
    """
    # Auto-detect ID column if default not found
    if id_col not in df.columns:
        detected = _auto_detect_id_column(df)
        if detected:
            logger.info(f"Auto-detected ID column: '{detected}' ('{id_col}' not found)")
            id_col = detected
        else:
            logger.warning(f"ID column '{id_col}' not found and could not be auto-detected")
            id_col = None

    # Auto-detect target column if default not found
    if target_col not in df.columns:
        detected = _auto_detect_target_column(df)
        if detected:
            logger.info(f"Auto-detected target column: '{detected}' ('{target_col}' not found)")
            target_col = detected
        else:
            logger.warning(f"Target column '{target_col}' not found and could not be auto-detected")
            target_col = None

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
        if df[col].isna().all():
            logger.warning(f"Skipping all-null column: '{col}'")
            continue

        # Skip zero-variance columns (only 1 unique non-null value)
        if df[col].nunique() <= 1:
            logger.warning(f"Skipping zero-variance column: '{col}'")
            continue

        # Skip near-zero-variance columns (>99% same value) — non-informative
        try:
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            if top_freq > 0.99:
                logger.warning(f"Skipping near-zero-variance column: '{col}' ({top_freq:.1%} dominant)")
                continue
        except Exception:
            pass

        # Temporal heuristic: object column with month-year patterns like "25-Jan"
        if not pd.api.types.is_numeric_dtype(df[col]) and _looks_temporal(df[col]):
            types["temporal"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            types["numerical"].append(col)
        else:
            types["categorical"].append(col)

    return types


def _looks_temporal(series: pd.Series, sample_size: int = 50) -> bool:
    """Heuristic: check if string values look like month-year formats."""
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    formats = ["%y-%b", "%b-%y", "%Y-%m", "%m-%Y", "%b %Y", "%Y-%b",
               "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
    for fmt in formats:
        try:
            pd.to_datetime(sample, format=fmt)
            return True
        except (ValueError, TypeError):
            pass
    return False


def profile_data(
    df: pd.DataFrame, feature_types: dict[str, list[str]], label: str = "dataset"
) -> pd.DataFrame:
    """Generate a profiling summary for the dataset."""
    rows = []
    for category, cols in feature_types.items():
        for col in cols:
            rows.append(
                {
                    "column": col,
                    "type": category,
                    "dtype": str(df[col].dtype),
                    "n_unique": df[col].nunique(),
                    "null_pct": df[col].isna().mean() * 100,
                    "sample_value": df[col].dropna().iloc[0] if df[col].notna().any() else None,
                }
            )
    profile = pd.DataFrame(rows)
    logger.info(f"{label} profile: {len(profile)} columns "
                f"({len(feature_types['numerical'])} num, "
                f"{len(feature_types['categorical'])} cat, "
                f"{len(feature_types['temporal'])} temporal)")
    return profile
