import pandas as pd
import numpy as np
from utils import logger


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logger.info(f"Loaded train: {train.shape}, test: {test.shape}")
    return train, test


def detect_feature_types(
    df: pd.DataFrame,
    id_col: str = "CustomerID",
    target_col: str = "ChurnStatus",
) -> dict[str, list[str]]:
    """Auto-detect feature types: numerical, categorical, temporal, id.

    Returns a dict with keys: 'numerical', 'categorical', 'temporal', 'id', 'target'.
    """
    types: dict[str, list[str]] = {
        "numerical": [],
        "categorical": [],
        "temporal": [],
        "id": [],
        "target": [],
    }

    for col in df.columns:
        if col == id_col:
            types["id"].append(col)
            continue
        if col == target_col:
            types["target"].append(col)
            continue

        # Temporal heuristic: object column with month-year patterns like "25-Jan"
        if not pd.api.types.is_numeric_dtype(df[col]) and _looks_temporal(df[col]):
            types["temporal"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # High-cardinality numeric is truly numerical; low-cardinality might be categorical
            nunique = df[col].nunique()
            if nunique <= 2 and set(df[col].dropna().unique()).issubset({0, 1}):
                types["categorical"].append(col)
            else:
                types["numerical"].append(col)
        else:
            types["categorical"].append(col)

    return types


def _looks_temporal(series: pd.Series, sample_size: int = 50) -> bool:
    """Heuristic: check if string values look like month-year (e.g. '25-Jan')."""
    sample = series.dropna().head(sample_size)
    if len(sample) == 0:
        return False
    try:
        pd.to_datetime(sample, format="%y-%b")
        return True
    except (ValueError, TypeError):
        pass
    try:
        pd.to_datetime(sample, format="%b-%y")
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
