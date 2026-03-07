import numpy as np
import pandas as pd
from utils import logger


def compute_temporal_weights(
    train: pd.DataFrame,
    month_col: str,
    month_format: str = "%y-%b",
    decay_rate: float = 0.15,
) -> np.ndarray:
    """Exponential recency weights: recent months get higher weight.

    Formula: w_i = exp(decay_rate * (month_i - max_month))
    Weights are normalized so sum(weights) == len(train).
    """
    months = pd.to_datetime(train[month_col], format=month_format, errors="coerce")
    month_nums = months.dt.month + (months.dt.year - months.dt.year.min()) * 12
    max_month = month_nums.max()
    raw_weights = np.exp(decay_rate * (month_nums.values - max_month))
    # Normalize so sum == len(train)
    weights = raw_weights * len(train) / raw_weights.sum()
    logger.info(
        f"Temporal weights: min={weights.min():.3f}, max={weights.max():.3f}, "
        f"mean={weights.mean():.3f}"
    )
    return weights


def normalize_data_quality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lowercase and normalize hyphens in object-typed columns.

    Skips ID and target columns to preserve values like 'Yes'/'No'.
    """
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    train_clean = train.copy()
    test_clean = test.copy()

    normalized = []
    for col in train_clean.columns:
        if col in skip_cols:
            continue
        if train_clean[col].dtype == object:
            for df in [train_clean, test_clean]:
                df[col] = df[col].astype(str).str.lower().str.replace("-", " ", regex=False)
            normalized.append(col)

    if normalized:
        logger.info(f"Normalized {len(normalized)} columns: {normalized}")
    return train_clean, test_clean


def get_drift_drop_columns(
    drift_df: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> list[str]:
    """Return temporal columns to drop (100% train/test separation)."""
    return list(feature_types.get("temporal", []))


def apply_all_mitigations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drift_df: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Run all mitigation stages in order.

    Order matters:
    1. Compute temporal weights BEFORE normalization (date parsing is case-sensitive)
    2. Normalize data quality (lowercase, hyphen removal)
    3. Identify columns to drop (temporal columns with 100% separation)

    Returns (train_mitigated, test_mitigated, sample_weights, drop_cols).
    """
    # Stage 5: Temporal reweighting (must happen before normalization)
    temporal_cols = feature_types.get("temporal", [])
    if temporal_cols:
        sample_weights = compute_temporal_weights(train, temporal_cols[0])
    else:
        sample_weights = np.ones(len(train))

    # Stage 4: Data quality normalization
    train_m, test_m = normalize_data_quality(train, test, feature_types)

    # Stage 6: Identify leaky temporal columns to drop
    drop_cols = get_drift_drop_columns(drift_df, feature_types)
    if drop_cols:
        logger.info(f"Columns to drop (temporal leakage): {drop_cols}")

    return train_m, test_m, sample_weights, drop_cols
