import numpy as np
import pandas as pd
from utils import logger


CITY_ABBREVIATIONS = {
    "LA": "Los Angeles",
    "SD": "San Diego",
    "SJ": "San Jose",
}


def normalize_data_quality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize data quality: city abbreviation mapping, lowercase, hyphen removal.

    Skips ID and target columns to preserve values like 'Yes'/'No'.
    """
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    train_clean = train.copy()
    test_clean = test.copy()

    # City abbreviation mapping (before lowercasing)
    city_col = "LocationCity"
    if city_col in train_clean.columns:
        for df in [train_clean, test_clean]:
            df[city_col] = df[city_col].replace(CITY_ABBREVIATIONS)
        logger.info(f"Mapped city abbreviations: {list(CITY_ABBREVIATIONS.keys())}")

    # Lowercase + hyphen removal on all object columns
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
    """Return features to drop based on drift severity.

    Drops:
    - All temporal columns (100% train/test separation, temporal leakage)
    - All features with 'high' drift severity (significant covariate shift
      that risks concept drift and hurts generalization)

    The drift severity is assigned by drift_detector based on statistical tests
    (PSI, KS for numerical; JSD, Chi-squared for categorical). Features with
    high severity have distributions that differ substantially between train
    and test, making their learned relationships unreliable for prediction.
    """
    # Always drop temporal columns
    drops = list(feature_types.get("temporal", []))

    # Drop high-severity drift features
    if not drift_df.empty:
        high_drift = drift_df[drift_df["severity"] == "high"]["column"].tolist()
        for col in high_drift:
            if col not in drops:
                drops.append(col)

    return drops


def apply_all_mitigations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drift_df: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Run all mitigation stages in order.

    Pipeline:
    1. Normalize data quality (city abbreviations, lowercase, hyphen removal)
    2. Re-run drift detection on normalized data to measure true distributional
       drift (not case/formatting artifacts)
    3. Drop features with high drift severity

    No sample reweighting is applied — experiments showed that after removing
    high-drift features, the remaining features have negligible covariate shift
    (adversarial AUC ~0.50), making reweighting unnecessary.

    Returns (train_mitigated, test_mitigated, sample_weights, drop_cols).
    """
    from drift_detector import run_drift_analysis

    # Stage 1: Data quality normalization
    train_m, test_m = normalize_data_quality(train, test, feature_types)

    # Stage 2: Re-run drift detection on normalized data
    # This ensures we measure real distributional drift, not case artifacts
    # (e.g., Contract "Month-to-Month" vs "month-to-month" is not real drift)
    drift_df_norm = run_drift_analysis(train_m, test_m, feature_types)

    # Stage 3: Drop high-drift features
    drop_cols = get_drift_drop_columns(drift_df_norm, feature_types)

    # No sample reweighting — uniform weights
    sample_weights = np.ones(len(train_m))

    return train_m, test_m, sample_weights, drop_cols
