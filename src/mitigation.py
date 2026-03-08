import numpy as np
import pandas as pd
from utils import logger


def detect_abbreviation_mappings(
    train: pd.DataFrame, test: pd.DataFrame, col: str
) -> dict[str, str]:
    """Auto-detect city abbreviation-to-full-name mappings from the data.

    Candidate abbreviations: 2-3 uppercase alpha strings (e.g. "LA", "NYC").
    Candidate full names: multi-word values (contain a space).
    Matches abbreviation to full name whose word-initial letters match.
    Ties broken by combined frequency.
    """
    all_values = set(train[col].dropna().unique()) | set(test[col].dropna().unique())

    abbreviations = [v for v in all_values if len(v) in (2, 3) and v.isupper() and v.isalpha()]
    full_names = [v for v in all_values if " " in v]

    if not abbreviations or not full_names:
        return {}

    # Combined frequency across both datasets
    combined_freq = train[col].value_counts().add(test[col].value_counts(), fill_value=0)

    mapping = {}
    for abbrev in abbreviations:
        matches = [
            fn for fn in full_names
            if "".join(w[0].upper() for w in fn.split()) == abbrev
        ]
        if matches:
            best = max(matches, key=lambda fn: combined_freq.get(fn, 0))
            mapping[abbrev] = best

    return mapping


def normalize_data_quality(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Normalize data quality: city abbreviation mapping, lowercase, hyphen removal.

    Skips ID and target columns to preserve values like 'Yes'/'No'.
    Returns (train_clean, test_clean, norm_log) where norm_log maps column names
    to lists of normalization action labels applied.
    """
    skip_cols = set(feature_types.get("id", []) + feature_types.get("target", []))
    train_clean = train.copy()
    test_clean = test.copy()
    norm_log: dict[str, list[str]] = {}

    # City abbreviation mapping (before lowercasing)
    city_col = "LocationCity"
    if city_col in train_clean.columns:
        abbreviation_map = detect_abbreviation_mappings(train_clean, test_clean, city_col)
        if abbreviation_map:
            for df in [train_clean, test_clean]:
                df[city_col] = df[city_col].replace(abbreviation_map)
            logger.info(f"Auto-detected city abbreviation mappings: {abbreviation_map}")
            norm_log.setdefault(city_col, []).append("City names standardized")

    # Lowercase + hyphen removal on all object columns
    normalized = []
    for col in train_clean.columns:
        if col in skip_cols:
            continue
        if train_clean[col].dtype == object:
            for df in [train_clean, test_clean]:
                df[col] = df[col].astype(str).str.lower().str.replace("-", " ", regex=False)
            normalized.append(col)
            norm_log.setdefault(col, []).extend(["Lowercased", "Hyphens removed"])

    if normalized:
        logger.info(f"Normalized {len(normalized)} columns: {normalized}")
    return train_clean, test_clean, norm_log


# ── Strategy functions ────────────────────────────────────────────────────


def clip_outliers(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Clip both columns to [1st, 99th] percentile of train."""
    vals = train_col.dropna().values.astype(float)
    lo, hi = np.percentile(vals, [1, 99])
    return train_col.clip(lo, hi), test_col.clip(lo, hi)


def log_transform(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series] | None:
    """Apply log1p if all values >= 0; return None to skip otherwise."""
    if (train_col.dropna() < 0).any() or (test_col.dropna() < 0).any():
        return None
    return np.log1p(train_col), np.log1p(test_col)


def quantile_bin(
    train_col: pd.Series, test_col: pd.Series, n_bins: int = 10
) -> tuple[pd.Series, pd.Series]:
    """Discretize into quantile bins defined by train edges."""
    vals = train_col.dropna().values.astype(float)
    percentiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = np.percentile(vals, percentiles)
    edges = np.unique(edges)
    return (
        pd.Series(np.digitize(train_col, edges), index=train_col.index, name=train_col.name),
        pd.Series(np.digitize(test_col, edges), index=test_col.index, name=test_col.name),
    )


def group_rare_categories(
    train_col: pd.Series, test_col: pd.Series, min_freq: float = 0.01
) -> tuple[pd.Series, pd.Series]:
    """Merge categories below min_freq in train + unseen-in-train into '_other'."""
    freq = train_col.value_counts(normalize=True)
    rare_cats = set(freq[freq < min_freq].index)
    train_cats = set(train_col.dropna().unique())

    def _remap(s: pd.Series) -> pd.Series:
        return s.map(lambda v: "_other" if v in rare_cats or v not in train_cats else v)

    return _remap(train_col), _remap(test_col)


def encode_temporal_cyclical(
    train_col: pd.Series, test_col: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Parse %y-%b -> month -> sin/cos pair. Returns (train_sin, train_cos, test_sin, test_cos)."""
    def _to_month(s: pd.Series) -> pd.Series:
        return pd.to_datetime(s, format="%y-%b", errors="coerce").dt.month.fillna(1).astype(float)

    train_month = _to_month(train_col)
    test_month = _to_month(test_col)

    train_sin = np.sin(2 * np.pi * train_month / 12)
    train_cos = np.cos(2 * np.pi * train_month / 12)
    test_sin = np.sin(2 * np.pi * test_month / 12)
    test_cos = np.cos(2 * np.pi * test_month / 12)

    name = train_col.name
    return (
        pd.Series(train_sin, index=train_col.index, name=f"{name}_sin"),
        pd.Series(train_cos, index=train_col.index, name=f"{name}_cos"),
        pd.Series(test_sin, index=test_col.index, name=f"{name}_sin"),
        pd.Series(test_cos, index=test_col.index, name=f"{name}_cos"),
    )


def _clip_then_log(train_col, test_col):
    tr_c, te_c = clip_outliers(train_col, test_col)
    return log_transform(tr_c, te_c)


# ── Progressive mitigation loop ──────────────────────────────────────────


def mitigate_feature_progressive(
    train: pd.DataFrame,
    test: pd.DataFrame,
    col: str,
    feature_type: str,
    original_severity: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Try graduated strategies on a single high-drift feature.

    Returns (train, test, log_entry) where log_entry records what happened.
    """
    from drift_detector import detect_numerical_drift, detect_categorical_drift

    log_entry = {
        "column": col,
        "feature_type": feature_type,
        "original_severity": original_severity,
        "strategy_applied": "Dropped",
        "final_severity": original_severity,
    }

    if feature_type == "temporal":
        # Try cyclical encoding
        train_sin, train_cos, test_sin, test_cos = encode_temporal_cyclical(
            train[col], test[col]
        )
        sin_drift = detect_numerical_drift(train_sin, test_sin)
        cos_drift = detect_numerical_drift(train_cos, test_cos)
        # Accept if BOTH sin and cos are below high
        if sin_drift["severity"] != "high" and cos_drift["severity"] != "high":
            train = train.drop(columns=[col])
            test = test.drop(columns=[col])
            train[train_sin.name] = train_sin.values
            train[train_cos.name] = train_cos.values
            test[test_sin.name] = test_sin.values
            test[test_cos.name] = test_cos.values
            worst = sin_drift["severity"] if sin_drift["severity"] == "moderate" else cos_drift["severity"]
            log_entry["strategy_applied"] = "Cyclical-encoded"
            log_entry["final_severity"] = worst
            return train, test, log_entry
        # Cyclical failed -> drop
        return train, test, log_entry

    if feature_type == "numerical":
        strategies = [
            ("Clipped", lambda tc, sc: clip_outliers(tc, sc)),
            ("Log-transformed", lambda tc, sc: log_transform(tc, sc)),
            ("Clipped+Log", lambda tc, sc: _clip_then_log(tc, sc)),
            ("Binned", lambda tc, sc: quantile_bin(tc, sc)),
        ]
        for name, fn in strategies:
            result = fn(train[col], test[col])
            if result is None:
                continue  # strategy not applicable (e.g. log with negatives)
            tr_col, te_col = result
            drift = detect_numerical_drift(tr_col, te_col)
            if drift["severity"] != "high":
                train[col] = tr_col.values
                test[col] = te_col.values
                log_entry["strategy_applied"] = name
                log_entry["final_severity"] = drift["severity"]
                return train, test, log_entry
        # All strategies failed -> drop
        return train, test, log_entry

    if feature_type == "categorical":
        tr_col, te_col = group_rare_categories(train[col], test[col])
        drift = detect_categorical_drift(tr_col, te_col)
        if drift["severity"] != "high":
            train[col] = tr_col.values
            test[col] = te_col.values
            log_entry["strategy_applied"] = "Rare-grouped"
            log_entry["final_severity"] = drift["severity"]
            return train, test, log_entry
        # Rare grouping failed -> drop
        return train, test, log_entry

    return train, test, log_entry


# ── Main entry point ─────────────────────────────────────────────────────


def apply_all_mitigations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    drift_df: pd.DataFrame,
    feature_types: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[dict], dict[str, list[str]]]:
    """Run all mitigation stages in order.

    Pipeline:
    1. Normalize data quality (city abbreviations, lowercase, hyphen removal)
    2. Re-run drift detection on normalized data
    3. Progressive mitigation: try graduated strategies before dropping

    Returns (train_mitigated, test_mitigated, sample_weights, drop_cols, mitigation_log, norm_log).
    """
    from drift_detector import run_drift_analysis

    # Stage 1: Data quality normalization
    train_m, test_m, norm_log = normalize_data_quality(train, test, feature_types)

    # Stage 2: Re-run drift detection on normalized data
    drift_df_norm = run_drift_analysis(train_m, test_m, feature_types)

    # Stage 3: Progressive mitigation for high-drift features
    # Collect all high-drift features (including temporal)
    high_drift_features = []
    if not drift_df_norm.empty:
        for _, row in drift_df_norm[drift_df_norm["severity"] == "high"].iterrows():
            high_drift_features.append((row["column"], row["feature_type"], row["severity"]))

    mitigation_log = []
    drop_cols = []

    for col, ftype, severity in high_drift_features:
        if col not in train_m.columns:
            continue
        train_m, test_m, entry = mitigate_feature_progressive(
            train_m, test_m, col, ftype, severity
        )
        mitigation_log.append(entry)
        if entry["strategy_applied"] == "Dropped":
            drop_cols.append(col)
        else:
            logger.info(f"  {col}: {entry['strategy_applied']} ({severity} -> {entry['final_severity']})")

    if drop_cols:
        logger.info(f"Dropping {len(drop_cols)} features after all strategies exhausted: {drop_cols}")

    # No sample reweighting — uniform weights
    sample_weights = np.ones(len(train_m))

    return train_m, test_m, sample_weights, drop_cols, mitigation_log, norm_log
