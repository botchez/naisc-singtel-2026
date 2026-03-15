import time
import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("naisc")


def safe_n_splits(n_samples: int, desired: int = 5, min_per_fold: int = 20) -> int:
    """Compute a safe number of CV splits given sample size."""
    return max(2, min(desired, n_samples // min_per_fold))


_TEMPORAL_FORMATS = [
    "%y-%b", "%b-%y", "%Y-%m", "%m-%Y", "%b %Y", "%Y-%b",
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
]


def _parse_month_labels(series: pd.Series) -> pd.Series | None:
    """Parse a temporal series into month-period strings for stratification.

    Returns a Series of month labels (e.g. '2025-01') or None if parsing fails.
    """
    raw = series.astype(str).str.title()
    for fmt in _TEMPORAL_FORMATS:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if parsed.notna().sum() > len(series) * 0.5:
            return parsed.dt.to_period("M").astype(str)
    return None


def stratified_subsample_idx(
    n_total: int,
    n_sample: int,
    month_values: pd.Series | np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Return indices for a month-stratified subsample.

    Preserves the proportional distribution of months. Falls back to
    random sampling if month_values is None or all-null.
    """
    rng = np.random.RandomState(seed)

    if n_total <= n_sample:
        return np.arange(n_total)

    # Fallback: random sampling
    if month_values is None:
        return rng.choice(n_total, size=n_sample, replace=False)

    if isinstance(month_values, pd.Series):
        groups = month_values.values
    else:
        groups = np.asarray(month_values)

    # If all null / single group, fall back to random
    unique_groups = pd.unique(groups)
    valid = unique_groups[pd.notna(unique_groups)]
    if len(valid) <= 1:
        return rng.choice(n_total, size=n_sample, replace=False)

    # Proportional allocation per group
    indices = []
    group_indices = {}
    for g in valid:
        group_indices[g] = np.where(groups == g)[0]

    # Handle NaN group (rows that didn't parse)
    nan_mask = pd.isna(groups)
    if nan_mask.any():
        group_indices["_nan_"] = np.where(nan_mask)[0]
        valid = np.append(valid, "_nan_")

    # Compute per-group allocation
    total_valid = sum(len(v) for v in group_indices.values())
    allocations = {}
    remaining = n_sample
    for g in valid:
        alloc = int(np.floor(n_sample * len(group_indices[g]) / total_valid))
        allocations[g] = min(alloc, len(group_indices[g]))
        remaining -= allocations[g]

    # Distribute remainder to groups with largest fractional parts
    fracs = []
    for g in valid:
        frac = (n_sample * len(group_indices[g]) / total_valid) - allocations[g]
        headroom = len(group_indices[g]) - allocations[g]
        if headroom > 0:
            fracs.append((frac, g))
    fracs.sort(key=lambda x: x[0], reverse=True)
    for _, g in fracs[:remaining]:
        allocations[g] += 1

    # Sample within each group
    for g in valid:
        g_idx = group_indices[g]
        k = allocations[g]
        if k >= len(g_idx):
            indices.append(g_idx)
        else:
            indices.append(rng.choice(g_idx, size=k, replace=False))

    return np.concatenate(indices)


class Timer:
    """Simple context-manager timer that prints elapsed seconds."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
