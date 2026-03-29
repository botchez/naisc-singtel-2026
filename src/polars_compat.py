"""Centralised helpers for polars ↔ pandas/numpy boundary conversions.

External libraries (LightGBM, scikit-learn, scipy) do not accept polars
objects directly.  Every conversion should go through this module so there
is a single place to update when library support improves.
"""

from __future__ import annotations

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# DataFrame / Series → pandas (for LightGBM, sklearn, etc.)
# ---------------------------------------------------------------------------

def to_pandas_for_lgbm(df: pl.DataFrame) -> "pd.DataFrame":
    """Convert a polars DataFrame to pandas for LightGBM .fit() / .predict_proba()."""
    import pandas as pd
    return df.to_pandas()


def to_pandas_df(df: pl.DataFrame) -> "pd.DataFrame":
    """Generic polars DataFrame → pandas DataFrame."""
    import pandas as pd
    return df.to_pandas()


def to_pandas_series(s: pl.Series) -> "pd.Series":
    """Generic polars Series → pandas Series."""
    import pandas as pd
    return s.to_pandas()


# ---------------------------------------------------------------------------
# Series → numpy (for scipy.stats, sklearn metrics, etc.)
# ---------------------------------------------------------------------------

def to_numpy_for_scipy(s: pl.Series) -> np.ndarray:
    """Extract a float64 numpy array from a polars Series (drops nulls)."""
    return s.cast(pl.Float64, strict=False).drop_nulls().to_numpy()


# ---------------------------------------------------------------------------
# Type checking helpers (replaces pd.api.types.is_numeric_dtype)
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def is_numeric_col(s: pl.Series) -> bool:
    """Return True if the Series has a numeric dtype."""
    return s.dtype in _NUMERIC_DTYPES


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Return True if *dtype* is numeric."""
    return dtype in _NUMERIC_DTYPES


# ---------------------------------------------------------------------------
# Numeric coercion (replaces pd.to_numeric(x, errors="coerce"))
# ---------------------------------------------------------------------------

def to_numeric_coerce(s: pl.Series) -> pl.Series:
    """Cast a Series to Float64, coercing failures to null.

    Mirrors ``pd.to_numeric(s, errors="coerce")``.
    """
    if s.dtype in _NUMERIC_DTYPES:
        return s.cast(pl.Float64)
    # String-like column: attempt cast
    return s.cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False)


# ---------------------------------------------------------------------------
# Conversion from pandas → polars
# ---------------------------------------------------------------------------

def from_pandas_df(df: "pd.DataFrame") -> pl.DataFrame:
    """pandas DataFrame → polars DataFrame."""
    return pl.from_pandas(df)


def from_pandas_series(s: "pd.Series") -> pl.Series:
    """pandas Series → polars Series."""
    return pl.from_pandas(s)


# ---------------------------------------------------------------------------
# Auto-conversion guards (accept pandas or polars, always return polars)
# ---------------------------------------------------------------------------

def ensure_pl_df(df) -> pl.DataFrame:
    """Auto-convert pandas DataFrame to polars if needed."""
    if isinstance(df, pl.DataFrame):
        return df
    import pandas as pd
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


def ensure_pl_series(s) -> pl.Series:
    """Auto-convert pandas Series to polars if needed."""
    if isinstance(s, pl.Series):
        return s
    import pandas as pd
    if isinstance(s, pd.Series):
        return pl.from_pandas(s)
    return pl.Series(values=s)
