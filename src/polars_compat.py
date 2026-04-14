"""Centralised helpers for polars ↔ pandas/numpy boundary conversions.

External libraries (LightGBM, scikit-learn, scipy) do not accept polars
objects directly.  Every conversion should go through this module so there
is a single place to update when library support improves.
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# DataFrame / Series → pandas (for LightGBM, sklearn, etc.)
# ---------------------------------------------------------------------------

def to_pandas_for_lgbm(df: pl.DataFrame) -> "pd.DataFrame":
    """Convert a polars DataFrame to pandas for LightGBM .fit() / .predict_proba()."""
    return df.to_pandas()


# ---------------------------------------------------------------------------
# Type checking helpers
# ---------------------------------------------------------------------------

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def is_numeric_col(s: pl.Series) -> bool:
    """Return True if the Series has a numeric dtype."""
    return s.dtype in _NUMERIC_DTYPES


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
