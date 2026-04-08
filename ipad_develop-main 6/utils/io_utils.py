"""I/O utility functions for CSV export and timestamp creation."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def dataframe_to_csv_bytes(df: pd.DataFrame, encoding: str = "utf-8-sig") -> bytes:
    """Convert dataframe to CSV bytes for Streamlit download.

    Args:
        df: Source dataframe.
        encoding: Text encoding for CSV bytes.

    Returns:
        Encoded CSV as bytes.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    return df.to_csv(index=False).encode(encoding)


def get_timestamp(fmt: str) -> str:
    """Return current timestamp string with caller-provided format."""
    if not fmt.strip():
        raise ValueError("Timestamp format must not be empty.")
    return datetime.now().strftime(fmt)
