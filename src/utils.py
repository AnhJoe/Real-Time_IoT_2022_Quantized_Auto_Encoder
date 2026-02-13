from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd


def as_table(obj: Any) -> pd.DataFrame:
    """
    Convert a dataclass or dict into a 1-row pandas DataFrame.

    Usage:
        report_df = as_table(validation_report)
    """
    if is_dataclass(obj):
        obj = asdict(obj)
    if not isinstance(obj, dict):
        raise TypeError("as_table expects a dataclass or dict.")
    return pd.DataFrame([obj])


def show_table(
    data: Any,
    *,
    max_rows: int = 30,
    max_cols: int = 20,
) -> pd.DataFrame:
    """
    Return a DataFrame (or DataFrame preview) so Jupyter/Quarto renders it
    natively when it is the last expression in a cell.

    Notes:
      - This function does NOT print or call display().
      - To render, call it as the last line in a cell (or return it from a function).

    Usage:
        show_table(df)                      # renders
        show_table(df, max_rows=10)         # renders first 10 rows
        show_table(series)                  # converts to DataFrame and renders
    """
    if isinstance(data, pd.Series):
        df = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)

    return df.iloc[:max_rows, :max_cols]
