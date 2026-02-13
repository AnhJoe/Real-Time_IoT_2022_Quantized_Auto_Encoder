# src/data_validation.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class XyValidationReport:
    n_rows: int
    n_features: int
    n_missing_total: int
    n_duplicate_rows: int
    non_numeric_feature_cols: list[str]
    constant_feature_cols: list[str]


def validate_binary_Xy(X: pd.DataFrame, y: pd.Series) -> XyValidationReport:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Row mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows")

    classes = pd.unique(y.dropna())
    if len(classes) != 2:
        raise ValueError(f"Binary y must have exactly 2 classes. Found: {classes.tolist()}")

    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    constant = [c for c in X.columns if X[c].dropna().nunique() <= 1]

    return XyValidationReport(
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        n_missing_total=int(X.isna().sum().sum()),
        n_duplicate_rows=int(X.duplicated().sum()),
        non_numeric_feature_cols=non_numeric,
        constant_feature_cols=constant,
    )


def class_balance(y: pd.Series) -> pd.DataFrame:
    counts = y.value_counts().sort_index()
    rates = counts / counts.sum()
    out = pd.DataFrame({"count": counts, "rate": rates})
    out.index.name = "class"
    return out
