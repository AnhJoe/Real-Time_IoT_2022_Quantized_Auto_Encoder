# src/preprocessing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PrepConfig:
    drop_cols: Sequence[str] = ()
    drop_non_numeric: bool = True
    drop_constant: bool = True
    impute_strategy: str = "median"  # "median" or "mean"


def prepare_X(
    X: pd.DataFrame,
    cfg: PrepConfig,
) -> Tuple[pd.DataFrame, list[str]]:
    Xp = X.copy()

    # Drop user-specified columns if present
    for c in cfg.drop_cols:
        if c in Xp.columns:
            Xp = Xp.drop(columns=[c])

    # Keep numeric only (baseline)
    if cfg.drop_non_numeric:
        numeric_cols = [c for c in Xp.columns if pd.api.types.is_numeric_dtype(Xp[c])]
        Xp = Xp[numeric_cols]

    # Drop constant columns
    if cfg.drop_constant:
        non_constant_cols = [c for c in Xp.columns if Xp[c].dropna().nunique() > 1]
        Xp = Xp[non_constant_cols]

    if Xp.shape[1] == 0:
        raise ValueError("No features left after preprocessing.")

    # Impute missing values (baseline: computed on full Xp)
    if cfg.impute_strategy == "median":
        fill_vals = Xp.median(numeric_only=True)
    elif cfg.impute_strategy == "mean":
        fill_vals = Xp.mean(numeric_only=True)
    else:
        raise ValueError("impute_strategy must be 'median' or 'mean'.")

    Xp = Xp.fillna(fill_vals)

    return Xp, list(Xp.columns)


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.values)
    return scaler


def transform_with_scaler(scaler: StandardScaler, X: pd.DataFrame):
    return scaler.transform(X.values)
