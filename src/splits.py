# src/splits.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.20
    val_size: float = 0.20  # applied after test split
    random_state: int = 42


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: SplitConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=cfg.val_size, stratify=y_trainval, random_state=cfg.random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test
