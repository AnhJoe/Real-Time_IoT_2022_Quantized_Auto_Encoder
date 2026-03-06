from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def preprocess_categorical_ports(
    df: pd.DataFrame,
    port_col: str = "id.resp_p",
    service_col: str = "service",
    proto_col: str = "proto",
    drop_first: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group response ports into service categories and apply one-hot encoding
    to categorical network features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing categorical columns.
    port_col : str
        Column containing response port identifiers.
    service_col : str
        Column containing service names.
    proto_col : str
        Column containing protocol types.
    drop_first : bool
        Whether to drop the first category in one-hot encoding.

    Returns
    -------
    df_encoded : pd.DataFrame
        One-hot encoded categorical feature matrix.
    port_group_counts : pd.DataFrame
        Frequency table of grouped port categories for EDA.
    """

    df = df.copy()

    # ---------------------------
    # Port grouping
    # ---------------------------
    def categorize_port(port):

        if port in {80, 443, 8080, 8443}:
            return "web"

        elif port == 53:
            return "dns"

        elif port == 22:
            return "ssh"

        elif port in {1883, 8883}:
            return "mqtt"

        elif port == 123:
            return "ntp"

        elif port in {25, 465, 587}:
            return "smtp"

        elif port in {110, 995, 143, 993}:
            return "mail"

        elif port in {20, 21}:
            return "ftp"

        elif port < 1024:
            return "well_known_other"

        else:
            return "high_port"


    df["port_group"] = df[port_col].apply(categorize_port)

    # ---------------------------
    # Port group frequency table
    # ---------------------------
    port_group_counts = (
        df["port_group"]
        .value_counts()
        .rename_axis("port_group")
        .reset_index(name="count")
    )

    # ---------------------------
    # One-hot encoding
    # ---------------------------
    categorical_cols = [service_col, proto_col, "port_group"]

    df_encoded = pd.get_dummies(
        df[categorical_cols],
        prefix=categorical_cols,
        drop_first=drop_first
    )

    return df_encoded, port_group_counts


# ---------------------------
# Stratified split
# ---------------------------

def stratified_train_val_test_split(
    X,
    y,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    downsample_train=None,   # optional fraction of training set to keep
    random_state=42,
):
    """
    Split X and y into stratified train/validation/test sets.

    Optionally applies class-balanced downsampling to the training set only.
    Validation and test sets are left unchanged for unbiased evaluation.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like
        Target vector.
    train_size : float, default=0.70
        Proportion of data assigned to training.
    val_size : float, default=0.15
        Proportion of data assigned to validation.
    test_size : float, default=0.15
        Proportion of data assigned to test.
    downsample_train : float or None, default=None
        Fraction of the training set to keep, using stratified sampling.
        Example: 0.25 keeps 25% of the training rows while preserving class balance.
        If None, no downsampling is applied.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    y = np.asarray(y)

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    # ---------------------------
    # 1) Split train vs temp
    # ---------------------------
    # temp will later be split into validation and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )

    # ---------------------------
    # 2) Split temp into val/test
    # ---------------------------
    val_prop = val_size / (val_size + test_size)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_prop),
        stratify=y_temp,
        random_state=random_state,
    )

    # ---------------------------
    # 3) Optional class-balanced downsampling of training set
    # ---------------------------
    # Keeps only a fraction of the training data while preserving
    # the class proportions in y_train.
    if downsample_train is not None:
        if not (0 < downsample_train <= 1):
            raise ValueError("downsample_train must be in (0, 1]")

        if downsample_train < 1.0:
            if hasattr(X_train, "iloc"):
                # If X is a DataFrame, use row positions and keep DataFrame output
                train_idx = np.arange(len(y_train))
                keep_idx, _ = train_test_split(
                    train_idx,
                    train_size=downsample_train,
                    stratify=y_train,
                    random_state=random_state,
                )
                X_train = X_train.iloc[keep_idx]
            else:
                # If X is a NumPy array, sample indices the same way
                train_idx = np.arange(len(y_train))
                keep_idx, _ = train_test_split(
                    train_idx,
                    train_size=downsample_train,
                    stratify=y_train,
                    random_state=random_state,
                )
                X_train = X_train[keep_idx]

            y_train = y_train[keep_idx]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ---------------------------
# Matrix prep for clustering
# ---------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    scaler: str = "standard"        # {"standard","robust"}
    impute: str = "median"          # {"mean","median"}
    sample_n: Optional[int] = 50000 # None for all rows
    random_state: int = 42


def build_preprocess_pipeline(cfg: PreprocessConfig) -> Pipeline:
    """Impute + scale numeric matrix for clustering/PCA."""
    if cfg.scaler not in {"standard", "robust"}:
        raise ValueError("cfg.scaler must be one of {'standard','robust'}")
    if cfg.impute not in {"mean", "median"}:
        raise ValueError("cfg.impute must be one of {'mean','median'}")

    scaler = StandardScaler() if cfg.scaler == "standard" else RobustScaler()
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cfg.impute)),
        ("scaler", scaler),
    ])


def prepare_matrix(
    X: pd.DataFrame,
    cfg: PreprocessConfig,
) -> Tuple[np.ndarray, pd.Index]:
    """
    Transform (impute + scale) numeric feature matrix. Optionally sample rows
    for speed/plot readability. Returns (X_transformed, index_used).
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    # Optional sampling
    if cfg.sample_n is not None and len(X) > cfg.sample_n:
        X_sub = X.sample(n=cfg.sample_n, random_state=cfg.random_state)
    else:
        X_sub = X

    pipe = build_preprocess_pipeline(cfg)
    X_t = pipe.fit_transform(X_sub)

    return X_t, X_sub.index