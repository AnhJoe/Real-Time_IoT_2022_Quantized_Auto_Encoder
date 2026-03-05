from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

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