# src/modeling_logreg.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class LogRegConfig:
    C: float = 1.0
    max_iter: int = 2000
    solver: str = "lbfgs"
    class_weight: str | None = None  # set "balanced" if needed
    random_state: int = 42


def train_logreg(X_train: np.ndarray, y_train: np.ndarray, cfg: LogRegConfig) -> LogisticRegression:
    model = LogisticRegression(
        C=cfg.C,
        max_iter=cfg.max_iter,
        solver=cfg.solver,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
    )
    model.fit(X_train, y_train)
    return model


def proba_attack(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def tune_threshold_max_f1_attack(y_val: np.ndarray, score_val: np.ndarray, n_grid: int = 501) -> Tuple[float, float]:
    thresholds = np.linspace(0.0, 1.0, n_grid)
    best_thr, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (score_val >= t).astype(int)
        f1 = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = float(f1), float(t)
    return best_thr, best_f1
