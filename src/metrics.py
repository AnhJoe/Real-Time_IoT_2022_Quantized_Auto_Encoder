# src/metrics.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    precision_attack: float
    recall_attack: float
    f1_attack: float
    f1_macro: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> BinaryMetrics:
    y_pred = (y_score >= threshold).astype(int)

    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_a = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    roc = roc_auc_score(y_true, y_score)
    pr  = average_precision_score(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return BinaryMetrics(
        threshold=float(threshold),
        precision_attack=float(prec),
        recall_attack=float(rec),
        f1_attack=float(f1_a),
        f1_macro=float(f1_m),
        roc_auc=float(roc),
        pr_auc=float(pr),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
