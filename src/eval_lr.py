from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np
import pandas as pd


def tune_threshold_max_f1_attack_lr(y_true, y_score, thresholds=None):
    """
    Select the probability threshold that maximizes F1 for the attack class.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_thr = 0.5
    best_f1 = -np.inf

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def fit_tune_evaluate_lr(
    Xtr,
    ytr,
    Xva,
    yva,
    Xte,
    yte,
    model_name,
    random_state=42,
):
    """
    Fit a logistic regression pipeline, tune threshold on validation,
    and evaluate on the test set.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state
        ))
    ])

    # Fit model
    pipe.fit(Xtr, ytr)

    # Validation probabilities for threshold tuning
    val_score = pipe.predict_proba(Xva)[:, 1]

    # Tune threshold using validation F1 for attack class
    best_thr, best_val_f1 = tune_threshold_max_f1_attack_lr(
        np.asarray(yva),
        val_score
    )

    # Test probabilities and thresholded predictions
    test_score = pipe.predict_proba(Xte)[:, 1]
    yte_pred = (test_score >= best_thr).astype(int)

    # Collect summary metrics
    results = {
        "model": model_name,
        "n_features": Xtr.shape[1],
        "threshold": best_thr,
        "val_f1": f1_score(yva, (val_score >= best_thr).astype(int), pos_label=1, zero_division=0),
        "test_roc_auc": roc_auc_score(yte, test_score),
        "test_pr_auc": average_precision_score(yte, test_score),
        "test_accuracy": accuracy_score(yte, yte_pred),
        "test_precision": precision_score(yte, yte_pred, pos_label=1, zero_division=0),
        "test_recall": recall_score(yte, yte_pred, pos_label=1, zero_division=0),
        "test_f1": f1_score(yte, yte_pred, pos_label=1, zero_division=0),
    }

    cm = confusion_matrix(yte, yte_pred)

    artifacts = {
        "pipe": pipe,
        "val_score": val_score,
        "test_score": test_score,
        "y_test_pred": yte_pred,
        "confusion_matrix": cm,
        "results": results,
    }

    return artifacts