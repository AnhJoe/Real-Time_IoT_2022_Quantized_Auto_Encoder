import numpy as np
import pandas as pd
import warnings

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from sklearn.metrics import f1_score


def _fit_logit_bic(X_df: pd.DataFrame, y: np.ndarray) -> tuple[float, object | None]:
    """
    Fit a logistic regression model with an intercept and return its BIC.

    Parameters
    ----------
    X_df : pd.DataFrame
        Predictor matrix for the candidate model.
    y : np.ndarray
        Binary response variable.

    Returns
    -------
    tuple[float, object | None]
        - BIC of the fitted model
        - Fitted statsmodels result object, or None if fitting fails

    Notes
    -----
    If statsmodels fails to fit the model (e.g., singular matrix,
    perfect separation, convergence issues), this function returns
    infinite BIC so the candidate model is treated as unfavorable.
    """
    try:
        # Add intercept term explicitly
        X_with_const = sm.add_constant(X_df, has_constant="add")

        # Suppress convergence warnings only for this model fit
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

            result = sm.Logit(y, X_with_const).fit(disp=False, maxiter=200)

        return float(result.bic), result

    except Exception:
        # If model fitting fails, assign worst possible score
        return float("inf"), None


def stepwise_bic_bidirectional(
    X_df: pd.DataFrame,
    y: np.ndarray,
    max_steps: int = 200,
    min_improve: float = 1e-6,
    verbose: bool = True,
) -> dict:
    """
    Perform bidirectional stepwise feature selection using BIC.

    The procedure starts with the intercept-only model and repeatedly:
      1. tries adding each remaining feature (forward step),
      2. tries dropping each currently selected feature (backward step),
    then chooses the single action that gives the lowest BIC.

    The algorithm stops when no candidate model improves BIC by at least
    `min_improve`, or when `max_steps` is reached.

    Parameters
    ----------
    X_df : pd.DataFrame
        Full set of candidate predictors.
    y : np.ndarray
        Binary response variable.
    max_steps : int, default=200
        Maximum number of stepwise iterations.
    min_improve : float, default=1e-6
        Minimum BIC decrease required to accept a step.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    dict
        Dictionary containing:
        - "selected_features": list of chosen feature names
        - "bic": final BIC
        - "result": fitted statsmodels result object for final model
    """
    selected_features = []
    remaining_features = list(X_df.columns)

    current_bic, current_result = _fit_logit_bic(X_df[selected_features], y)

    # Store step-by-step history for reporting
    history = [{
        "step": 0,
        "action": "start",
        "feature": None,
        "bic": current_bic,
        "delta_bic": np.nan,
        "n_selected": 0,
        "selected_features": []
    }]

    if verbose:
        print(f"Start BIC: {current_bic:.3f} | selected={len(selected_features)}")

    for step in range(1, max_steps + 1):
        best_candidate = (float("inf"), None, None)
        best_result = None

        # Forward step: try adding each remaining feature
        for feature in remaining_features:
            trial_features = selected_features + [feature]
            trial_bic, trial_result = _fit_logit_bic(X_df[trial_features], y)

            if trial_bic < best_candidate[0]:
                best_candidate = (trial_bic, "add", feature)
                best_result = trial_result

        # Backward step: try dropping each selected feature
        for feature in selected_features:
            trial_features = [f for f in selected_features if f != feature]
            trial_bic, trial_result = _fit_logit_bic(X_df[trial_features], y)

            if trial_bic < best_candidate[0]:
                best_candidate = (trial_bic, "drop", feature)
                best_result = trial_result

        new_bic, action, feature = best_candidate

        if verbose:
            print(
                f"Step {step}: {action} {feature} -> "
                f"BIC {new_bic:.3f} (current {current_bic:.3f})"
            )

        # Stop if no meaningful improvement
        if new_bic + min_improve >= current_bic:
            if verbose:
                print("No BIC improvement. Stop.")
            break

        # Apply chosen move
        if action == "add":
            selected_features.append(feature)
            remaining_features.remove(feature)
        elif action == "drop":
            selected_features.remove(feature)
            remaining_features.append(feature)

        delta_bic = new_bic - current_bic
        current_bic = new_bic
        current_result = best_result

        # Save this accepted step
        history.append({
            "step": step,
            "action": action,
            "feature": feature,
            "bic": current_bic,
            "delta_bic": delta_bic,
            "n_selected": len(selected_features),
            "selected_features": selected_features.copy()
        })

    return {
        "selected_features": selected_features,
        "bic": current_bic,
        "result": current_result,
        "history": history,
    }