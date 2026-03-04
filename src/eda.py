import numpy as np
import pandas as pd


def effect_size(
    df: pd.DataFrame,
    normal_mask: pd.Series,
    attack_mask: pd.Series,
    eps: float = 1e-12,
):
    """
    Create an effect_size(col) function that computes absolute Cohen's d
    between normal and attack groups for a single column.

    This returns a closure so you can do:
        effect_size = make_effect_size(df, normal_mask, attack_mask, eps)
        X_num.columns.to_series().apply(effect_size)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the feature columns.
    normal_mask : pd.Series
        Boolean mask for normal rows (aligned to df.index).
    attack_mask : pd.Series
        Boolean mask for attack rows (aligned to df.index).
    eps : float, default=1e-12
        Stability term to avoid division by ~0.

    Returns
    -------
    callable
        Function effect_size(col: str) -> float
    """

    # Ensure masks align to df index
    normal_mask = normal_mask.reindex(df.index).fillna(False)
    attack_mask = attack_mask.reindex(df.index).fillna(False)

    def effect_size(col: str) -> float:
        # Convert to numeric and drop invalid values
        n = pd.to_numeric(df.loc[normal_mask, col], errors="coerce").dropna()
        a = pd.to_numeric(df.loc[attack_mask, col], errors="coerce").dropna()

        # If either group is empty after coercion, treat as missing
        if n.empty or a.empty:
            return np.nan

        pooled_std = np.sqrt((n.var(ddof=0) + a.var(ddof=0)) / 2)

        if not np.isfinite(pooled_std) or pooled_std < eps:
            return 0.0  # essentially no variability (or degenerate)

        return float(abs(n.mean() - a.mean()) / pooled_std)

    return effect_size