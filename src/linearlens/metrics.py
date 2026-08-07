from __future__ import annotations

import numpy as np


def cronbach_alpha(items: np.ndarray) -> float:
    """Compute Cronbach's alpha from an [respondents, items] matrix."""
    x = np.asarray(items, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("items must be [respondents, >=2 items]")
    item_var = x.var(axis=0, ddof=1)
    total_var = x.sum(axis=1).var(ddof=1)
    if total_var <= 1e-12:
        raise ValueError("total response variance is zero")
    k = x.shape[1]
    return float(k / (k - 1) * (1 - item_var.sum() / total_var))


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape")
    error = y - p
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    denom = np.abs(y) + np.abs(p)
    ratio = np.divide(
        2 * np.abs(error),
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    smape = float(np.mean(ratio) * 100)
    return {"mae": mae, "rmse": rmse, "smape": smape}
