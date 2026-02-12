from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def c_index(y_true_days: np.ndarray, risk_score: np.ndarray, event_observed: np.ndarray | None = None) -> float:
    if event_observed is None:
        event_observed = np.ones_like(y_true_days)
    return float(concordance_index(y_true_days, risk_score, event_observed))


def brier_score_at_horizon(y_true_days: np.ndarray, y_pred_days: np.ndarray, horizon: float) -> float:
    # Event indicator by horizon using predicted resistance time as deterministic event-time predictor.
    y_event_true = (y_true_days <= horizon).astype(float)
    y_event_pred = (y_pred_days <= horizon).astype(float)
    return float(np.mean((y_event_pred - y_event_true) ** 2))


def calibration_curve_data(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(y_pred, q=n_bins, duplicates="drop")
    tmp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bin": q})
    out = tmp.groupby("bin", observed=True).agg(
        pred_mean=("y_pred", "mean"),
        true_mean=("y_true", "mean"),
        n=("y_true", "size"),
    )
    return out.reset_index(drop=True)


def paired_t_test(metric_df: pd.DataFrame, model_a: str, model_b: str, metric_col: str) -> dict[str, float]:
    a = metric_df.loc[metric_df["model"] == model_a, metric_col].values
    b = metric_df.loc[metric_df["model"] == model_b, metric_col].values
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    stat, pval = stats.ttest_rel(a, b, nan_policy="omit")
    return {"t_stat": float(stat), "p_value": float(pval)}


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats_arr = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stats_arr.append(metric_fn(y_true[idx], y_pred[idx]))

    lo = (1 - alpha) / 2
    hi = 1 - lo
    return {
        "mean": float(np.mean(stats_arr)),
        "ci_low": float(np.quantile(stats_arr, lo)),
        "ci_high": float(np.quantile(stats_arr, hi)),
    }
