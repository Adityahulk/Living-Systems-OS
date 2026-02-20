from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def c_index(y_true_days: np.ndarray, risk_score: np.ndarray, event_observed: np.ndarray | None = None) -> float:
    """Compute concordance index for survival ranking quality."""
    if event_observed is None:
        event_observed = np.ones_like(y_true_days)
    return float(concordance_index(y_true_days, risk_score, event_observed))


def brier_score_at_horizon(y_true_days: np.ndarray, y_pred_days: np.ndarray, horizon: float) -> float:
    """Compute Brier score at a fixed horizon using predicted event times."""
    # Event indicator by horizon using predicted resistance time as deterministic event-time predictor.
    y_event_true = (y_true_days <= horizon).astype(float)
    y_event_pred = (y_pred_days <= horizon).astype(float)
    return float(np.mean((y_event_pred - y_event_true) ** 2))


def calibration_curve_data(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Build calibration bins with robust fallback for low-variance predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return pd.DataFrame(columns=["pred_mean", "true_mean", "n"])

    # Robust binning: quantile bins first, fallback to ranked equal-width bins for low-variance predictions.
    if np.unique(y_pred).size > 1:
        q = pd.qcut(y_pred, q=n_bins, duplicates="drop")
    else:
        q = pd.Series(np.zeros_like(y_pred, dtype=int))
    tmp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bin": q})
    if tmp["bin"].nunique(dropna=True) <= 1 and len(tmp) >= 2:
        ranks = pd.Series(y_pred).rank(method="first", pct=True).to_numpy()
        edges = np.linspace(0.0, 1.0, min(n_bins, len(tmp)) + 1)
        bin_ids = np.digitize(ranks, edges[1:-1], right=True)
        tmp["bin"] = bin_ids

    out = tmp.groupby("bin", observed=True).agg(
        pred_mean=("y_pred", "mean"),
        true_mean=("y_true", "mean"),
        n=("y_true", "size"),
    )
    return out.reset_index(drop=True)


def paired_t_test(metric_df: pd.DataFrame, model_a: str, model_b: str, metric_col: str) -> dict[str, float]:
    """Run paired t-test on fold metrics between two models."""
    a = metric_df.loc[metric_df["model"] == model_a, metric_col].values
    b = metric_df.loc[metric_df["model"] == model_b, metric_col].values
    if len(a) != len(b):
        m = min(len(a), len(b))
        a = a[:m]
        b = b[:m]
    stat, pval = stats.ttest_rel(a, b, nan_policy="omit")
    return {
        "t_stat": float(stat),
        "p_value": float(pval),
        "mean_a": float(np.nanmean(a)),
        "mean_b": float(np.nanmean(b)),
        "delta_a_minus_b": float(np.nanmean(a - b)),
    }


def paired_bootstrap_rmse_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval and p-value for RMSE delta between models."""
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)
    m = min(len(y_true), len(a), len(b))
    y_true, a, b = y_true[:m], a[:m], b[:m]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        da = rmse(y_true[idx], a[idx])
        db = rmse(y_true[idx], b[idx])
        deltas.append(da - db)  # <0 means model A better

    deltas = np.asarray(deltas)
    lo = (1 - alpha) / 2
    hi = 1 - lo
    p_nonpos = np.mean(deltas <= 0)
    p_nonneg = np.mean(deltas >= 0)
    p_two_sided = float(2 * min(p_nonpos, p_nonneg))
    return {
        "delta_rmse_mean_a_minus_b": float(np.mean(deltas)),
        "ci_low": float(np.quantile(deltas, lo)),
        "ci_high": float(np.quantile(deltas, hi)),
        "bootstrap_p_two_sided": p_two_sided,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a metric function."""
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
