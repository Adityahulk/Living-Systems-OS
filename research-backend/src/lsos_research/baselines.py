from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _pred_df(
    case_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event: np.ndarray,
    fold: int,
    model: str,
    risk_score: np.ndarray | None = None,
) -> pd.DataFrame:
    if risk_score is None:
        risk_score = -y_pred
    return pd.DataFrame(
        {
            "case_id": case_ids,
            "y_true_pfs_days": y_true,
            "y_pred_resistance_days": y_pred,
            "risk_score": risk_score,
            "event": event,
            "fold": fold,
            "model": model,
        }
    )


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray, event: np.ndarray, fold: int, model: str) -> dict:
    cidx = concordance_index(y_true, -y_pred, event)
    return {
        "fold": fold,
        "model": model,
        "rmse": _rmse(y_true, y_pred),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "c_index": float(cidx),
    }


def run_cox_cv(
    x_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    data = x_df.join(survival_df.set_index("case_id"), how="inner")
    preds, metrics = [], []

    for fold, (tr, va) in enumerate(splits, start=1):
        tr_df = data.iloc[tr].copy()
        va_df = data.iloc[va].copy()
        cph = CoxPHFitter(penalizer=1e-3)
        cph.fit(tr_df, duration_col="pfs_days", event_col="pfs_event")

        risk = cph.predict_partial_hazard(va_df).values
        pred_time = np.percentile(tr_df["pfs_days"].values, 50) / (1.0 + risk)
        y_true = va_df["pfs_days"].values
        event = va_df["pfs_event"].values

        metrics.append(_metric_row(y_true, pred_time, event, fold, "cox"))
        preds.append(_pred_df(va_df.index.values, y_true, pred_time, event, fold, "cox", risk))

    return pd.concat(preds, ignore_index=True), metrics


def run_aft_cv(
    x_df: pd.DataFrame,
    survival_df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_name: str = "weibull_aft",
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    data = x_df.join(survival_df.set_index("case_id"), how="inner")
    preds, metrics = [], []

    fitter_cls = WeibullAFTFitter if model_name == "weibull_aft" else LogNormalAFTFitter

    for fold, (tr, va) in enumerate(splits, start=1):
        tr_df = data.iloc[tr].copy()
        va_df = data.iloc[va].copy()

        feat_cols = [c for c in tr_df.columns if c not in {"pfs_days", "pfs_event"}]
        scaler = StandardScaler()
        tr_x = scaler.fit_transform(tr_df[feat_cols].values)
        va_x = scaler.transform(va_df[feat_cols].values)

        tr_fit = pd.DataFrame(tr_x, columns=feat_cols, index=tr_df.index)
        tr_fit["pfs_days"] = tr_df["pfs_days"].values
        tr_fit["pfs_event"] = tr_df["pfs_event"].values

        va_fit = pd.DataFrame(va_x, columns=feat_cols, index=va_df.index)
        aft = fitter_cls(penalizer=0.1)
        aft._scipy_fit_method = "SLSQP"
        aft.fit(tr_fit, duration_col="pfs_days", event_col="pfs_event")
        pred_time = aft.predict_median(va_fit).values
        y_true = va_df["pfs_days"].values
        event = va_df["pfs_event"].values

        metrics.append(_metric_row(y_true, pred_time, event, fold, model_name))
        preds.append(_pred_df(va_df.index.values, y_true, pred_time, event, fold, model_name))

    return pd.concat(preds, ignore_index=True), metrics


def _run_regression_cv(
    x_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model,
    model_name: str,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    x = x_df.values
    y = y_days.values
    e = y_event.values
    preds, metrics = [], []

    for fold, (tr, va) in enumerate(splits, start=1):
        model.fit(x[tr], y[tr])
        y_hat = model.predict(x[va])
        y_true = y[va]
        event = e[va]
        metrics.append(_metric_row(y_true, y_hat, event, fold, model_name))
        preds.append(_pred_df(x_df.iloc[va].index.values, y_true, y_hat, event, fold, model_name))

    return pd.concat(preds, ignore_index=True), metrics


def run_mlp_cv(
    x_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        max_iter=500,
        random_state=seed,
    )
    return _run_regression_cv(x_df, y_days, y_event, splits, model, "mlp")


def run_gbr_cv(
    x_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    model = GradientBoostingRegressor(random_state=seed, n_estimators=300, learning_rate=0.05)
    return _run_regression_cv(x_df, y_days, y_event, splits, model, "gbr")


def run_rf_cv(
    x_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    model = RandomForestRegressor(random_state=seed, n_estimators=500, min_samples_leaf=3, n_jobs=-1)
    return _run_regression_cv(x_df, y_days, y_event, splits, model, "rf")


class DeepSurvNet(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _cox_ph_loss(risk: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(durations, descending=True)
    risk = risk[order]
    events = events[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    per_event = risk - log_cumsum
    loss = -(per_event * events).sum() / (events.sum() + 1e-8)
    return loss


def run_deepsurv_cv(
    x_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    x = x_df.values.astype(np.float32)
    y = y_days.values.astype(np.float32)
    e = y_event.values.astype(np.float32)
    preds, metrics = [], []

    torch.manual_seed(seed)
    np.random.seed(seed)

    for fold, (tr, va) in enumerate(splits, start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_va = scaler.transform(x[va])

        xt = torch.tensor(x_tr, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        et = torch.tensor(e[tr], dtype=torch.float32)

        model = DeepSurvNet(in_dim=x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best = float("inf")
        best_state = None
        stale = 0
        for _ in range(300):
            model.train()
            risk = model(xt)
            loss = _cox_ph_loss(risk, yt, et)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            lv = float(loss.item())
            if lv < best:
                best = lv
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 30:
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            risk_va = model(torch.tensor(x_va, dtype=torch.float32)).numpy()

        y_true = y[va]
        event = e[va]
        # Monotonic map for RMSE comparability.
        pred_time = np.percentile(y[tr], 50) / (1.0 + np.exp(risk_va))

        metrics.append(_metric_row(y_true, pred_time, event, fold, "deepsurv"))
        preds.append(
            _pred_df(
                x_df.iloc[va].index.values,
                y_true,
                pred_time,
                event,
                fold,
                "deepsurv",
                risk_score=risk_va,
            )
        )

    return pd.concat(preds, ignore_index=True), metrics
