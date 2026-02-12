from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from lifelines.utils import concordance_index

from .hybrid_model import (
    ODESimConfig,
    ParameterNet,
    differentiable_resistance_time_from_trajectory,
    make_dose_fn,
    plausibility_penalty,
    resistance_time_from_trajectory,
    simulate_trajectories,
    smoothness_penalty,
)


@dataclass
class TrainingConfig:
    hidden_dims: list[int]
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    n_splits: int
    ode_t_max_days: int
    ode_steps: int
    loss_weights: dict[str, float]


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_hybrid_cv(
    x_df: pd.DataFrame,
    y_pfs: pd.Series,
    y_event: pd.Series,
    cfg: TrainingConfig,
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ablation: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    x = torch.tensor(x_df.values, dtype=torch.float32)
    y = torch.tensor(y_pfs.values, dtype=torch.float32)

    if splits is None:
        raise ValueError("train_hybrid_cv requires explicit CV splits")
    fold_metrics: list[dict[str, float]] = []
    preds: list[pd.DataFrame] = []

    y_event_t = torch.tensor(y_event.values, dtype=torch.float32)

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        model = ParameterNet(in_dim=x.shape[1], hidden_dims=cfg.hidden_dims)

        if ablation == "freeze_parameters":
            for p in model.parameters():
                p.requires_grad = False

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        ) if any(p.requires_grad for p in model.parameters()) else None

        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_va, y_va = x[va_idx], y[va_idx]
        e_va = y_event_t[va_idx]

        best_val = float("inf")
        best_state = None
        stale = 0

        for _ in range(cfg.epochs):
            model.train()
            constrained = ablation != "unconstrained_parameters"
            r, alpha, k = model(x_tr, constrained=constrained)

            t, traj = simulate_trajectories(
                r,
                alpha,
                k,
                ODESimConfig(
                    t_max_days=cfg.ode_t_max_days,
                    n_steps=cfg.ode_steps,
                    use_drug_term=(ablation != "no_drug_term"),
                ),
                dose_fn=make_dose_fn(),
            )

            pred_res_t_soft = differentiable_resistance_time_from_trajectory(t, traj)
            loss = cfg.loss_weights["mse"] * torch.mean((pred_res_t_soft - y_tr) ** 2)

            if ablation != "unconstrained_parameters":
                loss = loss + cfg.loss_weights["plausibility"] * plausibility_penalty(r, alpha, k)

            loss = loss + cfg.loss_weights["smoothness"] * smoothness_penalty(traj)
            loss = loss + cfg.loss_weights["l2"] * ((r**2).mean() + (alpha**2).mean() + (k**2).mean())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                r_va, alpha_va, k_va = model(x_va, constrained=constrained)
                t_va, traj_va = simulate_trajectories(
                    r_va,
                    alpha_va,
                    k_va,
                    ODESimConfig(
                        t_max_days=cfg.ode_t_max_days,
                        n_steps=cfg.ode_steps,
                        use_drug_term=(ablation != "no_drug_term"),
                    ),
                    dose_fn=make_dose_fn(),
                )
                pred_va = resistance_time_from_trajectory(t_va, traj_va)
                val_loss = torch.mean((pred_va - y_va) ** 2).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            constrained = ablation != "unconstrained_parameters"
            r_va, alpha_va, k_va = model(x_va, constrained=constrained)
            t_va, traj_va = simulate_trajectories(
                r_va,
                alpha_va,
                k_va,
                ODESimConfig(
                    t_max_days=cfg.ode_t_max_days,
                    n_steps=cfg.ode_steps,
                    use_drug_term=(ablation != "no_drug_term"),
                ),
                dose_fn=make_dose_fn(),
            )
            pred_va = resistance_time_from_trajectory(t_va, traj_va).cpu().numpy()

        y_true = y_va.cpu().numpy()
        fold_rmse = _rmse(y_true, pred_va)
        cidx = concordance_index(
            event_times=y_true,
            predicted_scores=-pred_va,
            event_observed=e_va.cpu().numpy(),
        )

        fold_metrics.append({
            "fold": fold,
            "rmse": fold_rmse,
            "mse": float(np.mean((y_true - pred_va) ** 2)),
            "c_index": float(cidx),
        })

        pred_fold = pd.DataFrame({
            "case_id": x_df.iloc[va_idx].index,
            "y_true_pfs_days": y_true,
            "y_pred_resistance_days": pred_va,
            "event": e_va.cpu().numpy(),
            "fold": fold,
            "model": "hybrid" if ablation is None else f"hybrid_{ablation}",
        })
        preds.append(pred_fold)

    return pd.concat(preds, ignore_index=True), fold_metrics
