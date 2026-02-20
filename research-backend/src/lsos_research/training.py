from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

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
    simulate_trajectories,
    smoothness_penalty,
)


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime controls for hybrid model training."""

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
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _censor_aware_loss(
    pred_time: torch.Tensor,
    y_time: torch.Tensor,
    y_event: torch.Tensor,
    event_weight: float = 3.0,
    censored_weight: float = 0.3,
) -> torch.Tensor:
    """Compute event-weighted loss that handles right-censored targets."""
    event_mask = y_event > 0.5
    censored_mask = ~event_mask

    if torch.any(event_mask):
        event_loss = torch.mean((pred_time[event_mask] - y_time[event_mask]) ** 2)
    else:
        event_loss = torch.tensor(0.0, dtype=pred_time.dtype, device=pred_time.device)

    # For censored cases, penalize only if prediction occurs before censoring time.
    if torch.any(censored_mask):
        early_error = torch.relu(y_time[censored_mask] - pred_time[censored_mask])
        censored_loss = torch.mean(early_error**2)
    else:
        censored_loss = torch.tensor(0.0, dtype=pred_time.dtype, device=pred_time.device)

    return event_weight * event_loss + censored_weight * censored_loss


def train_hybrid_cv(
    x_df: pd.DataFrame,
    y_pfs: pd.Series,
    y_event: pd.Series,
    cfg: TrainingConfig,
    seed: int,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ablation: str | None = None,
    x_folds: list[pd.DataFrame] | None = None,
    progress_dir: str | Path | None = None,
    checkpoint_every: int = 10,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Train hybrid model across CV folds and return OOF predictions + metrics."""
    y = torch.tensor(y_pfs.values, dtype=torch.float32)

    if splits is None:
        raise ValueError("train_hybrid_cv requires explicit CV splits")
    if x_folds is not None and len(x_folds) != len(splits):
        raise ValueError("x_folds length must match number of CV splits")
    fold_metrics: list[dict[str, float]] = []
    preds: list[pd.DataFrame] = []

    y_event_t = torch.tensor(y_event.values, dtype=torch.float32)
    base_index = x_df.index
    progress_path = Path(progress_dir) if progress_dir is not None else None
    if progress_path is not None:
        progress_path.mkdir(parents=True, exist_ok=True)

    print(
        f"[Hybrid] Starting CV training: folds={len(splits)}, epochs={cfg.epochs}, patience={cfg.patience}",
        flush=True,
    )

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        x_fold_df = x_folds[fold - 1] if x_folds is not None else x_df
        x_fold_df = x_fold_df.reindex(base_index)
        x = torch.tensor(x_fold_df.values, dtype=torch.float32)
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
        e_tr = y_event_t[tr_idx]
        x_va, y_va = x[va_idx], y[va_idx]
        e_va = y_event_t[va_idx]
        print(
            (
                f"[Hybrid][Fold {fold}/{len(splits)}] train_n={len(tr_idx)} "
                f"val_n={len(va_idx)} events_train={int(e_tr.sum().item())} "
                f"events_val={int(e_va.sum().item())}"
            ),
            flush=True,
        )

        best_val = float("inf")
        best_state = None
        stale = 0
        fold_t0 = time.time()
        epoch_log_every = max(1, cfg.epochs // 10)

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            constrained = ablation != "unconstrained_parameters"
            r_s, r_r, alpha_s, k, rho = model(x_tr, constrained=constrained)

            t, traj = simulate_trajectories(
                r_s,
                alpha_s,
                k,
                ODESimConfig(
                    t_max_days=cfg.ode_t_max_days,
                    n_steps=cfg.ode_steps,
                    use_drug_term=(ablation != "no_drug_term"),
                ),
                dose_fn=make_dose_fn(),
                r_r=r_r,
                rho=rho,
            )

            pred_res_t_soft = differentiable_resistance_time_from_trajectory(t, traj)
            loss = cfg.loss_weights["mse"] * _censor_aware_loss(
                pred_res_t_soft,
                y_tr,
                e_tr,
                event_weight=3.0,
                censored_weight=0.3,
            )

            if ablation != "unconstrained_parameters":
                loss = loss + cfg.loss_weights["plausibility"] * plausibility_penalty(r_s, r_r, alpha_s, k, rho)

            loss = loss + cfg.loss_weights["smoothness"] * smoothness_penalty(traj)
            loss = loss + cfg.loss_weights["l2"] * (
                (r_s**2).mean() + (r_r**2).mean() + (alpha_s**2).mean() + (k**2).mean() + (rho**2).mean()
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                r_va_s, r_va_r, alpha_va_s, k_va, rho_va = model(x_va, constrained=constrained)
                t_va, traj_va = simulate_trajectories(
                    r_va_s,
                    alpha_va_s,
                    k_va,
                    ODESimConfig(
                        t_max_days=cfg.ode_t_max_days,
                        n_steps=cfg.ode_steps,
                        use_drug_term=(ablation != "no_drug_term"),
                    ),
                    dose_fn=make_dose_fn(),
                    r_r=r_va_r,
                    rho=rho_va,
                )
                pred_va = differentiable_resistance_time_from_trajectory(t_va, traj_va)
                pred_va = torch.nan_to_num(
                    pred_va,
                    nan=float(cfg.ode_t_max_days),
                    posinf=float(cfg.ode_t_max_days),
                    neginf=0.0,
                )
                val_loss = _censor_aware_loss(
                    pred_va,
                    y_va,
                    e_va,
                    event_weight=3.0,
                    censored_weight=0.3,
                ).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

            if epoch == 1 or epoch % epoch_log_every == 0:
                print(
                    (
                        f"[Hybrid][Fold {fold}] epoch={epoch}/{cfg.epochs} "
                        f"train_loss={float(loss.item()):.4f} val_loss={float(val_loss):.4f} "
                        f"best_val={float(best_val):.4f} stale={stale}"
                    ),
                    flush=True,
                )

            if progress_path is not None and (epoch == 1 or epoch % checkpoint_every == 0):
                checkpoint = {
                    "fold": fold,
                    "epoch": epoch,
                    "elapsed_sec": round(time.time() - fold_t0, 3),
                    "train_loss": float(loss.item()),
                    "val_loss": float(val_loss),
                    "best_val": float(best_val),
                    "stale": int(stale),
                    "ablation": ablation or "hybrid",
                }
                (progress_path / f"fold_{fold:02d}.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            constrained = ablation != "unconstrained_parameters"
            r_va_s, r_va_r, alpha_va_s, k_va, rho_va = model(x_va, constrained=constrained)
            t_va, traj_va = simulate_trajectories(
                r_va_s,
                alpha_va_s,
                k_va,
                ODESimConfig(
                    t_max_days=cfg.ode_t_max_days,
                    n_steps=cfg.ode_steps,
                    use_drug_term=(ablation != "no_drug_term"),
                ),
                dose_fn=make_dose_fn(),
                r_r=r_va_r,
                rho=rho_va,
            )
            pred_va = differentiable_resistance_time_from_trajectory(t_va, traj_va)
            pred_va = torch.nan_to_num(
                pred_va,
                nan=float(cfg.ode_t_max_days),
                posinf=float(cfg.ode_t_max_days),
                neginf=0.0,
            ).cpu().numpy()

        y_true = y_va.cpu().numpy()
        fold_rmse = _rmse(y_true, pred_va)
        sat_frac = float(np.mean(pred_va >= (cfg.ode_t_max_days - 1e-6)))
        try:
            cidx = concordance_index(
                event_times=y_true,
                predicted_scores=-pred_va,
                event_observed=e_va.cpu().numpy(),
            )
        except ZeroDivisionError:
            cidx = float("nan")

        fold_metrics.append({
            "fold": fold,
            "rmse": fold_rmse,
            "mse": float(np.mean((y_true - pred_va) ** 2)),
            "c_index": float(cidx),
            "sat_tmax_frac": sat_frac,
        })
        print(
            (
                f"[Hybrid][Fold {fold}] done in {time.time() - fold_t0:.1f}s "
                f"rmse={fold_rmse:.3f} c_index={float(cidx):.3f} sat_frac={sat_frac:.3f}"
            ),
            flush=True,
        )

        pred_fold = pd.DataFrame({
            "case_id": x_df.iloc[va_idx].index,
            "y_true_pfs_days": y_true,
            "y_pred_resistance_days": pred_va,
            "event": e_va.cpu().numpy(),
            "fold": fold,
            "model": "hybrid" if ablation is None else f"hybrid_{ablation}",
        })
        preds.append(pred_fold)

        if progress_path is not None:
            pd.DataFrame([fold_metrics[-1]]).to_csv(progress_path / f"fold_{fold:02d}_metrics.tsv", sep="\t", index=False)
            pred_fold.to_csv(progress_path / f"fold_{fold:02d}_preds.tsv", sep="\t", index=False)

    print("[Hybrid] CV training complete", flush=True)
    return pd.concat(preds, ignore_index=True), fold_metrics
