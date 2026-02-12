from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


class ParameterNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 3),
        )

    def forward(self, x: torch.Tensor, constrained: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.net(x)
        r_raw, alpha_raw, k_raw = raw[:, 0], raw[:, 1], raw[:, 2]
        if constrained:
            r = torch.exp(r_raw)
            alpha = torch.sigmoid(alpha_raw)
            k = torch.exp(k_raw)
        else:
            r, alpha, k = r_raw, alpha_raw, k_raw
        return r, alpha, k


class TumorODEFunc(nn.Module):
    def __init__(self, dose_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.dose_fn = dose_fn

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        r: torch.Tensor,
        alpha: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        v = y
        d_t = self.dose_fn(t)
        return r * v * (1.0 - v / (k + 1e-8)) - alpha * d_t * v


@dataclass
class ODESimConfig:
    t_max_days: int
    n_steps: int
    use_drug_term: bool = True


def make_dose_fn(
    dose_scale: float = 1.0,
    start_day: float = 0.0,
    interruption_windows: list[tuple[float, float]] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    interruption_windows = interruption_windows or []

    def dose_fn(t: torch.Tensor) -> torch.Tensor:
        t_val = t.item() if t.ndim == 0 else float(t.mean().item())
        on = 1.0 if t_val >= start_day else 0.0
        for a, b in interruption_windows:
            if a <= t_val <= b:
                on = 0.0
                break
        return torch.tensor(on * dose_scale, dtype=t.dtype, device=t.device)

    return dose_fn


def simulate_trajectories(
    r: torch.Tensor,
    alpha: torch.Tensor,
    k: torch.Tensor,
    ode_cfg: ODESimConfig,
    dose_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = r.device
    t = torch.linspace(0, ode_cfg.t_max_days, ode_cfg.n_steps, device=device)

    if not ode_cfg.use_drug_term:
        dose_fn = make_dose_fn(dose_scale=0.0)

    ode_func = TumorODEFunc(dose_fn)
    v0 = torch.ones_like(r)

    trajectories = []
    for i in range(r.shape[0]):
        params = (r[i], alpha[i], k[i])

        def f(tt: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
            return ode_func(tt, yy, *params)

        y = odeint(f, v0[i : i + 1], t, method="rk4")
        trajectories.append(y.squeeze(-1).squeeze(-1))

    traj = torch.stack(trajectories, dim=0)
    return t, traj


def resistance_time_from_trajectory(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Resistance proxy: first time where tumor starts increasing after reaching minimum.
    min_idx = torch.argmin(v, dim=1)
    out = []
    for i in range(v.shape[0]):
        idx0 = int(min_idx[i].item())
        diffs = v[i, idx0 + 1 :] - v[i, idx0:-1]
        pos = torch.where(diffs > 0)[0]
        if len(pos) == 0:
            out.append(t[-1])
        else:
            out.append(t[idx0 + 1 + pos[0]])
    return torch.stack(out)


def differentiable_resistance_time_from_trajectory(
    t: torch.Tensor,
    v: torch.Tensor,
    nadir_beta: float = 12.0,
    growth_beta: float = 30.0,
    post_nadir_gamma: float = 0.05,
) -> torch.Tensor:
    # Soft nadir time via soft-argmin over tumor volume.
    w_nadir = torch.softmax(-nadir_beta * v, dim=1)
    t_nadir = torch.sum(w_nadir * t.unsqueeze(0), dim=1)  # [B]

    # Soft growth indicator over trajectory slopes.
    t_mid = 0.5 * (t[1:] + t[:-1])  # [T-1]
    dv = v[:, 1:] - v[:, :-1]  # [B, T-1]
    growth_prob = torch.sigmoid(growth_beta * dv)

    # Soft mask for times after nadir.
    post_mask = torch.sigmoid(post_nadir_gamma * (t_mid.unsqueeze(0) - t_nadir.unsqueeze(1)))
    score = growth_prob * post_mask + 1e-8

    pred_t = torch.sum(score * t_mid.unsqueeze(0), dim=1) / torch.sum(score, dim=1)
    return pred_t


def smoothness_penalty(v: torch.Tensor) -> torch.Tensor:
    d1 = v[:, 1:] - v[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return (d2**2).mean()


def plausibility_penalty(r: torch.Tensor, alpha: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # Soft regularization to keep parameters in realistic ranges.
    r_pen = torch.relu(r - 0.08).mean() + torch.relu(0.002 - r).mean()
    alpha_pen = torch.relu(alpha - 1.0).mean() + torch.relu(0.05 - alpha).mean()
    k_pen = torch.relu(0.8 - k).mean()
    return r_pen + alpha_pen + k_pen
