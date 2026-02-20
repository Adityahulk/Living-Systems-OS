from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint


class ParameterNet(nn.Module):
    """Neural map from pathway features to tumor ODE parameters."""

    def __init__(self, in_dim: int, hidden_dims: list[int]) -> None:
        """Create MLP with two hidden layers and 5-parameter output head."""
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 5),
        )

    def forward(
        self, x: torch.Tensor, constrained: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict tumor parameters, optionally applying stability constraints."""
        raw = self.net(x)
        r_s_raw, r_r_raw, alpha_s_raw, k_raw, rho_raw = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
        if constrained:
            # Bounded constraints improve ODE stability during training.
            r_s = torch.exp(r_s_raw).clamp(1e-4, 0.25)
            r_r = torch.exp(r_r_raw).clamp(1e-4, 0.25)
            alpha_s = torch.sigmoid(alpha_s_raw).clamp(0.01, 0.99)
            k = torch.exp(k_raw).clamp(0.2, 15.0)
            # S -> R transition pressure under therapy.
            rho = torch.sigmoid(rho_raw).clamp(1e-5, 0.2)
        else:
            r_s, r_r, alpha_s, k, rho = r_s_raw, r_r_raw, alpha_s_raw, k_raw, rho_raw
        return r_s, r_r, alpha_s, k, rho


class TumorODEFunc(nn.Module):
    """Two-compartment tumor ODE dynamics under treatment exposure."""

    def __init__(self, dose_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Store dosing schedule function used during ODE evaluation."""
        super().__init__()
        self.dose_fn = dose_fn

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        r_s: torch.Tensor,
        r_r: torch.Tensor,
        alpha_s: torch.Tensor,
        k: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        """Return derivatives for sensitive and resistant tumor compartments."""
        # Two-compartment dynamics:
        # S(t): drug-sensitive tumor burden
        # R(t): drug-resistant tumor burden
        y = torch.clamp(y, min=0.0, max=1e4)
        s = y[0]
        r = y[1]
        v = s + r
        d_t = self.dose_fn(t)
        growth_limit = (1.0 - v / (k + 1e-8))
        ds = r_s * s * growth_limit - alpha_s * d_t * s - rho * d_t * s
        dr = r_r * r * growth_limit + rho * d_t * s
        return torch.stack([ds, dr], dim=0)


@dataclass
class ODESimConfig:
    """Configuration for ODE simulation horizon and resolution."""

    t_max_days: int
    n_steps: int
    use_drug_term: bool = True


def make_dose_fn(
    dose_scale: float = 1.0,
    start_day: float = 0.0,
    interruption_windows: list[tuple[float, float]] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a time-dependent dosing function with delays/interruptions."""
    interruption_windows = interruption_windows or []

    def dose_fn(t: torch.Tensor) -> torch.Tensor:
        """Evaluate whether drug is on/off at time t for this schedule."""
        t_val = t.item() if t.ndim == 0 else float(t.mean().item())
        on = 1.0 if t_val >= start_day else 0.0
        for a, b in interruption_windows:
            if a <= t_val <= b:
                on = 0.0
                break
        return torch.tensor(on * dose_scale, dtype=t.dtype, device=t.device)

    return dose_fn


def simulate_trajectories(
    r_s: torch.Tensor,
    alpha_s: torch.Tensor,
    k: torch.Tensor,
    ode_cfg: ODESimConfig,
    dose_fn: Callable[[torch.Tensor], torch.Tensor],
    r_r: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate patient-specific tumor trajectories from predicted parameters."""
    device = r_s.device
    t = torch.linspace(0, ode_cfg.t_max_days, ode_cfg.n_steps, device=device)

    if not ode_cfg.use_drug_term:
        dose_fn = make_dose_fn(dose_scale=0.0)

    if r_r is None:
        r_r = torch.clamp(0.6 * r_s, 1e-4, 0.25)
    if rho is None:
        rho = torch.full_like(r_s, 0.01)

    ode_func = TumorODEFunc(dose_fn)
    # Start with mostly sensitive tumor, small resistant fraction.
    s0 = torch.full_like(r_s, 0.98)
    r0 = torch.full_like(r_s, 0.02)

    trajectories = []
    for i in range(r_s.shape[0]):
        params = (r_s[i], r_r[i], alpha_s[i], k[i], rho[i])

        def f(tt: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
            """ODE RHS wrapper that closes over patient-specific parameters."""
            return ode_func(tt, yy, *params)

        y0 = torch.stack([s0[i], r0[i]], dim=0)
        y = odeint(f, y0, t, method="rk4")
        y = torch.nan_to_num(y, nan=1e4, posinf=1e4, neginf=0.0)
        total_v = y[:, 0] + y[:, 1]
        trajectories.append(total_v)

    traj = torch.stack(trajectories, dim=0)
    return t, traj


def resistance_time_from_trajectory(t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Estimate first post-nadir regrowth time from simulated trajectories."""
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
    growth_beta: float = 40.0,
    post_nadir_gamma: float = 0.05,
) -> torch.Tensor:
    """Differentiable approximation of resistance timing for gradient training."""
    # Soft nadir time via soft-argmin over tumor volume.
    w_nadir = torch.softmax(-nadir_beta * v, dim=1)
    t_nadir = torch.sum(w_nadir * t.unsqueeze(0), dim=1)  # [B]

    # Soft growth indicator over trajectory slopes.
    t_mid = 0.5 * (t[1:] + t[:-1])  # [T-1]
    dv = v[:, 1:] - v[:, :-1]  # [B, T-1]
    growth_prob = torch.sigmoid(growth_beta * dv)

    # Soft mask for times after nadir.
    post_mask = torch.sigmoid(post_nadir_gamma * (t_mid.unsqueeze(0) - t_nadir.unsqueeze(1)))
    signal = torch.clamp(growth_prob * post_mask, min=1e-8, max=1 - 1e-6)  # [B, T-1]

    # Differentiable first-hit approximation:
    # w_i = signal_i * prod_{j<i}(1-signal_j), then expectation over time.
    one_minus = 1.0 - signal
    surv_prev = torch.cumprod(one_minus, dim=1)
    surv_prev = torch.cat([torch.ones_like(surv_prev[:, :1]), surv_prev[:, :-1]], dim=1)
    w_first = signal * surv_prev

    sum_w = torch.sum(w_first, dim=1) + 1e-8
    expected_first = torch.sum(w_first * t_mid.unsqueeze(0), dim=1) / sum_w

    # If growth signal is weak, resistance is likely not observed within horizon.
    p_any_growth = 1.0 - torch.exp(-torch.sum(signal, dim=1))
    pred_t = p_any_growth * expected_first + (1.0 - p_any_growth) * t[-1]
    return pred_t


def smoothness_penalty(v: torch.Tensor) -> torch.Tensor:
    """Penalize high curvature to discourage unrealistically noisy trajectories."""
    d1 = v[:, 1:] - v[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return (d2**2).mean()


def plausibility_penalty(
    r_s: torch.Tensor,
    r_r: torch.Tensor,
    alpha_s: torch.Tensor,
    k: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Penalize parameter values that violate expected biological ranges."""
    # Soft regularization to keep parameters in realistic ranges.
    rs_pen = torch.relu(r_s - 0.2).mean() + torch.relu(0.002 - r_s).mean()
    rr_pen = torch.relu(r_r - 0.2).mean() + torch.relu(0.001 - r_r).mean()
    alpha_pen = torch.relu(alpha_s - 1.0).mean() + torch.relu(0.05 - alpha_s).mean()
    k_pen = torch.relu(0.8 - k).mean()
    rho_pen = torch.relu(rho - 0.2).mean() + torch.relu(1e-5 - rho).mean()
    # Encourage resistant cells to not be unrealistically fitter than sensitive cells.
    ordering_pen = torch.relu(r_r - (r_s + 0.05)).mean()
    return rs_pen + rr_pen + alpha_pen + k_pen + rho_pen + ordering_pen
