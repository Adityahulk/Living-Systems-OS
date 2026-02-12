from __future__ import annotations

import pandas as pd
import torch

from .hybrid_model import ODESimConfig, make_dose_fn, resistance_time_from_trajectory, simulate_trajectories


def simulate_perturbations(
    r: torch.Tensor,
    alpha: torch.Tensor,
    k: torch.Tensor,
    scenarios: dict,
    t_max_days: int,
    n_steps: int,
) -> pd.DataFrame:
    rows = []
    for name, cfg in scenarios.items():
        windows = [tuple(x) for x in cfg.get("interruption_windows", [])]
        dose_fn = make_dose_fn(
            dose_scale=cfg.get("dose_scale", 1.0),
            start_day=cfg.get("start_day", 0.0),
            interruption_windows=windows,
        )
        t, traj = simulate_trajectories(
            r,
            alpha,
            k,
            ODESimConfig(t_max_days=t_max_days, n_steps=n_steps, use_drug_term=True),
            dose_fn=dose_fn,
        )
        res_t = resistance_time_from_trajectory(t, traj).detach().cpu().numpy()
        for i, rt in enumerate(res_t):
            rows.append({"idx": i, "scenario": name, "pred_resistance_days": float(rt)})

    return pd.DataFrame(rows)
