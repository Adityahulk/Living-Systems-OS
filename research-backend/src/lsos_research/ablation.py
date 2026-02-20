from __future__ import annotations

from dataclasses import replace

import pandas as pd
from lifelines.utils import concordance_index
from sklearn.neural_network import MLPRegressor

from .training import TrainingConfig, train_hybrid_cv


def run_no_ode_ablation(
    x_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    seed: int,
    splits: list[tuple],
    x_folds: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Ablation baseline that removes ODE and predicts time directly with MLP."""
    y = y_days.values
    e = y_event.values
    preds = []
    metrics = []

    for fold, (tr, va) in enumerate(splits, start=1):
        x_curr = x_folds[fold - 1] if x_folds is not None else x_df
        x = x_curr.values
        model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=400, random_state=seed + fold)
        model.fit(x[tr], y[tr])
        y_hat = model.predict(x[va])

        rmse = float(((y_hat - y[va]) ** 2).mean() ** 0.5)
        cidx = concordance_index(y[va], -y_hat, e[va])
        metrics.append(
            {"fold": fold, "rmse": rmse, "mse": float(((y_hat - y[va]) ** 2).mean()), "c_index": float(cidx)}
        )

        preds.append(
            pd.DataFrame(
                {
                    "case_id": x_curr.iloc[va].index,
                    "y_true_pfs_days": y[va],
                    "y_pred_resistance_days": y_hat,
                    "event": e[va],
                    "fold": fold,
                    "model": "hybrid_no_ode",
                }
            )
        )

    return pd.concat(preds, ignore_index=True), metrics


def run_ablation_suite(
    x_pathway: pd.DataFrame,
    x_gene: pd.DataFrame,
    y_pfs: pd.Series,
    y_event: pd.Series,
    base_cfg: TrainingConfig,
    seed: int,
    splits: list[tuple],
    ablations: list[str],
    x_pathway_folds: list[pd.DataFrame] | None = None,
    x_gene_folds: list[pd.DataFrame] | None = None,
    progress_root: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all requested ablations and combine predictions/metrics."""
    all_preds = []
    all_metrics = []

    for ab in ablations:
        if ab == "no_ode":
            preds, mets = run_no_ode_ablation(
                x_df=x_pathway,
                y_days=y_pfs,
                y_event=y_event,
                seed=seed,
                splits=splits,
                x_folds=x_pathway_folds,
            )
        elif ab == "no_pathway_compression":
            # Increase capacity for high-dimensional raw-gene input to keep ablation fair.
            gene_in_dim = x_gene.shape[1]
            h1 = int(min(256, max(64, (gene_in_dim ** 0.5) * 4)))
            h2 = int(min(128, max(32, h1 // 2)))
            gene_cfg = replace(base_cfg, hidden_dims=[h1, h2], weight_decay=max(base_cfg.weight_decay, 5e-4))
            preds, mets = train_hybrid_cv(
                x_df=x_gene,
                y_pfs=y_pfs,
                y_event=y_event,
                cfg=gene_cfg,
                seed=seed,
                splits=splits,
                ablation=None,
                x_folds=x_gene_folds,
                progress_dir=(f"{progress_root}/ablation_no_pathway_compression" if progress_root else None),
            )
            preds["model"] = "hybrid_no_pathway_compression"
        else:
            preds, mets = train_hybrid_cv(
                x_df=x_pathway,
                y_pfs=y_pfs,
                y_event=y_event,
                cfg=base_cfg,
                seed=seed,
                splits=splits,
                ablation=ab,
                x_folds=x_pathway_folds,
                progress_dir=(f"{progress_root}/ablation_{ab}" if progress_root else None),
            )

        all_preds.append(preds)
        mdf = pd.DataFrame(mets)
        mdf["model"] = preds["model"].iloc[0]
        all_metrics.append(mdf)

    return pd.concat(all_preds, ignore_index=True), pd.concat(all_metrics, ignore_index=True)
