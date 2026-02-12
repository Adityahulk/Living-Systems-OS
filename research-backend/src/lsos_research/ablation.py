from __future__ import annotations

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
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    x = x_df.values
    y = y_days.values
    e = y_event.values
    preds = []
    metrics = []

    for fold, (tr, va) in enumerate(splits, start=1):
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
                    "case_id": x_df.iloc[va].index,
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            )
        elif ab == "no_pathway_compression":
            preds, mets = train_hybrid_cv(
                x_df=x_gene,
                y_pfs=y_pfs,
                y_event=y_event,
                cfg=base_cfg,
                seed=seed,
                splits=splits,
                ablation=None,
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
            )

        all_preds.append(preds)
        mdf = pd.DataFrame(mets)
        mdf["model"] = preds["model"].iloc[0]
        all_metrics.append(mdf)

    return pd.concat(all_preds, ignore_index=True), pd.concat(all_metrics, ignore_index=True)
