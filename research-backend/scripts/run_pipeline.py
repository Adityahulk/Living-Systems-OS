#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lsos_research.ablation import run_ablation_suite
from lsos_research.baselines import (
    run_aft_cv,
    run_cox_cv,
    run_deepsurv_cv,
    run_gbr_cv,
    run_mlp_cv,
    run_rf_cv,
)
from lsos_research.config import ensure_dirs, load_config, resolve_paths, set_global_seed
from lsos_research.cv import build_repeated_stratified_splits
from lsos_research.data_ingest import (
    build_egfr_mutant_case_set,
    build_survival_frame,
    discover_mutation_maf_files,
    extract_clinical_tarball,
    load_clinical_tables,
)
from lsos_research.evaluation import bootstrap_ci, calibration_curve_data, paired_t_test, rmse
from lsos_research.features import compute_pathway_activity, load_pathway_gene_sets
from lsos_research.hybrid_model import (
    ODESimConfig,
    ParameterNet,
    differentiable_resistance_time_from_trajectory,
    make_dose_fn,
    simulate_trajectories,
)
from lsos_research.perturbation import simulate_perturbations
from lsos_research.preprocessing import (
    filter_genes,
    load_sample_sheet,
    build_tpm_matrix,
    log_zscore_normalize,
)
from lsos_research.training import TrainingConfig, train_hybrid_cv
from lsos_research.visualization import (
    save_ablation_barplot,
    save_calibration_plot,
    save_metric_boxplot,
    save_perturbation_plot,
)


def load_case_set(tsv_path: Path) -> set[str]:
    if not tsv_path.exists():
        return set()
    df = pd.read_csv(tsv_path, sep="\t")
    if "case_id" not in df.columns:
        return set()
    return set(df["case_id"].dropna().astype(str))


def build_egfr_cohort_sources(cfg: dict, root: Path, maf_case_set: set[str]) -> tuple[set[str], pd.DataFrame]:
    sources: dict[str, set[str]] = {}
    if maf_case_set:
        sources["maf_local"] = maf_case_set

    primary = root / cfg["cohort"].get("public_egfr_case_list", "")
    secondary = root / cfg["cohort"].get("public_egfr_case_list_secondary", "")
    pset = load_case_set(primary)
    sset = load_case_set(secondary)
    if pset:
        sources["public_primary"] = pset
    if sset:
        sources["public_secondary"] = sset

    if not sources:
        return set(), pd.DataFrame(columns=["source_a", "source_b", "n_a", "n_b", "overlap", "jaccard"])

    strategy = cfg["cohort"].get("egfr_case_strategy", "intersection")
    names = list(sources.keys())
    if strategy == "intersection":
        selected = set.intersection(*[sources[n] for n in names])
    elif strategy == "union":
        selected = set.union(*[sources[n] for n in names])
    elif strategy == "primary" and "public_primary" in sources:
        selected = set(sources["public_primary"])
    else:
        selected = set.intersection(*[sources[n] for n in names])

    if not selected:
        raise ValueError("EGFR cohort selection is empty under configured strategy")

    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = sources[a], sources[b]
            ov = len(sa & sb)
            jac = ov / max(1, len(sa | sb))
            rows.append(
                {
                    "source_a": a,
                    "source_b": b,
                    "n_a": len(sa),
                    "n_b": len(sb),
                    "overlap": ov,
                    "jaccard": jac,
                }
            )

    return selected, pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run LS-OS EGFR hybrid model pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample-sheet", default="data/interim/rna_sample_sheet_mapped.tsv")
    return ap.parse_args()


def fit_final_hybrid(pathway_df: pd.DataFrame, y_days: pd.Series, cfg: TrainingConfig) -> tuple[ParameterNet, np.ndarray]:
    x = torch.tensor(pathway_df.values, dtype=torch.float32)
    y = torch.tensor(y_days.values, dtype=torch.float32)

    model = ParameterNet(in_dim=x.shape[1], hidden_dims=cfg.hidden_dims)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best = float("inf")
    stale = 0
    best_state = None

    for _ in range(cfg.epochs):
        model.train()
        r, alpha, k = model(x, constrained=True)
        t, traj = simulate_trajectories(
            r,
            alpha,
            k,
            ODESimConfig(t_max_days=cfg.ode_t_max_days, n_steps=cfg.ode_steps),
            make_dose_fn(),
        )
        pred = differentiable_resistance_time_from_trajectory(t, traj)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

        lv = loss.item()
        if lv < best:
            best = lv
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        r, alpha, k = model(x, constrained=True)
    return model, np.column_stack([r.numpy(), alpha.numpy(), k.numpy()])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.root)
    ensure_dirs(paths)
    set_global_seed(int(cfg["project"]["seed"]))
    print("[Phase 0] Config loaded and seeds set", flush=True)

    # Phase 1: Clinical ingestion.
    print("[Phase 1] Extracting and loading clinical tables...", flush=True)
    clinical_dir = paths.interim_dir / "clinical"
    extract_clinical_tarball(Path(args.root) / cfg["paths"]["clinical_tarball"], clinical_dir)
    clinical_tables = load_clinical_tables(clinical_dir)
    survival_df = build_survival_frame(clinical_tables)
    print(f"[Phase 1] Survival rows: {len(survival_df)}", flush=True)

    # EGFR cohort construction across available sources.
    maf_files = discover_mutation_maf_files(paths.raw_dir)
    maf_egfr = build_egfr_mutant_case_set(maf_files) if maf_files else set()
    egfr_cases, concordance_df = build_egfr_cohort_sources(cfg, Path(args.root), maf_egfr)
    print(f"[Phase 1] EGFR case set size ({cfg['cohort'].get('egfr_case_strategy', 'intersection')}): {len(egfr_cases)}", flush=True)

    # Phase 2: RNA preprocessing.
    print("[Phase 2] Building expression matrices...", flush=True)
    sample_sheet = load_sample_sheet(Path(args.root) / args.sample_sheet)
    sample_sheet = sample_sheet.drop_duplicates(subset=["case_id", "sample_barcode"])

    # Primary tumor filter.
    sample_sheet = sample_sheet.loc[sample_sheet["sample_barcode"].astype(str).str[13:15] == "01"].copy()

    if egfr_cases:
        sample_sheet = sample_sheet.loc[sample_sheet["case_id"].isin(egfr_cases)].copy()
    print(f"[Phase 2] RNA samples after filters: {len(sample_sheet)}", flush=True)

    sample_key = Path(args.sample_sheet).stem
    tpm_cache = paths.processed_dir / f"tpm_matrix.{sample_key}.tsv"
    expr_cache = paths.processed_dir / f"expr_z.{sample_key}.tsv"
    pathway_cache = paths.processed_dir / f"pathway_activity.{sample_key}.tsv"

    if tpm_cache.exists() and expr_cache.exists():
        print("[Phase 2] Loading cached TPM/Z matrices", flush=True)
        tpm = pd.read_csv(tpm_cache, sep="\t", index_col=0)
        expr_z = pd.read_csv(expr_cache, sep="\t", index_col=0)
    else:
        tpm = build_tpm_matrix(sample_sheet)
        tpm = filter_genes(
            tpm,
            min_gene_tpm=float(cfg["preprocessing"]["min_gene_tpm"]),
            min_detected_fraction=float(cfg["preprocessing"]["min_detected_fraction"]),
            variance_threshold=float(cfg["preprocessing"]["variance_threshold"]),
        )
        expr_z = log_zscore_normalize(tpm)
        tpm.to_csv(tpm_cache, sep="\t")
        expr_z.to_csv(expr_cache, sep="\t")
        print("[Phase 2] Saved TPM/Z caches", flush=True)

    pathways = load_pathway_gene_sets(Path(args.root) / cfg["paths"]["pathways_file"])
    if pathway_cache.exists():
        x_pathway = pd.read_csv(pathway_cache, sep="\t", index_col=0)
    else:
        x_pathway = compute_pathway_activity(expr_z, pathways)
        x_pathway.to_csv(pathway_cache, sep="\t")
    print(f"[Phase 2] Pathway matrix shape: {x_pathway.shape}", flush=True)

    # Gene-level matrix for ablation (patients x genes).
    x_gene = expr_z.T.copy()

    data = x_pathway.merge(
        survival_df[["case_id", "pfs_days", "pfs_event"]],
        left_index=True,
        right_on="case_id",
        how="inner",
    )
    data = data.dropna(subset=["pfs_days"])

    x_path = data.drop(columns=["case_id", "pfs_days", "pfs_event"]).set_index(data["case_id"])
    y_pfs = data.set_index("case_id")["pfs_days"]
    y_event = data.set_index("case_id")["pfs_event"]
    x_gene = x_gene.loc[x_path.index]
    print(f"[Phase 2] Final cohort size: {len(x_path)}", flush=True)
    print(f"[Phase 2] Event count: {int(y_event.sum())} / {len(y_event)}", flush=True)

    splits = build_repeated_stratified_splits(
        y_event=y_event.values,
        n_splits=int(cfg["training"]["n_splits"]),
        n_repeats=int(cfg["training"].get("n_repeats", 3)),
        seed=int(cfg["project"]["seed"]),
    )
    print(f"[Phase 2] CV splits: {len(splits)}", flush=True)

    train_cfg = TrainingConfig(
        hidden_dims=list(cfg["model"]["hidden_dims"]),
        learning_rate=float(cfg["model"]["learning_rate"]),
        weight_decay=float(cfg["model"]["weight_decay"]),
        epochs=int(cfg["model"]["epochs"]),
        patience=int(cfg["model"]["early_stopping_patience"]),
        n_splits=int(cfg["training"]["n_splits"]),
        ode_t_max_days=int(cfg["model"]["ode_t_max_days"]),
        ode_steps=int(cfg["model"]["ode_steps"]),
        loss_weights={
            "mse": float(cfg["loss"]["mse_weight"]),
            "l2": float(cfg["loss"]["param_l2_weight"]),
            "plausibility": float(cfg["loss"]["plausibility_weight"]),
            "smoothness": float(cfg["loss"]["smoothness_weight"]),
        },
    )

    # Phase 4/5: Hybrid + baselines.
    print("[Phase 4/5] Training hybrid and baselines...", flush=True)
    hybrid_preds, hybrid_metrics = train_hybrid_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        cfg=train_cfg,
        seed=int(cfg["project"]["seed"]),
        splits=splits,
    )
    cox_preds, cox_metrics = run_cox_cv(
        x_path,
        data[["case_id", "pfs_days", "pfs_event"]],
        splits=splits,
    )
    weibull_preds, weibull_metrics = run_aft_cv(
        x_path,
        data[["case_id", "pfs_days", "pfs_event"]],
        splits=splits,
        model_name="weibull_aft",
    )
    logn_preds, logn_metrics = run_aft_cv(
        x_path,
        data[["case_id", "pfs_days", "pfs_event"]],
        splits=splits,
        model_name="lognormal_aft",
    )
    deepsurv_preds, deepsurv_metrics = run_deepsurv_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
    )
    mlp_preds, mlp_metrics = run_mlp_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
    )
    gbr_preds, gbr_metrics = run_gbr_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
    )
    rf_preds, rf_metrics = run_rf_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
    )

    # Phase 7: Ablations.
    print("[Phase 7] Running ablations...", flush=True)
    ab_preds, ab_metrics = run_ablation_suite(
        x_pathway=x_path,
        x_gene=x_gene,
        y_pfs=y_pfs,
        y_event=y_event,
        base_cfg=train_cfg,
        seed=int(cfg["project"]["seed"]),
        splits=splits,
        ablations=list(cfg["ablation"]["run"]),
    )

    preds_all = pd.concat(
        [
            hybrid_preds,
            cox_preds,
            weibull_preds,
            logn_preds,
            deepsurv_preds,
            mlp_preds,
            gbr_preds,
            rf_preds,
            ab_preds,
        ],
        ignore_index=True,
    )

    metric_frames = []
    for name, mets in [
        ("hybrid", hybrid_metrics),
        ("cox", cox_metrics),
        ("weibull_aft", weibull_metrics),
        ("lognormal_aft", logn_metrics),
        ("deepsurv", deepsurv_metrics),
        ("mlp", mlp_metrics),
        ("gbr", gbr_metrics),
        ("rf", rf_metrics),
    ]:
        mdf = pd.DataFrame(mets)
        if "model" not in mdf.columns:
            mdf["model"] = name
        metric_frames.append(mdf)
    metric_frames.append(ab_metrics)
    metrics_all = pd.concat(metric_frames, ignore_index=True)

    # Phase 6/9: Stats + calibration + CI.
    print("[Phase 6/9] Computing statistics and calibration...", flush=True)
    calib = calibration_curve_data(
        y_true=hybrid_preds["y_true_pfs_days"].to_numpy(),
        y_pred=hybrid_preds["y_pred_resistance_days"].to_numpy(),
    )

    tt_hybrid_vs_mlp = paired_t_test(metrics_all, "hybrid", "mlp", "rmse")
    tt_hybrid_vs_cox = paired_t_test(metrics_all, "hybrid", "cox", "rmse")
    tt_hybrid_vs_deepsurv = paired_t_test(metrics_all, "hybrid", "deepsurv", "rmse")

    ci_hybrid_rmse = bootstrap_ci(
        y_true=hybrid_preds["y_true_pfs_days"].to_numpy(),
        y_pred=hybrid_preds["y_pred_resistance_days"].to_numpy(),
        metric_fn=rmse,
        n_bootstrap=int(cfg["training"]["bootstrap_iterations"]),
        seed=int(cfg["project"]["seed"]),
    )

    # Phase 6 perturbation scenarios using a final fit.
    print("[Phase 6] Running perturbation scenarios...", flush=True)
    final_model, params = fit_final_hybrid(x_path, y_pfs, train_cfg)
    with torch.no_grad():
        x_t = torch.tensor(x_path.values, dtype=torch.float32)
        r, alpha, k = final_model(x_t, constrained=True)

    perturb_df = simulate_perturbations(
        r,
        alpha,
        k,
        scenarios=cfg["perturbations"]["scenarios"],
        t_max_days=train_cfg.ode_t_max_days,
        n_steps=train_cfg.ode_steps,
    )

    # Outputs.
    results_dir = Path(args.root) / cfg["paths"]["results_dir"]
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)

    preds_all.to_csv(results_dir / "tables" / "predictions_all.tsv", sep="\t", index=False)
    metrics_all.to_csv(results_dir / "metrics" / "fold_metrics.tsv", sep="\t", index=False)
    calib.to_csv(results_dir / "metrics" / "calibration_data.tsv", sep="\t", index=False)
    perturb_df.to_csv(results_dir / "metrics" / "perturbation_results.tsv", sep="\t", index=False)

    pd.DataFrame(params, columns=["r", "alpha", "K"], index=x_path.index).to_csv(
        results_dir / "tables" / "hybrid_parameters.tsv", sep="\t"
    )

    stats_summary = {
        "paired_t_hybrid_vs_mlp_rmse": tt_hybrid_vs_mlp,
        "paired_t_hybrid_vs_cox_rmse": tt_hybrid_vs_cox,
        "paired_t_hybrid_vs_deepsurv_rmse": tt_hybrid_vs_deepsurv,
        "bootstrap_hybrid_rmse": ci_hybrid_rmse,
    }
    (results_dir / "metrics" / "statistical_tests.json").write_text(
        json.dumps(stats_summary, indent=2), encoding="utf-8"
    )

    torch.save(final_model.state_dict(), results_dir / "models" / "hybrid_final.pt")
    concordance_df.to_csv(results_dir / "tables" / "egfr_source_concordance.tsv", sep="\t", index=False)

    save_metric_boxplot(metrics_all, results_dir / "figures" / "rmse_boxplot.png", metric="rmse")
    save_calibration_plot(calib, results_dir / "figures" / "hybrid_calibration.png")
    save_ablation_barplot(
        metrics_all.loc[metrics_all["model"].str.contains("hybrid")].copy(),
        results_dir / "figures" / "ablation_rmse.png",
    )
    save_perturbation_plot(perturb_df, results_dir / "figures" / "perturbation_robustness.png")

    print("Pipeline completed successfully")
    print(f"Cohort size: {len(x_path)}")


if __name__ == "__main__":
    main()
