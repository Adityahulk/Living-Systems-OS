#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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
from lsos_research.evaluation import (
    bootstrap_ci,
    calibration_curve_data,
    paired_bootstrap_rmse_delta,
    rmse,
)
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
from lsos_research.training import _censor_aware_loss
from lsos_research.visualization import (
    save_ablation_barplot,
    save_calibration_plot,
    save_metric_boxplot,
    save_perturbation_plot,
)


def load_case_set(tsv_path: Path) -> set[str]:
    """Load case IDs from a TSV containing a `case_id` column."""
    if not tsv_path.exists():
        return set()
    df = pd.read_csv(tsv_path, sep="\t")
    if "case_id" not in df.columns:
        return set()
    return set(df["case_id"].dropna().astype(str))


def build_egfr_cohort_sources(cfg: dict, root: Path, maf_case_set: set[str]) -> tuple[set[str], pd.DataFrame]:
    """Combine EGFR case sources using configured strategy and report overlap."""
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
    """Parse CLI arguments for running the end-to-end research pipeline."""
    ap = argparse.ArgumentParser(description="Run LS-OS EGFR hybrid model pipeline")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--sample-sheet", default="data/interim/rna_sample_sheet_mapped.tsv")
    return ap.parse_args()


def _write_progress(
    results_dir: Path,
    name: str,
    preds: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> None:
    """Write intermediate predictions/metrics to progress directory."""
    prog = results_dir / "progress"
    prog.mkdir(parents=True, exist_ok=True)
    if preds is not None:
        preds.to_csv(prog / f"{name}_preds.tsv", sep="\t", index=False)
    if metrics is not None:
        metrics.to_csv(prog / f"{name}_metrics.tsv", sep="\t", index=False)


def _aligned_predictions(
    preds_all: pd.DataFrame,
    model_a: str,
    model_b: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two models' predictions on shared case/fold rows for paired stats."""
    a = preds_all.loc[preds_all["model"] == model_a, ["case_id", "fold", "y_true_pfs_days", "y_pred_resistance_days"]].copy()
    b = preds_all.loc[preds_all["model"] == model_b, ["case_id", "fold", "y_true_pfs_days", "y_pred_resistance_days"]].copy()
    m = a.merge(
        b,
        on=["case_id", "fold"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    if m.empty:
        return np.array([]), np.array([]), np.array([])
    y_true = m["y_true_pfs_days_a"].to_numpy()
    pred_a = m["y_pred_resistance_days_a"].to_numpy()
    pred_b = m["y_pred_resistance_days_b"].to_numpy()
    return y_true, pred_a, pred_b


def fit_final_hybrid(
    pathway_df: pd.DataFrame,
    y_days: pd.Series,
    y_event: pd.Series,
    cfg: TrainingConfig,
) -> tuple[ParameterNet, np.ndarray]:
    """Fit hybrid model on full cohort and return trained model + parameters."""
    x = torch.tensor(pathway_df.values, dtype=torch.float32)
    y = torch.tensor(y_days.values, dtype=torch.float32)
    e = torch.tensor(y_event.values, dtype=torch.float32)

    model = ParameterNet(in_dim=x.shape[1], hidden_dims=cfg.hidden_dims)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best = float("inf")
    stale = 0
    best_state = None

    for _ in range(cfg.epochs):
        model.train()
        r_s, r_r, alpha_s, k, rho = model(x, constrained=True)
        t, traj = simulate_trajectories(
            r_s,
            alpha_s,
            k,
            ODESimConfig(t_max_days=cfg.ode_t_max_days, n_steps=cfg.ode_steps),
            make_dose_fn(),
            r_r=r_r,
            rho=rho,
        )
        pred = differentiable_resistance_time_from_trajectory(t, traj)
        pred = torch.nan_to_num(
            pred,
            nan=float(cfg.ode_t_max_days),
            posinf=float(cfg.ode_t_max_days),
            neginf=0.0,
        )
        loss = _censor_aware_loss(pred, y, e, event_weight=3.0, censored_weight=0.3)
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
        r_s, r_r, alpha_s, k, rho = model(x, constrained=True)
    return model, np.column_stack([r_s.numpy(), r_r.numpy(), alpha_s.numpy(), k.numpy(), rho.numpy()])


def build_foldwise_feature_matrices(
    raw_tpm: pd.DataFrame,
    case_order: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
    pathways: dict[str, list[str]],
    preprocessing_cfg: dict,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Create fold-specific features using train-fold-only normalization/filtering."""
    tpm = raw_tpm.copy()
    if tpm.columns.duplicated().any():
        tpm = tpm.T.groupby(level=0).mean().T

    tpm = tpm.reindex(columns=case_order).fillna(0.0)
    log_all = np.log2(tpm + 1.0)
    min_gene_tpm = float(preprocessing_cfg["min_gene_tpm"])
    min_detected_fraction = float(preprocessing_cfg["min_detected_fraction"])
    variance_threshold = float(preprocessing_cfg["variance_threshold"])

    path_folds: list[pd.DataFrame] = []
    gene_folds: list[pd.DataFrame] = []
    for tr_idx, _ in splits:
        train_cases = [case_order[i] for i in tr_idx]
        tpm_tr = tpm.loc[:, train_cases]
        log_tr = np.log2(tpm_tr + 1.0)

        keep_detected = (tpm_tr > min_gene_tpm).mean(axis=1) >= min_detected_fraction
        keep_var = log_tr.var(axis=1) > variance_threshold
        keep = keep_detected & keep_var
        if int(keep.sum()) == 0:
            raise ValueError("Fold-wise preprocessing removed all genes; adjust filtering thresholds")

        mu = log_tr.loc[keep].mean(axis=1)
        sigma = log_tr.loc[keep].std(axis=1).replace(0.0, np.nan)
        z_all = log_all.loc[keep].sub(mu, axis=0).div(sigma, axis=0).fillna(0.0)

        x_path_fold = compute_pathway_activity(z_all, pathways).reindex(case_order).fillna(0.0)
        x_gene_fold = z_all.T.reindex(case_order).fillna(0.0)
        path_folds.append(x_path_fold)
        gene_folds.append(x_gene_fold)

    all_path_cols = sorted({c for df in path_folds for c in df.columns})
    path_folds = [df.reindex(columns=all_path_cols, fill_value=0.0) for df in path_folds]
    return path_folds, gene_folds


def main() -> None:
    """Run full LS-OS research workflow from ingestion through outputs."""
    args = parse_args()
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.root)
    ensure_dirs(paths)
    results_dir = Path(args.root) / cfg["paths"]["results_dir"]
    (results_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (results_dir / "figures").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
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
    print(f"[Phase 2] Loaded sample sheet rows: {len(sample_sheet)}", flush=True)
    sample_sheet = sample_sheet.drop_duplicates(subset=["case_id", "sample_barcode"])
    print(f"[Phase 2] After de-dup by (case_id, sample_barcode): {len(sample_sheet)}", flush=True)

    # Primary tumor filter.
    sample_sheet = sample_sheet.loc[sample_sheet["sample_barcode"].astype(str).str[13:15] == "01"].copy()
    print(f"[Phase 2] After primary tumor filter (code=01): {len(sample_sheet)}", flush=True)

    if egfr_cases:
        sample_sheet = sample_sheet.loc[sample_sheet["case_id"].isin(egfr_cases)].copy()
        print(f"[Phase 2] After EGFR cohort filter: {len(sample_sheet)}", flush=True)
    print(f"[Phase 2] RNA samples after filters: {len(sample_sheet)}", flush=True)

    sample_key = Path(args.sample_sheet).stem
    raw_tpm_cache = paths.processed_dir / f"tpm_raw_matrix.{sample_key}.tsv"
    tpm_cache = paths.processed_dir / f"tpm_matrix.{sample_key}.tsv"
    expr_cache = paths.processed_dir / f"expr_z.{sample_key}.tsv"
    pathway_cache = paths.processed_dir / f"pathway_activity.{sample_key}.tsv"

    if raw_tpm_cache.exists():
        print(f"[Phase 2] Loading raw TPM cache: {raw_tpm_cache}", flush=True)
        tpm_raw = pd.read_csv(raw_tpm_cache, sep="\t", index_col=0)
    else:
        tpm_raw = build_tpm_matrix(sample_sheet, log_every=25, logger=lambda m: print(m, flush=True))
        tpm_raw.to_csv(raw_tpm_cache, sep="\t")
        print("[Phase 2] Saved raw TPM cache", flush=True)
    print(f"[Phase 2] Raw TPM shape: {tpm_raw.shape}", flush=True)

    if tpm_cache.exists() and expr_cache.exists():
        print("[Phase 2] Loading cached TPM/Z matrices", flush=True)
        tpm = pd.read_csv(tpm_cache, sep="\t", index_col=0)
        expr_z = pd.read_csv(expr_cache, sep="\t", index_col=0)
    else:
        tpm = tpm_raw.copy()
        n_genes_pre = tpm.shape[0]
        tpm = filter_genes(
            tpm,
            min_gene_tpm=float(cfg["preprocessing"]["min_gene_tpm"]),
            min_detected_fraction=float(cfg["preprocessing"]["min_detected_fraction"]),
            variance_threshold=float(cfg["preprocessing"]["variance_threshold"]),
        )
        print(
            f"[Phase 2] Gene filter kept {tpm.shape[0]}/{n_genes_pre} genes "
            f"({(100.0 * tpm.shape[0] / max(1, n_genes_pre)):.1f}%)",
            flush=True,
        )
        expr_z = log_zscore_normalize(tpm)
        tpm.to_csv(tpm_cache, sep="\t")
        expr_z.to_csv(expr_cache, sep="\t")
        print("[Phase 2] Saved TPM/Z caches", flush=True)
    print(f"[Phase 2] Filtered TPM shape: {tpm.shape}, Z-shape: {expr_z.shape}", flush=True)

    pathways = load_pathway_gene_sets(Path(args.root) / cfg["paths"]["pathways_file"])
    print(f"[Phase 2] Loaded pathway sets: {len(pathways)}", flush=True)
    if pathway_cache.exists():
        print(f"[Phase 2] Loading pathway activity cache: {pathway_cache}", flush=True)
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
    x_path_folds, x_gene_folds = build_foldwise_feature_matrices(
        raw_tpm=tpm_raw,
        case_order=x_path.index.tolist(),
        splits=splits,
        pathways=pathways,
        preprocessing_cfg=cfg["preprocessing"],
    )
    print(
        f"[Phase 2] Fold-wise leakage-safe features ready: pathway_dim={x_path_folds[0].shape[1]}, "
        f"gene_dim={x_gene_folds[0].shape[1]}",
        flush=True,
    )

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
    t_hybrid = time.time()
    hybrid_preds, hybrid_metrics = train_hybrid_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        cfg=train_cfg,
        seed=int(cfg["project"]["seed"]),
        splits=splits,
        x_folds=x_path_folds,
        progress_dir=results_dir / "progress" / "hybrid_train",
    )
    _write_progress(results_dir, "hybrid", hybrid_preds, pd.DataFrame(hybrid_metrics))
    print(f"[Phase 4] Hybrid done in {time.time() - t_hybrid:.1f}s", flush=True)

    sat_overall = float(np.mean(hybrid_preds["y_pred_resistance_days"].to_numpy() >= (train_cfg.ode_t_max_days - 1e-6)))
    sat_threshold = float(cfg["training"].get("max_sat_fraction", 0.2))
    if sat_overall > sat_threshold:
        raise RuntimeError(
            f"Hybrid predictions saturated at ODE horizon: sat_fraction={sat_overall:.3f} > threshold={sat_threshold:.3f}"
        )
    pred_std = float(np.std(hybrid_preds["y_pred_resistance_days"].to_numpy()))
    unique_pred = int(np.unique(np.round(hybrid_preds["y_pred_resistance_days"].to_numpy(), 3)).size)
    min_pred_std = float(cfg["training"].get("min_pred_std", 5.0))
    min_unique_preds = int(cfg["training"].get("min_unique_preds", 8))
    if pred_std < min_pred_std or unique_pred < min_unique_preds:
        raise RuntimeError(
            f"Hybrid prediction collapse detected: std={pred_std:.3f}, unique={unique_pred} "
            f"(required std>={min_pred_std}, unique>={min_unique_preds})"
        )

    print("[Phase 5] Running baseline: Cox", flush=True)
    t0 = time.time()
    cox_preds, cox_metrics = run_cox_cv(
        x_path,
        data[["case_id", "pfs_days", "pfs_event"]],
        splits=splits,
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] Cox done in {time.time() - t0:.1f}s", flush=True)

    print("[Phase 5] Running baseline: Weibull AFT", flush=True)
    t0 = time.time()
    weibull_preds, weibull_metrics = run_aft_cv(
        x_path,
        data[["case_id", "pfs_days", "pfs_event"]],
        splits=splits,
        model_name="weibull_aft",
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] Weibull AFT done in {time.time() - t0:.1f}s", flush=True)

    print("[Phase 5] Running baseline: LogNormal AFT", flush=True)
    t0 = time.time()
    logn_preds, logn_metrics = run_aft_cv(
        x_path,
        data[["case_id", "pfs_days", "pfs_event"]],
        splits=splits,
        model_name="lognormal_aft",
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] LogNormal AFT done in {time.time() - t0:.1f}s", flush=True)

    print("[Phase 5] Running baseline: DeepSurv", flush=True)
    t0 = time.time()
    deepsurv_preds, deepsurv_metrics = run_deepsurv_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] DeepSurv done in {time.time() - t0:.1f}s", flush=True)

    print("[Phase 5] Running baseline: MLP", flush=True)
    t0 = time.time()
    mlp_preds, mlp_metrics = run_mlp_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] MLP done in {time.time() - t0:.1f}s", flush=True)

    print("[Phase 5] Running baseline: GradientBoosting", flush=True)
    t0 = time.time()
    gbr_preds, gbr_metrics = run_gbr_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] GradientBoosting done in {time.time() - t0:.1f}s", flush=True)

    print("[Phase 5] Running baseline: RandomForest", flush=True)
    t0 = time.time()
    rf_preds, rf_metrics = run_rf_cv(
        x_path,
        y_pfs,
        y_event=y_event,
        splits=splits,
        seed=int(cfg["project"]["seed"]),
        x_folds=x_path_folds,
    )
    print(f"[Phase 5] RandomForest done in {time.time() - t0:.1f}s", flush=True)
    _write_progress(results_dir, "cox", cox_preds, pd.DataFrame(cox_metrics))
    _write_progress(results_dir, "weibull_aft", weibull_preds, pd.DataFrame(weibull_metrics))
    _write_progress(results_dir, "lognormal_aft", logn_preds, pd.DataFrame(logn_metrics))
    _write_progress(results_dir, "deepsurv", deepsurv_preds, pd.DataFrame(deepsurv_metrics))
    _write_progress(results_dir, "mlp", mlp_preds, pd.DataFrame(mlp_metrics))
    _write_progress(results_dir, "gbr", gbr_preds, pd.DataFrame(gbr_metrics))
    _write_progress(results_dir, "rf", rf_preds, pd.DataFrame(rf_metrics))

    # Phase 7: Ablations.
    print("[Phase 7] Running ablations...", flush=True)
    t_ab = time.time()
    ab_preds, ab_metrics = run_ablation_suite(
        x_pathway=x_path,
        x_gene=x_gene,
        y_pfs=y_pfs,
        y_event=y_event,
        base_cfg=train_cfg,
        seed=int(cfg["project"]["seed"]),
        splits=splits,
        ablations=list(cfg["ablation"]["run"]),
        x_pathway_folds=x_path_folds,
        x_gene_folds=x_gene_folds,
        progress_root=str(results_dir / "progress"),
    )
    _write_progress(results_dir, "ablation", ab_preds, ab_metrics)
    print(f"[Phase 7] Ablations done in {time.time() - t_ab:.1f}s", flush=True)

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

    y_true_h_mlp, y_hat_h_mlp, y_hat_mlp = _aligned_predictions(preds_all, "hybrid", "mlp")
    y_true_h_cox, y_hat_h_cox, y_hat_cox = _aligned_predictions(preds_all, "hybrid", "cox")
    y_true_h_ds, y_hat_h_ds, y_hat_ds = _aligned_predictions(preds_all, "hybrid", "deepsurv")

    boot_hybrid_vs_mlp = paired_bootstrap_rmse_delta(
        y_true_h_mlp,
        y_hat_h_mlp,
        y_hat_mlp,
        n_bootstrap=int(cfg["training"]["bootstrap_iterations"]),
        seed=int(cfg["project"]["seed"]),
    ) if len(y_true_h_mlp) else {}
    boot_hybrid_vs_cox = paired_bootstrap_rmse_delta(
        y_true_h_cox,
        y_hat_h_cox,
        y_hat_cox,
        n_bootstrap=int(cfg["training"]["bootstrap_iterations"]),
        seed=int(cfg["project"]["seed"]),
    ) if len(y_true_h_cox) else {}
    boot_hybrid_vs_deepsurv = paired_bootstrap_rmse_delta(
        y_true_h_ds,
        y_hat_h_ds,
        y_hat_ds,
        n_bootstrap=int(cfg["training"]["bootstrap_iterations"]),
        seed=int(cfg["project"]["seed"]),
    ) if len(y_true_h_ds) else {}

    ci_hybrid_rmse = bootstrap_ci(
        y_true=hybrid_preds["y_true_pfs_days"].to_numpy(),
        y_pred=hybrid_preds["y_pred_resistance_days"].to_numpy(),
        metric_fn=rmse,
        n_bootstrap=int(cfg["training"]["bootstrap_iterations"]),
        seed=int(cfg["project"]["seed"]),
    )

    # Phase 6 perturbation scenarios using a final fit.
    print("[Phase 6] Running perturbation scenarios...", flush=True)
    t_pert = time.time()
    final_model, params = fit_final_hybrid(x_path, y_pfs, y_event, train_cfg)
    with torch.no_grad():
        x_t = torch.tensor(x_path.values, dtype=torch.float32)
        r_s, r_r, alpha_s, k, rho = final_model(x_t, constrained=True)

    perturb_df = simulate_perturbations(
        r_s,
        alpha_s,
        k,
        scenarios=cfg["perturbations"]["scenarios"],
        t_max_days=train_cfg.ode_t_max_days,
        n_steps=train_cfg.ode_steps,
        r_r=r_r,
        rho=rho,
    )
    print(f"[Phase 6] Perturbations done in {time.time() - t_pert:.1f}s", flush=True)

    preds_all.to_csv(results_dir / "tables" / "predictions_all.tsv", sep="\t", index=False)
    metrics_all.to_csv(results_dir / "metrics" / "fold_metrics.tsv", sep="\t", index=False)
    calib.to_csv(results_dir / "metrics" / "calibration_data.tsv", sep="\t", index=False)
    perturb_df.to_csv(results_dir / "metrics" / "perturbation_results.tsv", sep="\t", index=False)

    pd.DataFrame(params, columns=["r_sensitive", "r_resistant", "alpha_sensitive", "K", "rho_transition"], index=x_path.index).to_csv(
        results_dir / "tables" / "hybrid_parameters.tsv", sep="\t"
    )

    stats_summary = {
        "paired_bootstrap_hybrid_vs_mlp_rmse": boot_hybrid_vs_mlp,
        "paired_bootstrap_hybrid_vs_cox_rmse": boot_hybrid_vs_cox,
        "paired_bootstrap_hybrid_vs_deepsurv_rmse": boot_hybrid_vs_deepsurv,
        "bootstrap_hybrid_rmse": ci_hybrid_rmse,
        "hybrid_prediction_diagnostics": {
            "sat_tmax_fraction": sat_overall,
            "pred_std": pred_std,
            "unique_predictions_rounded_1e-3": unique_pred,
        },
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

    print("Pipeline completed successfully", flush=True)
    print(f"Cohort size: {len(x_path)}", flush=True)


if __name__ == "__main__":
    main()
