from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def load_sample_sheet(path: str | Path) -> pd.DataFrame:
    """Load and validate required columns from RNA sample sheet TSV."""
    df = pd.read_csv(path, sep="\t")
    required = {"file_path", "case_id", "sample_barcode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample sheet missing required columns: {sorted(missing)}")
    return df


def _load_augmented_star_counts(file_path: str | Path) -> pd.DataFrame:
    """Load one STAR count file and return per-gene TPM-ready table."""
    df = pd.read_csv(file_path, sep="\t", comment="#")

    if "gene_id" not in df.columns:
        raise ValueError(f"gene_id column missing in {file_path}")

    # Keep protein-coding features and remove STAR summary rows.
    df = df[~df["gene_id"].astype(str).str.startswith("N_")]

    cols = set(df.columns)
    if "tpm_unstranded" in cols:
        out = df[["gene_id", "gene_name", "tpm_unstranded"]].copy()
        out = out.rename(columns={"tpm_unstranded": "tpm"})
        return out

    if "unstranded" in cols and "gene_length" in cols:
        counts = pd.to_numeric(df["unstranded"], errors="coerce").fillna(0.0)
        lengths_kb = pd.to_numeric(df["gene_length"], errors="coerce").replace(0, np.nan) / 1000.0
        rpk = counts / lengths_kb
        scale = np.nansum(rpk) / 1e6
        tpm = rpk / scale
        out = df[["gene_id", "gene_name"]].copy()
        out["tpm"] = tpm.fillna(0.0)
        return out

    raise ValueError(
        f"Could not derive TPM from {file_path}; expected tpm_unstranded or unstranded+gene_length"
    )


def build_tpm_matrix(
    sample_sheet: pd.DataFrame,
    log_every: int = 25,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Build a genes x cases TPM matrix from all RNA files in sample sheet."""
    matrices: list[pd.DataFrame] = []
    n_total = len(sample_sheet)
    if n_total == 0:
        raise ValueError("Sample sheet is empty after filtering; cannot build TPM matrix")

    if logger is not None:
        logger(f"[Preprocess] Building TPM matrix from {n_total} RNA files...")

    for i, row in enumerate(sample_sheet.itertuples(index=False), start=1):
        fpath = Path(row.file_path)
        frame = _load_augmented_star_counts(fpath)
        frame = frame[["gene_name", "tpm"]].dropna()
        frame = frame.groupby("gene_name", as_index=False)["tpm"].sum()
        frame = frame.rename(columns={"tpm": row.case_id})
        matrices.append(frame)

        if logger is not None and (i == 1 or i == n_total or (log_every > 0 and i % log_every == 0)):
            logger(f"[Preprocess] TPM parsed {i}/{n_total} files")

    merged = matrices[0]
    for m in matrices[1:]:
        merged = merged.merge(m, on="gene_name", how="outer")

    merged = merged.fillna(0.0)
    merged = merged.set_index("gene_name")
    merged = merged.sort_index()
    if logger is not None:
        logger(f"[Preprocess] TPM matrix ready: genes={merged.shape[0]}, samples={merged.shape[1]}")
    return merged


def log_zscore_normalize(tpm_matrix: pd.DataFrame) -> pd.DataFrame:
    """Apply log2(TPM+1) then per-gene z-score normalization across cases."""
    x = np.log2(tpm_matrix + 1.0)
    mu = x.mean(axis=1)
    sigma = x.std(axis=1).replace(0, np.nan)
    z = x.sub(mu, axis=0).div(sigma, axis=0).fillna(0.0)
    return z


def filter_genes(
    tpm_matrix: pd.DataFrame,
    min_gene_tpm: float,
    min_detected_fraction: float,
    variance_threshold: float,
) -> pd.DataFrame:
    """Filter genes by detection frequency and expression variance thresholds."""
    detection_rate = (tpm_matrix > min_gene_tpm).mean(axis=1)
    keep_detected = detection_rate >= min_detected_fraction

    log_x = np.log2(tpm_matrix + 1.0)
    keep_var = log_x.var(axis=1) > variance_threshold

    filtered = tpm_matrix.loc[keep_detected & keep_var].copy()
    return filtered
