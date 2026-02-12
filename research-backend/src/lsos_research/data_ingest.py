from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pandas as pd


CASE_BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")


def read_gdc_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def filter_manifest_for_core_files(manifest: pd.DataFrame) -> pd.DataFrame:
    targets = manifest["filename"].str.contains(
        r"augmented_star_gene_counts\.tsv$|clinical\.|wxs\..*maf\.gz$|aliquot_ensemble_raw\.maf\.gz$",
        case=False,
        regex=True,
    )
    return manifest.loc[targets].copy()


def extract_clinical_tarball(clinical_tarball: str | Path, out_dir: str | Path) -> list[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with tarfile.open(clinical_tarball, "r:gz") as tf:
        tf.extractall(path=out_path)

    return sorted(out_path.glob("*.tsv"))


def load_clinical_tables(clinical_dir: str | Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for tsv in Path(clinical_dir).glob("*.tsv"):
        tables[tsv.stem] = pd.read_csv(tsv, sep="\t")
    return tables


def _to_case_barcode(x: str) -> str | None:
    if not isinstance(x, str):
        return None
    match = CASE_BARCODE_RE.search(x)
    if not match:
        return None
    return match.group(0)


def build_survival_frame(clinical_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    clinical = clinical_tables.get("clinical")
    if clinical is None:
        raise FileNotFoundError("clinical.tsv not found in extracted clinical tables")

    df = clinical.copy()
    id_col = None
    for c in ["case_submitter_id", "submitter_id", "cases.submitter_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError("No recognized case identifier column in clinical.tsv")
    df["case_id"] = df[id_col].map(_to_case_barcode)

    death_col = None
    for c in ["days_to_death", "demographic.days_to_death"]:
        if c in df.columns:
            death_col = c
            break
    if death_col is None:
        df["__days_to_death"] = pd.NA
        death_col = "__days_to_death"

    lfu_col = None
    for c in ["days_to_last_follow_up", "diagnoses.days_to_last_follow_up"]:
        if c in df.columns:
            lfu_col = c
            break
    if lfu_col is None:
        df["__days_to_last_follow_up"] = pd.NA
        lfu_col = "__days_to_last_follow_up"

    death = pd.to_numeric(df[death_col], errors="coerce")
    lfu = pd.to_numeric(df[lfu_col], errors="coerce")

    os_time = death.fillna(lfu)
    os_event = death.notna().astype(int)

    out = pd.DataFrame(
        {
            "case_id": df["case_id"],
            "os_days": os_time,
            "os_event": os_event,
            "lfu_days": lfu,
        }
    ).dropna(subset=["case_id"])

    follow_up = clinical_tables.get("follow_up")
    if follow_up is None:
        raise ValueError("follow_up.tsv is required for strict progression-based PFS endpoint")

    fu = follow_up.copy()
    fu_id_col = None
    for c in ["case_submitter_id", "submitter_id", "cases.submitter_id"]:
        if c in fu.columns:
            fu_id_col = c
            break
    if fu_id_col is None:
        raise ValueError("No recognized case identifier column in follow_up.tsv")
    fu["case_id"] = fu[fu_id_col].map(_to_case_barcode)

    progress_time_col = None
    for c in [
        "days_to_progression",
        "follow_ups.days_to_progression",
        "days_to_new_tumor_event_after_initial_treatment",
        "days_to_tumor_progression",
    ]:
        if c in fu.columns:
            progress_time_col = c
            break
    if progress_time_col is None:
        raise ValueError("No progression timing column found in follow_up.tsv")

    p = fu[["case_id", progress_time_col]].copy()
    p["progress_days"] = pd.to_numeric(p[progress_time_col], errors="coerce")
    p = p.groupby("case_id", as_index=False)["progress_days"].min()
    p["pfs_event"] = p["progress_days"].notna().astype(int)

    out = out.merge(p, on="case_id", how="left")
    out["pfs_event"] = out["pfs_event"].fillna(0).astype(int)
    out["pfs_days"] = out["progress_days"]
    out.loc[out["pfs_event"] == 0, "pfs_days"] = out.loc[out["pfs_event"] == 0, "lfu_days"]

    out = (
        out.groupby("case_id", as_index=False)
        .agg(
            os_days=("os_days", "min"),
            os_event=("os_event", "max"),
            pfs_days=("pfs_days", "min"),
            pfs_event=("pfs_event", "max"),
        )
        .dropna(subset=["pfs_days"])
    )
    out["os_days"] = out["os_days"].clip(lower=1.0)
    out["pfs_days"] = out["pfs_days"].clip(lower=1.0)
    return out


def discover_mutation_maf_files(search_dir: str | Path) -> list[Path]:
    root = Path(search_dir)
    return sorted(root.rglob("*.maf.gz"))


def build_egfr_mutant_case_set(maf_files: list[Path]) -> set[str]:
    case_ids: set[str] = set()

    for maf in maf_files:
        try:
            mdf = pd.read_csv(maf, sep="\t", comment="#", low_memory=False)
        except Exception:
            continue

        if "Hugo_Symbol" not in mdf.columns or "Tumor_Sample_Barcode" not in mdf.columns:
            continue

        egfr_rows = mdf.loc[mdf["Hugo_Symbol"].astype(str).str.upper() == "EGFR"]
        for barcode in egfr_rows["Tumor_Sample_Barcode"].astype(str):
            case = _to_case_barcode(barcode)
            if case:
                case_ids.add(case)

    return case_ids


def infer_primary_tumor(case_sample_barcode: str) -> bool:
    # TCGA aliquot barcodes include sample type code in positions 14-15.
    # Example: TCGA-XX-YYYY-01A... where 01 = Primary Solid Tumor.
    if not isinstance(case_sample_barcode, str) or len(case_sample_barcode) < 15:
        return False
    return case_sample_barcode[13:15] == "01"
