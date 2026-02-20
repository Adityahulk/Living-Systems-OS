from __future__ import annotations

import re
import tarfile
from pathlib import Path

import pandas as pd


CASE_BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")


def read_gdc_manifest(path: str | Path) -> pd.DataFrame:
    """Read a GDC manifest TSV file."""
    return pd.read_csv(path, sep="\t")


def filter_manifest_for_core_files(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep only RNA counts, clinical bundles, and MAF files from manifest."""
    targets = manifest["filename"].str.contains(
        r"augmented_star_gene_counts\.tsv$|clinical\.|wxs\..*maf\.gz$|aliquot_ensemble_raw\.maf\.gz$",
        case=False,
        regex=True,
    )
    return manifest.loc[targets].copy()


def extract_clinical_tarball(clinical_tarball: str | Path, out_dir: str | Path) -> list[Path]:
    """Safely extract a clinical tarball and return extracted TSV paths."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _is_within_directory(base: Path, target: Path) -> bool:
        """Return True only when target resolves inside base directory."""
        try:
            target.resolve().relative_to(base.resolve())
            return True
        except ValueError:
            return False

    with tarfile.open(clinical_tarball, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            member_path = out_path / m.name
            if not _is_within_directory(out_path, member_path):
                raise ValueError(f"Unsafe path in tar archive: {m.name}")
        tf.extractall(path=out_path)

    return sorted(out_path.glob("*.tsv"))


def load_clinical_tables(clinical_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all TSV clinical tables from a directory into a dict."""
    tables: dict[str, pd.DataFrame] = {}
    for tsv in Path(clinical_dir).glob("*.tsv"):
        tables[tsv.stem] = pd.read_csv(tsv, sep="\t")
    return tables


def _to_case_barcode(x: str) -> str | None:
    """Extract canonical TCGA case barcode from a longer identifier string."""
    if not isinstance(x, str):
        return None
    match = CASE_BARCODE_RE.search(x)
    if not match:
        return None
    return match.group(0)


def _pick_first_existing(columns: list[str], preferred: list[str]) -> str | None:
    """Return the first preferred column name found in available columns."""
    for name in preferred:
        if name in columns:
            return name
    return None


def _pick_by_suffix(columns: list[str], suffixes: list[str]) -> str | None:
    """Find the first column that matches any provided suffix."""
    for sfx in suffixes:
        for c in columns:
            if c.endswith(sfx):
                return c
    return None


def _resolve_case_id_column(df: pd.DataFrame) -> str | None:
    """Resolve a case identifier column using exact names, then suffix matching."""
    cols = list(df.columns)
    direct = _pick_first_existing(
        cols,
        [
            "cases.submitter_id",
            "case_submitter_id",
            "submitter_id",
            "case_id",
            "cases.case_id",
        ],
    )
    if direct:
        return direct
    return _pick_by_suffix(
        cols,
        [
            ".submitter_id",
            ".case_id",
        ],
    )


def build_survival_frame(clinical_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build case-level OS/PFS frame from clinical and follow-up tables."""
    clinical = clinical_tables.get("clinical")
    if clinical is None:
        raise FileNotFoundError("clinical.tsv not found in extracted clinical tables")

    df = clinical.copy()
    id_col = _resolve_case_id_column(df)
    if id_col is None:
        raise ValueError(f"No case identifier column found in clinical.tsv. Columns: {list(df.columns)[:20]}...")
    df["case_id"] = df[id_col].map(_to_case_barcode)

    clinical_cols = list(df.columns)
    death_col = _pick_first_existing(clinical_cols, ["demographic.days_to_death", "days_to_death"])
    if death_col is None:
        death_col = _pick_by_suffix(clinical_cols, [".days_to_death"])
    if death_col is None:
        df["__days_to_death"] = pd.NA
        death_col = "__days_to_death"

    lfu_col = _pick_first_existing(
        clinical_cols,
        [
            "diagnoses.days_to_last_follow_up",
            "days_to_last_follow_up",
            "cases.days_to_lost_to_followup",
        ],
    )
    if lfu_col is None:
        lfu_col = _pick_by_suffix(
            clinical_cols,
            [
                ".days_to_last_follow_up",
                ".days_to_lost_to_followup",
            ],
        )
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
    fu_id_col = _resolve_case_id_column(fu)
    if fu_id_col is None:
        raise ValueError(f"No case identifier column found in follow_up.tsv. Columns: {list(fu.columns)[:20]}...")
    fu["case_id"] = fu[fu_id_col].map(_to_case_barcode)

    fu_cols = list(fu.columns)
    progress_time_col = _pick_first_existing(
        fu_cols,
        [
            "follow_ups.days_to_progression",
            "days_to_progression",
            "days_to_new_tumor_event_after_initial_treatment",
            "days_to_tumor_progression",
        ],
    )
    if progress_time_col is None:
        progress_time_col = _pick_by_suffix(
            fu_cols,
            [
                ".days_to_progression",
                ".days_to_new_tumor_event_after_initial_treatment",
                ".days_to_tumor_progression",
            ],
        )
    if progress_time_col is None:
        raise ValueError("No progression timing column found in follow_up.tsv")

    # Prefer follow-up table censoring if available, otherwise keep clinical-derived censoring.
    fu_lfu_col = _pick_first_existing(
        fu_cols,
        [
            "follow_ups.days_to_follow_up",
            "days_to_follow_up",
            "follow_ups.days_to_last_follow_up",
            "days_to_last_follow_up",
        ],
    )
    if fu_lfu_col is None:
        fu_lfu_col = _pick_by_suffix(fu_cols, [".days_to_follow_up", ".days_to_last_follow_up"])

    selected_cols = ["case_id", progress_time_col]
    if fu_lfu_col:
        selected_cols.append(fu_lfu_col)

    p = fu[selected_cols].copy()
    p["progress_days"] = pd.to_numeric(p[progress_time_col], errors="coerce")
    if fu_lfu_col:
        p["followup_days"] = pd.to_numeric(p[fu_lfu_col], errors="coerce")
    else:
        p["followup_days"] = pd.NA

    p = p.groupby("case_id", as_index=False).agg(
        progress_days=("progress_days", "min"),
        followup_days=("followup_days", "max"),
    )
    p["pfs_event"] = p["progress_days"].notna().astype(int)

    out = out.merge(p, on="case_id", how="left")
    out["pfs_event"] = out["pfs_event"].fillna(0).astype(int)
    out["pfs_days"] = out["progress_days"]
    censor_source = out["followup_days"].fillna(out["lfu_days"])
    out.loc[out["pfs_event"] == 0, "pfs_days"] = censor_source[out["pfs_event"] == 0]

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
    """Recursively list all compressed MAF files under a directory."""
    root = Path(search_dir)
    return sorted(root.rglob("*.maf.gz"))


def build_egfr_mutant_case_set(maf_files: list[Path]) -> set[str]:
    """Collect case IDs with EGFR mutations from provided MAF files."""
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
    """Check whether a TCGA sample barcode corresponds to primary tumor (01)."""
    # TCGA aliquot barcodes include sample type code in positions 14-15.
    # Example: TCGA-XX-YYYY-01A... where 01 = Primary Solid Tumor.
    if not isinstance(case_sample_barcode, str) or len(case_sample_barcode) < 15:
        return False
    return case_sample_barcode[13:15] == "01"
