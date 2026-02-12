from __future__ import annotations

import json
import ast
from pathlib import Path

import pandas as pd
import requests

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"


def query_file_metadata(file_ids: list[str], fields: list[str], batch_size: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(0, len(file_ids), batch_size):
        chunk = file_ids[i : i + batch_size]
        payload = {
            "filters": {
                "op": "in",
                "content": {"field": "files.file_id", "value": chunk},
            },
            "fields": ",".join(fields),
            "format": "JSON",
            "size": str(len(chunk)),
        }
        resp = requests.post(
            GDC_FILES_ENDPOINT,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        rows.extend(data.get("data", {}).get("hits", []))

    return pd.json_normalize(rows)


def write_gdc_filtered_manifest(manifest_df: pd.DataFrame, out_path: str | Path) -> None:
    manifest_df.to_csv(out_path, sep="\t", index=False)


def build_rna_sample_sheet(gdc_download_dir: str | Path, out_path: str | Path) -> pd.DataFrame:
    rows = []
    root = Path(gdc_download_dir)

    for f in root.rglob("*.rna_seq.augmented_star_gene_counts.tsv"):
        fname = f.name
        # GDC naming commonly starts with aliquot UUID then suffix.
        submitter_stub = fname.split(".")[0]
        rows.append({
            "file_path": str(f.resolve()),
            "case_id": submitter_stub,  # placeholder until metadata join is provided
            "sample_barcode": submitter_stub,
            "filename": fname,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError("No augmented STAR gene count files found in download directory")

    df.to_csv(out_path, sep="\t", index=False)
    return df


def remap_sample_sheet_with_metadata(
    sample_sheet: pd.DataFrame,
    metadata_df: pd.DataFrame,
    out_path: str | Path,
) -> pd.DataFrame:
    md = metadata_df.copy()
    if "file_name" not in md.columns:
        raise ValueError("Metadata missing required field: file_name")

    if "cases.0.submitter_id" in md.columns and "cases.0.samples.0.submitter_id" in md.columns:
        md = md.rename(
            columns={
                "file_name": "filename",
                "cases.0.submitter_id": "case_id",
                "cases.0.samples.0.submitter_id": "sample_barcode",
            }
        )
    elif "cases" in md.columns:
        def _parse_cases(v):
            if isinstance(v, str):
                try:
                    v = ast.literal_eval(v)
                except Exception:
                    return None, None
            if not isinstance(v, list) or not v:
                return None, None
            case = v[0].get("submitter_id")
            sample = None
            samples = v[0].get("samples")
            if isinstance(samples, list) and samples:
                sample = samples[0].get("submitter_id")
            return case, sample

        parsed = md["cases"].apply(_parse_cases)
        md["case_id"] = parsed.apply(lambda x: x[0])
        md["sample_barcode"] = parsed.apply(lambda x: x[1])
        md = md.rename(columns={"file_name": "filename"})
    else:
        raise ValueError(
            "Metadata missing case/sample mapping fields; expected flattened cases.* or cases object"
        )

    merged = sample_sheet.drop(columns=["case_id", "sample_barcode"], errors="ignore").merge(
        md[["filename", "case_id", "sample_barcode"]], on="filename", how="left"
    )

    merged = merged.dropna(subset=["case_id", "sample_barcode"])
    merged.to_csv(out_path, sep="\t", index=False)
    return merged
