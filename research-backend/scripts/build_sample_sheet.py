#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lsos_research.gdc import build_rna_sample_sheet, query_file_metadata, remap_sample_sheet_with_metadata


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build and map RNA sample sheet from GDC downloads")
    ap.add_argument("--gdc-dir", default="data/raw/gdc")
    ap.add_argument("--out", default="data/interim/rna_sample_sheet.tsv")
    ap.add_argument("--mapped-out", default="data/interim/rna_sample_sheet_mapped.tsv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    sheet = build_rna_sample_sheet(args.gdc_dir, out)
    print(f"Created sample sheet with {len(sheet)} rows: {out}")

    file_ids = []
    for p in sheet["file_path"]:
        # The parent directory name in gdc-client download is typically file UUID.
        file_ids.append(Path(p).parent.name)

    fields = ["file_name", "cases.submitter_id", "cases.samples.submitter_id"]
    md = query_file_metadata(file_ids=file_ids, fields=fields)
    md_path = out.parent / "gdc_file_metadata.tsv"
    md.to_csv(md_path, sep="\t", index=False)

    mapped = remap_sample_sheet_with_metadata(sheet, md, args.mapped_out)
    print(f"Mapped sample sheet with case/sample IDs: {args.mapped_out} ({len(mapped)} rows)")


if __name__ == "__main__":
    main()
