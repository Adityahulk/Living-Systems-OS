#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lsos_research.data_ingest import filter_manifest_for_core_files, read_gdc_manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare Phase 1 GDC manifest subsets")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="data/interim")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = read_gdc_manifest(args.manifest)
    core = filter_manifest_for_core_files(mdf)

    core.to_csv(out_dir / "gdc_manifest.core.tsv", sep="\t", index=False)

    rna = core[core["filename"].str.contains("augmented_star_gene_counts", case=False)]
    maf = core[core["filename"].str.contains("maf.gz", case=False)]
    clinical = core[core["filename"].str.contains("clinical", case=False)]

    rna.to_csv(out_dir / "gdc_manifest.rna_counts.tsv", sep="\t", index=False)
    maf.to_csv(out_dir / "gdc_manifest.maf.tsv", sep="\t", index=False)
    clinical.to_csv(out_dir / "gdc_manifest.clinical.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        {
            "subset": ["core", "rna_counts", "maf", "clinical"],
            "n_files": [len(core), len(rna), len(maf), len(clinical)],
        }
    )
    summary.to_csv(out_dir / "manifest_summary.tsv", sep="\t", index=False)

    print(summary)
    print("\nNext download commands (run manually if network access is available):")
    print(f"./gdc-client download -m {out_dir / 'gdc_manifest.rna_counts.tsv'} -d data/raw/gdc")
    print(f"./gdc-client download -m {out_dir / 'gdc_manifest.maf.tsv'} -d data/raw/gdc")


if __name__ == "__main__":
    main()
