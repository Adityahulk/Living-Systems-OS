#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Validate that downloaded files exist and match manifest-declared sizes."""
    ap = argparse.ArgumentParser(description="Check download completeness for a GDC manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--download-dir", required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    dl_root = Path(args.download_dir)

    missing = []
    bad_size = []

    for row in manifest.itertuples(index=False):
        file_id = str(row.id)
        filename = str(row.filename)
        exp_size = int(row.size)

        path = dl_root / file_id / filename
        if not path.exists():
            missing.append(file_id)
            continue

        if path.stat().st_size != exp_size:
            bad_size.append(file_id)

    print(f"manifest files: {len(manifest)}")
    print(f"missing: {len(missing)}")
    print(f"size_mismatch: {len(bad_size)}")

    if missing:
        out = dl_root / "missing_ids.txt"
        out.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"missing ids -> {out}")

    if bad_size:
        out = dl_root / "size_mismatch_ids.txt"
        out.write_text("\n".join(bad_size) + "\n", encoding="utf-8")
        print(f"size mismatch ids -> {out}")


if __name__ == "__main__":
    main()
