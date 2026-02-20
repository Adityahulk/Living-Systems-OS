#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_BASE = "https://api.gdc.cancer.gov/data"


def build_session(total_retries: int, backoff_factor: float) -> requests.Session:
    """Create retry-enabled HTTP session for resilient GDC downloads."""
    s = requests.Session()
    retry = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        allowed_methods=["GET"],
        status_forcelist=[408, 429, 500, 502, 503, 504],
        backoff_factor=backoff_factor,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def md5_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute file MD5 checksum in streaming chunks."""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_complete(path: Path, expected_size: int | None, expected_md5: str | None) -> bool:
    """Check whether a local file matches expected size/hash metadata."""
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    if expected_md5:
        return md5_file(path) == expected_md5.lower()
    return True


def download_one(
    session: requests.Session,
    file_id: str,
    out_file: Path,
    expected_size: int | None,
    timeout: int,
) -> None:
    """Download one GDC file, resuming partial downloads when possible."""
    out_file.parent.mkdir(parents=True, exist_ok=True)

    existing = out_file.stat().st_size if out_file.exists() else 0
    headers = {}
    mode = "wb"
    if existing > 0 and expected_size and existing < expected_size:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    url = f"{API_BASE}/{file_id}"
    with session.get(url, stream=True, headers=headers, timeout=timeout) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {file_id}")
        with out_file.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def main() -> None:
    """Run robust manifest-driven downloader with integrity checks and retries."""
    ap = argparse.ArgumentParser(description="Robust downloader for GDC manifest files")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--backoff", type=float, default=0.8)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--max-files", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    required_cols = {"id", "filename", "size", "md5"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.max_files > 0:
        manifest = manifest.head(args.max_files).copy()

    session = build_session(total_retries=args.retries, backoff_factor=args.backoff)

    n_total = len(manifest)
    n_done = 0
    n_skipped = 0
    n_failed = 0
    failed_ids: list[str] = []

    start = time.time()
    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        file_id = str(row.id)
        filename = str(row.filename)
        expected_size = int(row.size) if not pd.isna(row.size) else None
        expected_md5 = str(row.md5) if not pd.isna(row.md5) else None

        out_dir = out_root / file_id
        out_file = out_dir / filename

        try:
            if is_complete(out_file, expected_size, expected_md5):
                n_skipped += 1
                print(f"[{i}/{n_total}] skip {file_id}")
                continue

            download_one(
                session=session,
                file_id=file_id,
                out_file=out_file,
                expected_size=expected_size,
                timeout=args.timeout,
            )

            if not is_complete(out_file, expected_size, expected_md5):
                raise RuntimeError("integrity check failed")

            n_done += 1
            print(f"[{i}/{n_total}] ok   {file_id}")
        except Exception as e:
            n_failed += 1
            failed_ids.append(file_id)
            print(f"[{i}/{n_total}] fail {file_id} :: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    elapsed = time.time() - start
    print("\nDownload summary")
    print(f"total={n_total} downloaded={n_done} skipped={n_skipped} failed={n_failed} elapsed_sec={elapsed:.1f}")

    if failed_ids:
        failed_path = out_root / "failed_ids.txt"
        failed_path.write_text("\n".join(failed_ids) + "\n", encoding="utf-8")
        print(f"failed IDs written to {failed_path}")


if __name__ == "__main__":
    main()
