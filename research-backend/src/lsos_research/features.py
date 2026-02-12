from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml


def load_pathway_gene_sets(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data: Mapping[str, list[str]] = yaml.safe_load(f)
    return {k: sorted(set(v)) for k, v in data.items()}


def compute_pathway_activity(expr_z: pd.DataFrame, pathway_sets: dict[str, list[str]]) -> pd.DataFrame:
    # expr_z shape: genes x patients
    out = {}
    gene_index_upper = {g.upper(): g for g in expr_z.index}

    for pathway, genes in pathway_sets.items():
        present = [gene_index_upper[g.upper()] for g in genes if g.upper() in gene_index_upper]
        if not present:
            continue
        out[pathway] = expr_z.loc[present].mean(axis=0)

    if not out:
        raise ValueError("No pathway genes overlapped expression matrix")

    feat = pd.DataFrame(out)
    feat.index.name = "case_id"
    return feat
