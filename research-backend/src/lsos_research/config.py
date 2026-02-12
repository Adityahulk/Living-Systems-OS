from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class ProjectPaths:
    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    results_dir: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict[str, Any], root: str | Path) -> ProjectPaths:
    root_path = Path(root).resolve()
    p = config["paths"]
    return ProjectPaths(
        root=root_path,
        raw_dir=root_path / p["raw_dir"],
        interim_dir=root_path / p["interim_dir"],
        processed_dir=root_path / p["processed_dir"],
        results_dir=root_path / p["results_dir"],
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    for d in [paths.raw_dir, paths.interim_dir, paths.processed_dir, paths.results_dir]:
        d.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
