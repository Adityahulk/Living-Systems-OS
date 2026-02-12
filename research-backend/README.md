# Research Backend: EGFR-Mutant NSCLC Hybrid Dynamics

This module contains the research pipeline for LS-OS paper-track development.

## Scope

Implements the end-to-end workflow for:
- TCGA-LUAD ingestion and cohort construction
- RNA-seq normalization and pathway activity features
- Hybrid mechanistic-learning model (NN -> ODE -> resistance time)
- Baselines (Cox and MLP)
- Perturbation robustness experiments
- Ablation suite
- Publication-grade metric/figure artifacts

## Directory Layout

```text
research-backend/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── pathways/
├── results/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   ├── models/
│   └── tables/
├── scripts/
├── src/lsos_research/
├── paper/
├── pyproject.toml
└── Dockerfile
```

## Setup

```bash
cd research-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Phase 1: Manifest Preparation

```bash
python scripts/prepare_phase1.py \
  --manifest gdc_manifest.2026-02-10.234928.txt \
  --out-dir data/interim
```

This creates filtered manifests for RNA counts, MAF files, and clinical files.

## Data Download (if network-enabled)

```bash
./gdc-client download -m data/interim/gdc_manifest.rna_counts.tsv -d data/raw/gdc
./gdc-client download -m data/interim/gdc_manifest.maf.tsv -d data/raw/gdc
```

## Build RNA Sample Sheet

```bash
python scripts/build_sample_sheet.py \
  --gdc-dir data/raw/gdc \
  --out data/interim/rna_sample_sheet.tsv \
  --mapped-out data/interim/rna_sample_sheet_mapped.tsv
```

## Run Full Pipeline

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --root . \
  --sample-sheet data/interim/rna_sample_sheet_mapped.tsv
```

## Outputs

Primary outputs are written to `results/`:
- `results/tables/predictions_all.tsv`
- `results/metrics/fold_metrics.tsv`
- `results/metrics/statistical_tests.json`
- `results/metrics/perturbation_results.tsv`
- `results/tables/hybrid_parameters.tsv`
- `results/models/hybrid_final.pt`
- `results/figures/*.png`

## Paper Artifacts

Generate a basic results markdown summary:

```bash
python scripts/build_report.py
```

Then continue writing from:
- `paper/outline.md`
- `paper/results_summary.md`

## Reproducibility Notes

- Global seeds are fixed via config.
- All configs are versioned under `configs/`.
- Outputs are deterministic given identical environment and data.
- For peer review portability, use the provided Dockerfile.
