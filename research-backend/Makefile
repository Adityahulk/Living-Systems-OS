PY=python

.PHONY: prepare sample-sheet run report

prepare:
	$(PY) scripts/prepare_phase1.py --manifest gdc_manifest.2026-02-10.234928.txt --out-dir data/interim

sample-sheet:
	$(PY) scripts/build_sample_sheet.py --gdc-dir data/raw/gdc --out data/interim/rna_sample_sheet.tsv --mapped-out data/interim/rna_sample_sheet_mapped.tsv

run:
	$(PY) scripts/run_pipeline.py --config configs/default.yaml --root . --sample-sheet data/interim/rna_sample_sheet_mapped.tsv

report:
	$(PY) scripts/build_report.py
