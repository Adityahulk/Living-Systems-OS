#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown_table(df: pd.DataFrame) -> str:
    """Render DataFrame as a markdown table string."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def main() -> None:
    """Generate markdown summary of key metrics for paper/reporting."""
    root = Path(".")
    metrics = pd.read_csv(root / "results/metrics/fold_metrics.tsv", sep="\t")
    tests = Path("results/metrics/statistical_tests.json")

    summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            c_index_mean=("c_index", "mean"),
            c_index_std=("c_index", "std"),
        )
        .sort_values("rmse_mean")
    )

    out = Path("paper/results_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Results Summary\n\n")
        f.write("## Cross-Validation RMSE\n\n")
        f.write(to_markdown_table(summary))
        f.write("\n\n")
        f.write("## Statistical Tests\n\n")
        f.write(f"See `{tests}` for paired tests and bootstrap confidence intervals.\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
