from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def save_metric_boxplot(metric_df: pd.DataFrame, out_path: str | Path, metric: str) -> None:
    """Save model-comparison boxplot for a chosen metric."""
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=metric_df, x="model", y=metric)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_calibration_plot(calib_df: pd.DataFrame, out_path: str | Path) -> None:
    """Save calibration curve (predicted vs observed), with empty-data fallback."""
    if calib_df.empty:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "Calibration unavailable (insufficient prediction variance)", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        return

    plt.figure(figsize=(6, 6))
    plt.plot(calib_df["pred_mean"], calib_df["true_mean"], marker="o", label="Model")
    lims = [
        min(calib_df["pred_mean"].min(), calib_df["true_mean"].min()),
        max(calib_df["pred_mean"].max(), calib_df["true_mean"].max()),
    ]
    plt.plot(lims, lims, linestyle="--", color="black", label="Ideal")
    plt.xlabel("Predicted mean resistance time (days)")
    plt.ylabel("Observed mean resistance proxy (days)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_ablation_barplot(metric_df: pd.DataFrame, out_path: str | Path) -> None:
    """Save mean+SD RMSE barplot for ablation models."""
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metric_df, x="model", y="rmse", estimator="mean", errorbar="sd")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("RMSE (days)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_perturbation_plot(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save scenario-wise resistance-time distribution plot."""
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="scenario", y="pred_resistance_days")
    plt.ylabel("Predicted resistance time (days)")
    plt.xlabel("Treatment perturbation scenario")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
