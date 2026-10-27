from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, METRICS_DIR


def plot_class_distribution(df: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = df["Pass_Fail"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index.to_numpy(), y=counts.to_numpy(), hue=counts.index.to_numpy(), dodge=False, palette="deep", ax=ax, legend=False)
    ax.set_title("Pass/Fail Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_missingness(df: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = df.isna().mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    missing.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top Missingness by Sensor")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing proportion")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_correlation(df: pd.DataFrame, output_path: str | Path, top_n: int = 30) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric = df.select_dtypes(include=[np.number]).copy()
    selected = numeric.iloc[:, :top_n]
    corr = selected.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar=True, ax=ax)
    ax.set_title("Correlation Heatmap for Selected Sensors")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_data_quality_report(summary: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return path


def create_eda_outputs(df: pd.DataFrame, output_dir: str | Path = FIGURES_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pass_fail_distribution": plot_class_distribution(df, output_dir / "pass_fail_distribution.png"),
        "missingness": plot_missingness(df, output_dir / "missingness.png"),
        "correlation": plot_correlation(df, output_dir / "correlation_heatmap.png"),
    }
    return {"figures": {k: str(v) for k, v in paths.items()}}
