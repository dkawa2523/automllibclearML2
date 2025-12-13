"""Plotting utilities for inference outputs."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt

from .model_utils import _running_best, _shorten_label, _shorten_labels


def plot_predictions(
    aggregated_df: pd.DataFrame,
    models: Iterable[str],
    output_dir: str,
    title: Optional[str] = None,
    per_model_results: Optional[Dict[str, pd.DataFrame]] = None,
    goal: Optional[str] = None,
) -> None:
    """Plot predictions across conditions for each model."""

    os.makedirs(output_dir, exist_ok=True)
    use_trial_axis = False
    if per_model_results:
        for df in per_model_results.values():
            if isinstance(df, pd.DataFrame) and "trial_index" in df.columns:
                use_trial_axis = True
                break
    plt.figure(figsize=(8, 5))
    used_labels: Set[str] = set()
    goal_normalized = (goal or "max").lower()
    if use_trial_axis:
        for model_label in models:
            df_model = per_model_results.get(model_label) if per_model_results else None
            if df_model is None or df_model.empty:
                continue
            df_ordered = df_model.sort_values("trial_index") if "trial_index" in df_model.columns else df_model
            x = df_ordered["trial_index"].to_numpy() if "trial_index" in df_ordered.columns else np.arange(len(df_ordered))
            if "prediction" in df_ordered.columns:
                y_series = pd.to_numeric(df_ordered["prediction"], errors="coerce")
            elif model_label in df_ordered.columns:
                y_series = pd.to_numeric(df_ordered[model_label], errors="coerce")
            else:
                continue
            y_values = y_series.to_numpy()
            running = _running_best(y_values, goal_normalized)
            valid = np.isfinite(running)
            if not valid.any():
                continue
            display_label = _shorten_label(model_label, used_labels)
            line, = plt.plot(x[valid], running[valid], linewidth=1.5, label=display_label)
            color = line.get_color()
            plt.scatter(x, y_values, s=12, color=color, alpha=0.5, edgecolors="none")
        plt.xlabel("Trial Index")
    else:
        if aggregated_df.empty:
            plt.close()
            return
        x = np.arange(len(aggregated_df))
        for model_label in models:
            if model_label not in aggregated_df.columns:
                continue
            y_series = pd.to_numeric(aggregated_df[model_label], errors="coerce")
            y_values = y_series.to_numpy()
            display_label = _shorten_label(model_label, used_labels)
            plt.plot(x, y_values, marker="o", markersize=3, linewidth=1.2, label=display_label)
        plt.xlabel("Condition Index")
    plt.ylabel("Predicted Target")
    plt.title(title or "Predictions across conditions")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    outpath = os.path.join(output_dir, "predictions_plot.png")
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_correlation_heatmap(
    predictions_df: pd.DataFrame,
    models: Iterable[str],
    output_dir: str,
    title: Optional[str] = None,
) -> None:
    """Plot a correlation matrix heatmap of model predictions."""

    if predictions_df.empty:
        return
    data = predictions_df[[m for m in models if m in predictions_df.columns]]
    if data.shape[1] < 2:
        return
    corr_matrix = data.corr(method="pearson")
    label_map = _shorten_labels(corr_matrix.columns)
    plt.figure(figsize=(6 + 0.4 * len(data.columns), 6))
    im = plt.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Pearson correlation")
    plt.xticks(
        range(len(corr_matrix.columns)),
        [label_map.get(col, col) for col in corr_matrix.columns],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(len(corr_matrix.index)),
        [label_map.get(idx, idx) for idx in corr_matrix.index],
    )
    plt.title(title or "Model Prediction Correlation")
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            plt.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if abs(val) > 0.5 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    outpath = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_agreement_heatmap(
    predictions_df: pd.DataFrame,
    models: Iterable[str],
    output_dir: str,
    title: Optional[str] = None,
) -> None:
    """Plot a heatmap showing agreement scores between models."""

    if predictions_df.empty:
        return
    model_cols = [m for m in models if m in predictions_df.columns]
    if len(model_cols) < 2:
        return
    data = predictions_df[model_cols]
    global_min = np.nanmin(data.values)
    global_max = np.nanmax(data.values)
    denom = max(global_max - global_min, 1e-8)
    agreement = pd.DataFrame(index=model_cols, columns=model_cols, dtype=float)
    for i, m1 in enumerate(model_cols):
        for j, m2 in enumerate(model_cols):
            if i == j:
                agreement.loc[m1, m2] = 1.0
            else:
                pair = data[[m1, m2]].dropna()
                if pair.empty:
                    agreement.loc[m1, m2] = np.nan
                else:
                    diff = np.abs(pair[m1] - pair[m2]) / denom
                    agreement.loc[m1, m2] = 1.0 - float(diff.mean())
    label_map = _shorten_labels(model_cols)
    plt.figure(figsize=(6 + 0.4 * len(model_cols), 6))
    im = plt.imshow(agreement.values.astype(float), cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Agreement score")
    plt.xticks(
        range(len(model_cols)),
        [label_map.get(col, col) for col in model_cols],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(len(model_cols)),
        [label_map.get(idx, idx) for idx in model_cols],
    )
    plt.title(title or "Model Prediction Agreement")
    for i in range(len(model_cols)):
        for j in range(len(model_cols)):
            val = agreement.iloc[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    outpath = os.path.join(output_dir, "agreement_heatmap.png")
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_consistency_bars(
    mean_scores: pd.DataFrame,
    score_col: str,
    output_dir: str,
    title: str,
    filename: str,
) -> None:
    """Plot a bar chart of mean consistency scores."""

    if mean_scores.empty:
        return
    os.makedirs(output_dir, exist_ok=True)
    df = mean_scores.sort_values(score_col, ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(df['model'], df[score_col], color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(score_col.replace('_', ' ').title())
    plt.title(title)
    plt.tight_layout()
    outpath = os.path.join(output_dir, filename)
    plt.savefig(outpath, dpi=300)
    plt.close()


__all__ = [
    "plot_predictions",
    "plot_correlation_heatmap",
    "plot_agreement_heatmap",
    "plot_consistency_bars",
]
