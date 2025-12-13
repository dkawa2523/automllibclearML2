"""File output helpers for inference results."""

from __future__ import annotations

import os
from typing import Dict

import pandas as pd


def save_results(
    aggregated_df: pd.DataFrame,
    per_model: Dict[str, pd.DataFrame],
    output_dir: str,
    prefix: str,
) -> None:
    """Save aggregated and per-model predictions to CSV files."""

    os.makedirs(output_dir, exist_ok=True)
    agg_path = os.path.join(output_dir, f"{prefix}_aggregated.csv")
    aggregated_df.to_csv(agg_path, index=False)
    for model_label, df in per_model.items():
        model_safe = model_label.replace(" ", "_")
        path = os.path.join(output_dir, f"{prefix}_{model_safe}.csv")
        df.to_csv(path, index=False)


def save_matrices_and_scores(
    corr_matrix: pd.DataFrame,
    agreement_matrix: pd.DataFrame,
    mean_corr: pd.DataFrame,
    mean_agreement: pd.DataFrame,
    output_dir: str,
    prefix: str,
) -> None:
    """Persist correlation/agreement matrices plus mean scores to CSV files."""

    os.makedirs(output_dir, exist_ok=True)
    if not corr_matrix.empty:
        corr_matrix.to_csv(os.path.join(output_dir, f"{prefix}_correlation_matrix.csv"))
    if not agreement_matrix.empty:
        agreement_matrix.to_csv(os.path.join(output_dir, f"{prefix}_agreement_matrix.csv"))
    if not mean_corr.empty:
        mean_corr.to_csv(os.path.join(output_dir, f"{prefix}_mean_correlation.csv"), index=False)
    if not mean_agreement.empty:
        mean_agreement.to_csv(os.path.join(output_dir, f"{prefix}_mean_agreement.csv"), index=False)


__all__ = [
    "save_results",
    "save_matrices_and_scores",
]
