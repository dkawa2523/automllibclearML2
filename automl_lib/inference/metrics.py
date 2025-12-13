"""Consistency scoring utilities for inference outputs."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def compute_consistency_scores(
    predictions_df: pd.DataFrame,
    models: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlation and agreement matrices plus mean scores."""

    model_cols = [m for m in models if m in predictions_df.columns]
    if len(model_cols) < 2:
        corr_matrix = pd.DataFrame()
        agreement_matrix = pd.DataFrame()
        mean_corr = pd.DataFrame({'model': model_cols, 'mean_correlation': [np.nan] * len(model_cols)})
        mean_agreement = pd.DataFrame({'model': model_cols, 'mean_agreement': [np.nan] * len(model_cols)})
        return corr_matrix, agreement_matrix, mean_corr, mean_agreement

    data = predictions_df[model_cols]
    corr_matrix = data.corr(method="pearson")

    global_min = np.nanmin(data.values)
    global_max = np.nanmax(data.values)
    denom = max(global_max - global_min, 1e-8)
    agreement_matrix = pd.DataFrame(index=model_cols, columns=model_cols, dtype=float)
    for i, m1 in enumerate(model_cols):
        for j, m2 in enumerate(model_cols):
            if i == j:
                agreement_matrix.loc[m1, m2] = 1.0
            else:
                pair = data[[m1, m2]].dropna()
                if pair.empty:
                    agreement_matrix.loc[m1, m2] = np.nan
                else:
                    diff = np.abs(pair[m1] - pair[m2]) / denom
                    agreement_matrix.loc[m1, m2] = 1.0 - float(diff.mean())

    mean_corr_vals = []
    mean_agree_vals = []
    for m in model_cols:
        corr_vals = [corr_matrix.loc[m, other] for other in model_cols if other != m]
        agree_vals = [agreement_matrix.loc[m, other] for other in model_cols if other != m]
        valid_corr = [val for val in corr_vals if pd.notna(val)]
        valid_agree = [val for val in agree_vals if pd.notna(val)]
        mean_corr_vals.append(float(np.mean(valid_corr)) if valid_corr else np.nan)
        mean_agree_vals.append(float(np.mean(valid_agree)) if valid_agree else np.nan)

    mean_corr = pd.DataFrame({'model': model_cols, 'mean_correlation': mean_corr_vals})
    mean_agreement = pd.DataFrame({'model': model_cols, 'mean_agreement': mean_agree_vals})
    return corr_matrix, agreement_matrix, mean_corr, mean_agreement


__all__ = ["compute_consistency_scores"]
