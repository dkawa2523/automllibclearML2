"""Interpretability utilities for the AutoML library.

This module provides functions to extract and plot feature importances from
models that expose ``coef_`` or ``feature_importances_`` attributes, and to
compute SHAP values for supported models. SHAP values quantify the
contribution of each feature to individual predictions, offering
model‑agnostic explanations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from sklearn.compose import TransformedTargetRegressor

import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore


def extract_feature_importance(model: object, feature_names: List[str]) -> Optional[pd.DataFrame]:
    """Extract feature importance or coefficient magnitudes from a trained model.

    Parameters
    ----------
    model : estimator
        Fitted estimator (e.g., RandomForestRegressor, LinearRegression).
    feature_names : list of str
        Names of the input features after preprocessing (must match the model's
        input order).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with columns ['feature', 'importance'] sorted descending. If
        the model does not expose importance attributes, returns None.
    """
    if isinstance(model, TransformedTargetRegressor):
        model = getattr(model, "regressor_", getattr(model, "regressor", model))

    importance: Optional[np.ndarray] = None
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        # Flatten in case of multi‑output
        importance = np.abs(coef).ravel()
    if importance is None:
        return None
    # Ensure lengths match feature names
    if len(importance) != len(feature_names):
        # Cannot compute meaningful importances
        return None
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 20,
    title: Optional[str] = None,
) -> None:
    """Plot bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance'].
    output_path : Path
        File path to save the plot.
    top_n : int, optional
        Number of top features to display.
    title : str, optional
        Plot title.
    """
    df = importance_df.head(top_n)
    plt.figure(figsize=(8, max(3, len(df) * 0.4)))
    plt.barh(df["feature"], df["importance"], color="tab:green")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title or "Feature Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_shap_summary(
    model_pipeline: object,
    X: np.ndarray | pd.DataFrame,
    output_path: Path,
    max_display: int = 20,
) -> None:
    """Compute SHAP values and plot a summary bar chart.

    This function requires the 'shap' library. The pipeline must be a
    scikit‑learn Pipeline containing a preprocessor and a fitted model. For
    tree‑based models, TreeExplainer is used; otherwise KernelExplainer.

    Parameters
    ----------
    model_pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline.
    X : array‑like
        Original training data (before preprocessing). SHAP uses the
        preprocessor inside the pipeline automatically.
    output_path : Path
        Path to save the summary plot.
    max_display : int
        Maximum number of features to display in the summary plot.
    """
    if shap is None:
        raise RuntimeError("SHAP library is not installed")
    # Decompose pipeline
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]
    if isinstance(model, TransformedTargetRegressor):
        model = getattr(model, "regressor_", getattr(model, "regressor", model))
    # Transform data for SHAP explainer; use dense array
    X_trans = preprocessor.transform(X)
    # Use appropriate explainer
    if hasattr(model, "predict_proba"):
        # Classification; treat as tree if possible
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(model.predict_proba, X_trans)
        shap_values = explainer.shap_values(X_trans)
        # For multi‑class, sum absolute values across classes
        if isinstance(shap_values, list):
            shap_values = np.sum([np.abs(sv) for sv in shap_values], axis=0)
    else:
        # Regression
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.KernelExplainer(model.predict, X_trans)
        shap_values = explainer.shap_values(X_trans)
    # Build feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback: generic indices
        feature_names = [f"f{i}" for i in range(X_trans.shape[1])]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The NumPy global RNG was seeded by calling.*",
            category=FutureWarning,
        )
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False, max_display=max_display)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt  # Re‑import for saving
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
