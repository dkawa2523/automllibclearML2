"""Visualization utilities for the AutoML library.

This module collects functions to produce plots that compare model
performances and illustrate the quality of predictions. The charts are
intended to help non‑experts understand which models perform best and
how well they fit the data. All functions accept file paths for saving
figures to disk.

The plotting routines use Matplotlib exclusively to avoid adding heavy
dependencies. Colors and styles are chosen to be distinct and readable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Use non‑interactive backend for file output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


_PREPROCESSOR_ABBREVIATIONS = {
    "impute_mean_scale_standard": "ImpMean+StdScale",
    "impute_mean": "ImpMean",
    "impute_median": "ImpMedian",
    "impute_most_frequent": "ImpFreq",
    "impute_constant": "ImpConst",
    "impute_knn": "ImpKNN",
    "scale_standard": "StdScale",
    "scale_minmax": "MinMax",
    "scale_robust": "RobustScale",
    "scale_power": "PowerScale",
    "scale_maxabs": "MaxAbsScale",
    "normalize": "Normalize",
    "encode_onehot": "OneHot",
    "encode_target": "TargetEnc",
    "encode_ordinal": "OrdinalEnc",
    "encode_label": "LabelEnc",
    "encode_binary": "BinaryEnc",
    "encode_hashing": "HashEnc",
    "encode_catboost": "CatBoostEnc",
    "encode_frequency": "FreqEnc",
    "encode_leave_one_out": "LOOEnc",
    "feature_select_variance_threshold": "VarThresh",
    "feature_select_kbest": "KBest",
    "feature_select_mutual_info": "MutInfo",
    "feature_select_rfe": "RFE",
    "polynomial_features": "PolyFeat",
    "engineer_datetime": "DateFeat",
    "impute_mean_scale_standard|impute_most_frequent_encode_onehot": "ImpMean+StdScale | ImpFreq+OneHot",
    "impute_most_frequent_encode_onehot": "ImpFreq+OneHot",
}

_PARAM_ABBREVIATIONS = {
    "n_estimators": "n_est",
    "learning_rate": "lr",
    "max_depth": "max_depth",
    "num_leaves": "num_leaves",
    "iterations": "iters",
    "depth": "depth",
    "epsilon": "eps",
    "hidden_layer_sizes": "layers",
    "activation": "act",
    "learning_rate_init": "lr_init",
    "n_iter_no_change": "no_change",
    "validation_fraction": "val_frac",
    "random_state": "rng",
    "min_child_samples": "min_child",
    "force_row_wise": "row_wise",
}


def _shorten_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def _abbreviate_preprocessor_component(component: str) -> str:
    component = component.strip()
    if not component:
        return ""
    if component in _PREPROCESSOR_ABBREVIATIONS:
        return _PREPROCESSOR_ABBREVIATIONS[component]
    tokens = [token for token in component.split("_") if token]
    pieces: List[str] = []
    idx = 0
    while idx < len(tokens):
        matched = False
        for span in range(min(3, len(tokens) - idx), 0, -1):
            key = "_".join(tokens[idx : idx + span])
            if key in _PREPROCESSOR_ABBREVIATIONS:
                pieces.append(_PREPROCESSOR_ABBREVIATIONS[key])
                idx += span
                matched = True
                break
        if matched:
            continue
        token = tokens[idx]
        pieces.append(token[:3].capitalize())
        idx += 1
    return "+".join(pieces)


def _summarize_preprocessor(preprocessor: Any, max_length: int = 48) -> str:
    if not isinstance(preprocessor, str):
        return ""
    components = [comp for comp in preprocessor.split("|") if comp]
    shortened = [_abbreviate_preprocessor_component(component) for component in components]
    text = " | ".join(part for part in shortened if part)
    return _shorten_text(text, max_length)


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        formatted = "/".join(_format_param_value(val) for val in value[:2])
        if len(value) > 2:
            formatted += "/…"
        return f"[{formatted}]"
    value_str = str(value)
    if len(value_str) > 15:
        return value_str[:12] + "…"
    return value_str


def _summarize_params(params: Mapping[str, Any], max_items: int = 3, max_length: int = 56) -> str:
    if not params:
        return ""
    items: List[str] = []
    for idx, (key, value) in enumerate(params.items()):
        if idx >= max_items:
            items.append("…")
            break
        key_short = _PARAM_ABBREVIATIONS.get(key, key)
        items.append(f"{key_short}={_format_param_value(value)}")
    text = ", ".join(items)
    return _shorten_text(text, max_length)


def _build_model_label(row: pd.Series) -> str:
    model_name = str(row.get("model", "")) or "Model"
    preprocessor_text = _summarize_preprocessor(row.get("preprocessor", ""))
    params_obj = row.get("params")
    params_text = _summarize_params(params_obj) if isinstance(params_obj, Mapping) else ""
    details = [part for part in [preprocessor_text, params_text] if part]
    if details:
        detail_line = " | ".join(details)
        detail_line = _shorten_text(detail_line, 64)
        return f"{model_name}\n{detail_line}"
    return model_name


def plot_bar_comparison(
    results_df: pd.DataFrame,
    metric: str,
    top_n: int,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Create a horizontal bar chart comparing models on a single metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame produced by ``evaluate_model_combinations``.
    metric : str
        Name of the metric column to plot (e.g. 'r2', 'accuracy').
    top_n : int
        Number of top performers to include. If fewer rows exist, all are
        plotted.
    output_path : Path
        Location where the PNG file will be saved.
    title : str, optional
        Plot title. If omitted, a default is constructed.
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric '{metric}' is not present in the results")
    # Drop rows with errors or missing metric
    df = results_df.dropna(subset=[metric])
    if df.empty:
        return
    df_sorted = df.sort_values(by=metric, ascending=False).head(top_n)
    labels = [_build_model_label(row) for _, row in df_sorted.iterrows()]
    values = df_sorted[metric].values
    plt.figure(figsize=(10, max(2, len(df_sorted) * 0.6)))
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, values, color="tab:blue")
    plt.yticks(y_pos, labels, fontsize=8)
    plt.xlabel(metric)
    if title:
        plt.title(title)
    else:
        plt.title(f"Top {len(df_sorted)} models by {metric}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predicted_vs_actual(
    pipeline: Pipeline,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    output_path: Path,
    title: Optional[str] = None,
    predictions: Optional[np.ndarray] = None,
    r2_override: Optional[float] = None,
    add_regression_line: bool = True,
) -> None:
    """Plot predicted vs. actual values for regression models.

    The pipeline should already be fitted. If it is not, the caller must
    fit it on training data beforehand. This function will produce a
    scatter plot of predictions versus targets and a diagonal line
    indicating perfect predictions.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline containing a preprocessor and a regressor.
    X : array‑like
        Test features.
    y : array‑like
        True target values.
    output_path : Path
        Path where the plot PNG will be saved.
    title : str, optional
        Custom title for the plot.
    """
    y_true = np.asarray(y)
    if predictions is None:
        if pipeline is None:
            raise ValueError("Either a fitted pipeline or predictions must be provided")
        y_pred = np.asarray(pipeline.predict(X))
    else:
        y_pred = np.asarray(predictions)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("Predictions and targets must have the same length for plotting")
    # Compute R^2 score
    if r2_override is not None:
        r2 = r2_override
    else:
        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = pipeline.score(X, y) if pipeline is not None else float("nan")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor="k", facecolor="tab:orange", label="samples")
    # Identity line
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Ideal")
    if add_regression_line:
        try:
            coef = np.polyfit(y_true, y_pred, deg=1)
            x_line = np.linspace(min_val, max_val, 100)
            y_line = coef[0] * x_line + coef[1]
            plt.plot(x_line, y_line, color="blue", linestyle="-", linewidth=1, label="Fit")
        except Exception:
            pass
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title or "Predicted vs. Actual")
    # Annotate R^2
    plt.text(
        0.05,
        0.95,
        f"$R^2$ = {r2:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def _maybe_import_plotly():
    try:  # pragma: no cover - optional
        import plotly.graph_objects as go  # type: ignore
        import plotly.express as px  # type: ignore

        return go, px
    except Exception:
        return None, None


def build_plotly_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    add_regression_line: bool = True,
) -> Optional["go.Figure"]:
    go, _ = _maybe_import_plotly()
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="samples"))
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Ideal", line=dict(dash="dash", color="red")))
    if add_regression_line:
        try:
            coef = np.polyfit(y_true, y_pred, deg=1)
            x_line = np.linspace(min_val, max_val, 100)
            y_line = coef[0] * x_line + coef[1]
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Fit", line=dict(color="blue")))
        except Exception:
            pass
    try:
        r2 = r2_score(y_true, y_pred)
        fig.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper", text=f"R²={r2:.3f}", showarrow=False, bgcolor="white")
    except Exception:
        pass
    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted")
    return fig


def build_plotly_residual_scatter(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    title: str,
) -> Optional["go.Figure"]:
    go, _ = _maybe_import_plotly()
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", name="residuals"))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Residual (Actual − Predicted)")
    return fig


def build_plotly_histogram(values: List[float], metric_name: str, title: str) -> Optional["go.Figure"]:
    go, _ = _maybe_import_plotly()
    if go is None:
        return None
    fig = go.Figure(data=[go.Histogram(x=values, nbinsx=min(30, max(5, len(values))))])
    fig.update_layout(title=title, xaxis_title=metric_name, yaxis_title="Frequency")
    return fig


def build_plotly_interpolation_space(
    features: np.ndarray,
    targets: np.ndarray,
    title: str,
) -> Optional["go.Figure"]:
    go, px = _maybe_import_plotly()
    if px is None:
        return None
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return None
    X = np.asarray(features)
    y = np.asarray(targets)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    n_components = 2 if X.shape[1] >= 2 else 1
    comps = PCA(n_components=n_components).fit_transform(X)
    if n_components == 1:
        fig = px.scatter(x=comps[:, 0], y=[0] * len(comps), color=y, labels={"x": "PC1"}, title=title, color_continuous_scale="Viridis")
    else:
        fig = px.scatter(
            x=comps[:, 0],
            y=comps[:, 1],
            color=y,
            labels={"x": "PC1", "y": "PC2", "color": "target"},
            title=title,
            color_continuous_scale="Viridis",
        )
    return fig


def plot_residual_scatter(
    pipeline: Pipeline,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    output_path: Path,
    title: Optional[str] = None,
    predictions: Optional[np.ndarray] = None,
    residuals: Optional[np.ndarray] = None,
    ) -> None:
    """Plot residuals versus predicted values for regression models.

    This plot helps identify patterns in the residuals (errors) such as
    non‑linearity or heteroscedasticity. A horizontal line at zero error
    indicates perfect predictions.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline containing a preprocessor and a regressor.
    X : array‑like
        Features used for prediction.
    y : array‑like
        True target values.
    output_path : Path
        File path to save the plot.
    title : str, optional
        Title for the plot.
    """
    y_true = np.asarray(y)
    if residuals is not None:
        resid = np.asarray(residuals)
        if resid.shape[0] != y_true.shape[0]:
            raise ValueError("Residuals and targets must have the same length")
        if predictions is None:
            y_pred = y_true - resid
        else:
            y_pred = np.asarray(predictions)
    else:
        if predictions is None:
            if pipeline is None:
                raise ValueError("Either a fitted pipeline or predictions must be provided")
            y_pred = np.asarray(pipeline.predict(X))
        else:
            y_pred = np.asarray(predictions)
        resid = y_true - y_pred
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("Predictions and targets must have the same length")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_pred, resid, alpha=0.6, edgecolor="k", facecolor="tab:purple")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual − Predicted)")
    plt.title(title or "Residuals vs. Predicted")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_hist(
    pipeline: Pipeline,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    output_path: Path,
    title: Optional[str] = None,
    bins: int = 30,
    predictions: Optional[np.ndarray] = None,
    residuals: Optional[np.ndarray] = None,
    ) -> None:
    """Plot histogram of residuals (errors) for a regression model.

    The histogram shows the distribution of prediction errors. A symmetric
    distribution centered around zero indicates unbiased predictions.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline.
    X : array‑like
        Features used for prediction.
    y : array‑like
        True target values.
    output_path : Path
        File path to save the plot.
    title : str, optional
        Title for the plot.
    bins : int
        Number of histogram bins.
    """
    y_true = np.asarray(y)
    if residuals is not None:
        resid = np.asarray(residuals)
        if resid.shape[0] != y_true.shape[0]:
            raise ValueError("Residuals and targets must have the same length")
    else:
        if predictions is None:
            if pipeline is None:
                raise ValueError("Either a fitted pipeline or predictions must be provided")
            y_pred = np.asarray(pipeline.predict(X))
        else:
            y_pred = np.asarray(predictions)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("Predictions and targets must have the same length")
        resid = y_true - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=bins, color="tab:green", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Residual (Actual − Predicted)")
    plt.ylabel("Frequency")
    plt.title(title or "Residual Distribution")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metric_histogram(
    values: List[float],
    output_path: Path,
    metric_name: str,
    title: Optional[str] = None,
    bins: int = 20,
) -> None:
    """Plot histogram for a list of metric values."""

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=bins, color="tab:blue", edgecolor="black", alpha=0.7)
    plt.xlabel(metric_name)
    plt.ylabel("Frequency")
    plt.title(title or f"{metric_name} distribution")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_interpolation_space(
    features: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Project features to 2D (PCA) to visualise interpolation space."""

    from sklearn.decomposition import PCA

    X = np.asarray(features)
    y = np.asarray(targets)
    if X.ndim != 2 or X.shape[0] < 2:
        return
    n_components = 2 if X.shape[1] >= 2 else 1
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    if n_components == 1:
        plt.scatter(comps[:, 0], [0] * len(comps), c=y, cmap="viridis", edgecolor="k", alpha=0.7)
        plt.xlabel("PC1")
        plt.yticks([])
    else:
        plt.scatter(comps[:, 0], comps[:, 1], c=y, cmap="viridis", edgecolor="k", alpha=0.7)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
    plt.colorbar(label="target")
    plt.title(title or "Interpolation space (PCA)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

# -------------------------------------------------------------------------
# New function: heatmap for model/metric comparison
# -------------------------------------------------------------------------
def plot_metric_heatmap(
    results_df: pd.DataFrame,
    metrics: List[str],
    output_path: Path,
    primary_metric: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """Plot a heatmap comparing models across multiple metrics.

    When many models are evaluated, a bar chart for each metric can become
    unwieldy. A heatmap provides a compact overview by placing models on
    the y-axis and metrics on the x-axis with colors indicating relative
    performance.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame where each row corresponds to a model (best configuration).
        Must include a 'model' column and metric columns.
    metrics : list of str
        Names of metric columns to include in the heatmap. Columns not present
        in the DataFrame are ignored.
    output_path : Path
        File path where the heatmap PNG will be saved.
    primary_metric : str, optional
        Metric used to sort models top-to-bottom. If not provided or not in
        ``metrics``, models are sorted alphabetically.
    title : str, optional
        Title for the heatmap.
    """
    if results_df.empty:
        return
    # Filter metrics that actually exist in the DataFrame
    metric_cols = [m for m in metrics if m in results_df.columns]
    if not metric_cols:
        return
    # Build matrix: index by model, columns by metric
    df = results_df.copy()
    # Ensure 'model' column is string
    df["model"] = df["model"].astype(str)
    matrix = df.set_index("model")[metric_cols]
    # Sort by primary metric descending if available
    if primary_metric and primary_metric in matrix.columns:
        matrix = matrix.sort_values(by=primary_metric, ascending=False)
    else:
        matrix = matrix.sort_index()
    # Plot heatmap
    plt.figure(figsize=(max(6.0, len(metric_cols) * 1.2), 0.4 * len(matrix.index) + 2.0))
    im = plt.imshow(matrix.values, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Score")
    plt.yticks(range(len(matrix.index)), matrix.index, fontsize=7)
    plt.xticks(range(len(metric_cols)), metric_cols, rotation=45, ha="right", fontsize=8)
    plt.title(title or "Model Comparison Heatmap")
    # Annotate each cell with the value
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                continue
            try:
                text = f"{val:.3f}"
            except Exception:
                text = str(val)
            plt.text(j, i, text, ha="center", va="center", color="white", fontsize=6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
