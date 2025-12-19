from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def build_single_conditions_table(*, row: Dict[str, Any], prediction: Optional[float]) -> pd.DataFrame:
    payload = dict(row or {})
    payload["prediction"] = prediction
    return pd.DataFrame([payload])


def build_topk_table(
    *,
    df_trials: pd.DataFrame,
    goal: str,
    top_k: int,
) -> pd.DataFrame:
    if df_trials is None or not isinstance(df_trials, pd.DataFrame) or df_trials.empty:
        return pd.DataFrame()
    if "prediction" not in df_trials.columns:
        return pd.DataFrame()

    direction = str(goal).strip().lower()
    ascending = True if direction == "min" else False
    df = df_trials.copy()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.sort_values("prediction", ascending=ascending, kind="mergesort")
    df = df.dropna(subset=["prediction"])
    df = df.head(int(max(1, top_k)))
    df = df.reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(str(c))
        except Exception:
            continue
    return cols


def _encode_non_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
    encoded = df.copy()
    mapping: Dict[str, Dict[Any, int]] = {}
    for c in encoded.columns:
        try:
            if pd.api.types.is_numeric_dtype(encoded[c]):
                continue
        except Exception:
            pass
        try:
            series = encoded[c].astype("string")
            cats = series.dropna().unique().tolist()
            map_c = {v: i for i, v in enumerate(cats)}
            mapping[str(c)] = map_c
            encoded[c] = series.map(map_c).astype(float)
        except Exception:
            try:
                encoded[c] = pd.to_numeric(encoded[c], errors="coerce")
            except Exception:
                encoded[c] = np.nan
    return encoded, mapping


def build_training_position_plot(
    *,
    df_train: pd.DataFrame,
    df_points: pd.DataFrame,
    title: str,
    point_name: str = "input",
    max_train_points: int = 2000,
) -> Optional[Any]:
    """Plot where inference points lie relative to training distribution (numeric features only)."""

    if df_train is None or df_points is None:
        return None
    if df_train.empty or df_points.empty:
        return None

    # Use shared numeric columns.
    train_num = df_train.select_dtypes(include=[np.number])
    points_num = df_points.select_dtypes(include=[np.number])
    cols = [c for c in train_num.columns if c in points_num.columns]
    if len(cols) < 2:
        return None

    X_train = train_num[cols].copy()
    X_points = points_num[cols].copy()
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    if X_train.empty:
        return None

    if len(X_train) > int(max_train_points):
        X_train = X_train.sample(n=int(max_train_points), random_state=0)

    use_pca = len(cols) > 2
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return None

    if use_pca:
        try:
            from sklearn.decomposition import PCA  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore

            scaler = StandardScaler()
            Z_train = scaler.fit_transform(X_train.values)
            Z_points = scaler.transform(X_points.fillna(X_train.mean()).values)
            pca = PCA(n_components=2, random_state=0)
            T_train = pca.fit_transform(Z_train)
            T_points = pca.transform(Z_points)
            x_train, y_train = T_train[:, 0], T_train[:, 1]
            x_points, y_points = T_points[:, 0], T_points[:, 1]
            xlab = "PC1"
            ylab = "PC2"
        except Exception:
            use_pca = False

    if not use_pca:
        c1, c2 = cols[0], cols[1]
        x_train = X_train[c1].values
        y_train = X_train[c2].values
        x_points = X_points[c1].values
        y_points = X_points[c2].values
        xlab = str(c1)
        ylab = str(c2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_train,
            y=y_train,
            mode="markers",
            name="train",
            marker={"size": 4, "opacity": 0.25},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_points,
            y=y_points,
            mode="markers",
            name=str(point_name),
            marker={"size": 10, "opacity": 0.95, "symbol": "x"},
        )
    )
    fig.update_layout(title=str(title), xaxis_title=xlab, yaxis_title=ylab)
    return fig


def build_loss_history_plot(
    *,
    df_trials: pd.DataFrame,
    goal: str,
    title: str,
) -> Optional[Any]:
    if df_trials is None or not isinstance(df_trials, pd.DataFrame) or df_trials.empty:
        return None
    if "prediction" not in df_trials.columns:
        return None

    y = pd.to_numeric(df_trials["prediction"], errors="coerce")
    if "trial_index" in df_trials.columns:
        x = pd.to_numeric(df_trials["trial_index"], errors="coerce")
    else:
        x = pd.Series(np.arange(len(df_trials)))

    direction = str(goal).strip().lower()
    if direction == "min":
        best = y.cummin()
    else:
        best = y.cummax()

    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="objective"))
    fig.add_trace(go.Scatter(x=x, y=best, mode="lines", name="best_so_far"))

    # Log scale only if all positive.
    if (y.dropna() > 0).all():
        fig.update_layout(yaxis_type="log")
    fig.update_layout(title=str(title), xaxis_title="trial", yaxis_title="objective")
    return fig


def build_parallel_coordinates_plot(
    *,
    df_trials: pd.DataFrame,
    title: str,
    max_rows: int = 500,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Return (figure, meta) where meta may include category mappings."""

    if df_trials is None or not isinstance(df_trials, pd.DataFrame) or df_trials.empty:
        return None, {}
    if "prediction" not in df_trials.columns:
        return None, {}

    df = df_trials.copy()
    # Keep only variable columns + prediction (drop trial_index).
    drop_cols = {"trial_index"}
    cols = [c for c in df.columns if c not in drop_cols]
    df = df[cols]
    if len(df) > int(max_rows):
        df = df.sample(n=int(max_rows), random_state=0)

    df_enc, mapping = _encode_non_numeric(df)
    # Require at least two variable columns besides prediction.
    num_cols = [c for c in df_enc.columns if c != "prediction"]
    if len(num_cols) < 2:
        return None, {"category_mapping": mapping}

    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return None, {"category_mapping": mapping}

    fig = px.parallel_coordinates(
        df_enc,
        dimensions=num_cols,
        color="prediction",
        color_continuous_scale=px.colors.sequential.Viridis,
        title=str(title),
    )
    return fig, {"category_mapping": mapping}


def build_feature_importance_plot(
    *,
    df_trials: pd.DataFrame,
    title: str,
) -> Optional[Any]:
    if df_trials is None or not isinstance(df_trials, pd.DataFrame) or df_trials.empty:
        return None
    if "prediction" not in df_trials.columns:
        return None

    df = df_trials.copy()
    y = pd.to_numeric(df["prediction"], errors="coerce")
    if y.dropna().empty:
        return None

    # Encode everything to numeric and compute abs Spearman corr vs prediction.
    df_enc, _ = _encode_non_numeric(df.drop(columns=["prediction"], errors="ignore"))
    scores: Dict[str, float] = {}
    for c in df_enc.columns:
        x = pd.to_numeric(df_enc[c], errors="coerce")
        if x.dropna().empty:
            continue
        try:
            corr = x.corr(y, method="spearman")
        except Exception:
            corr = None
        if corr is None or corr != corr:
            continue
        scores[str(c)] = float(abs(corr))

    if not scores:
        return None

    imp = (
        pd.DataFrame([{"feature": k, "importance": v} for k, v in scores.items()])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    try:
        import plotly.express as px  # type: ignore
    except Exception:
        return None

    fig = px.bar(imp, x="feature", y="importance", title=str(title))
    return fig


def build_contour_plot(
    *,
    df_trials: pd.DataFrame,
    title: str,
    bins: int = 25,
) -> Optional[Any]:
    if df_trials is None or not isinstance(df_trials, pd.DataFrame) or df_trials.empty:
        return None
    if "prediction" not in df_trials.columns:
        return None

    df = df_trials.copy()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["prediction"])
    if df.empty:
        return None

    num_cols = _numeric_cols(df.drop(columns=["prediction"], errors="ignore"))
    if len(num_cols) < 2:
        return None
    xcol, ycol = num_cols[0], num_cols[1]

    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    z = pd.to_numeric(df["prediction"], errors="coerce")
    mask = x.notna() & y.notna() & z.notna()
    x = x[mask].values
    y = y[mask].values
    z = z[mask].values
    if len(x) < 5:
        return None

    # Bin into a coarse grid and take mean prediction per bin.
    x_edges = np.linspace(float(np.min(x)), float(np.max(x)), int(bins) + 1)
    y_edges = np.linspace(float(np.min(y)), float(np.max(y)), int(bins) + 1)
    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1
    grid = np.full((int(bins), int(bins)), np.nan)
    counts = np.zeros((int(bins), int(bins)), dtype=int)
    sums = np.zeros((int(bins), int(bins)), dtype=float)
    for a, b, val in zip(xi, yi, z, strict=False):
        if a < 0 or a >= int(bins) or b < 0 or b >= int(bins):
            continue
        sums[b, a] += float(val)
        counts[b, a] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = sums / np.where(counts == 0, np.nan, counts)

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        return None

    fig = go.Figure(
        data=go.Contour(
            z=grid,
            x=x_centers,
            y=y_centers,
            contours={"coloring": "heatmap"},
            colorbar={"title": "prediction"},
        )
    )
    fig.update_layout(title=str(title), xaxis_title=str(xcol), yaxis_title=str(ycol))
    return fig

