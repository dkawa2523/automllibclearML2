"""
Visualization for comparison phase.
モデルごとの R2 / MSE / RMSE をヒストグラム・バーで表示する。
"""

import pandas as pd

try:
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None

from automl_lib.clearml import report_plotly, report_table


_MINIMIZE_METRICS = {
    "mse",
    "rmse",
    "mae",
    "mape",
    "smape",
    "logloss",
    "loss",
    "error",
}


def _metric_goal(metric: str) -> str:
    key = str(metric).strip().lower()
    return "min" if key in _MINIMIZE_METRICS else "max"


def _pref(title_prefix: str | None, name: str) -> str:
    if not title_prefix:
        return name
    base = str(title_prefix).strip().strip("/")
    return f"{base}/{name}" if base else name


def render_comparison_visuals(
    logger,
    metrics_df: pd.DataFrame,
    metric_cols=None,
    *,
    title_prefix: str | None = None,
    series: str = "comparison",
) -> None:
    """
    logger: ClearML Logger
    metrics_df: DataFrame with columns ['model', 'preprocessor', 'task_id', ...metrics]
    metric_cols: list of metric column names (default: ['r2','mse','rmse'])
    """
    if metrics_df is None or metrics_df.empty:
        return
    metric_cols = metric_cols or [c for c in ["r2", "mse", "rmse"] if c in metrics_df.columns]
    metric_cols = list(metric_cols)
    if "composite_score" in metrics_df.columns and "composite_score" not in metric_cols:
        metric_cols.append("composite_score")
    metric_cols = [m for m in metric_cols if m in metrics_df.columns]

    # テーブルをそのまま送る（ランキング済みの場合もあるので先頭を確認しやすくする）
    try:
        report_table(logger, title=_pref(title_prefix, "metrics"), df=metrics_df, series=series)
    except Exception:
        pass

    if px is None:
        return

    has_run = "run_id" in metrics_df.columns
    color_key = "run_id" if has_run else ("preprocessor" if "preprocessor" in metrics_df.columns else None)

    for m in metric_cols:
        # モデル別バー
        try:
            fig_bar = px.bar(
                metrics_df,
                x="model",
                y=m,
                color=color_key,
                barmode="group" if has_run else "relative",
                title=f"{m.upper()} by model",
            )
            report_plotly(logger, title=_pref(title_prefix, f"{m}_bar"), series=series, figure=fig_bar)
        except Exception:
            pass
        # メトリクスヒストグラム
        try:
            fig_hist = px.histogram(
                metrics_df,
                x=m,
                color=("run_id" if has_run else "model"),
                marginal="box",
                title=f"{m.upper()} distribution",
            )
            report_plotly(logger, title=_pref(title_prefix, f"{m}_hist"), series=series, figure=fig_hist)
        except Exception:
            pass

        # モデル×前処理のヒートマップ（存在する場合のみ）
        if "model" not in metrics_df.columns or "preprocessor" not in metrics_df.columns:
            continue
        try:
            heat_src = metrics_df.copy()
            heat_src[m] = pd.to_numeric(heat_src[m], errors="coerce")
            heat_src = heat_src.dropna(subset=[m])
            if heat_src.empty:
                continue
            pivot = heat_src.pivot_table(index="model", columns="preprocessor", values=m, aggfunc="mean")
            if pivot is None or pivot.empty:
                continue

            goal = _metric_goal(m)
            try:
                if goal == "max":
                    model_order = pivot.max(axis=1).sort_values(ascending=False).index
                else:
                    model_order = pivot.min(axis=1).sort_values(ascending=True).index
                pivot = pivot.loc[model_order]
            except Exception:
                pass
            try:
                if goal == "max":
                    preproc_order = pivot.max(axis=0).sort_values(ascending=False).index
                else:
                    preproc_order = pivot.min(axis=0).sort_values(ascending=True).index
                pivot = pivot.loc[:, preproc_order]
            except Exception:
                pass

            n_cells = int(pivot.shape[0] * pivot.shape[1])
            text_auto = (n_cells <= 400)
            try:
                fig = px.imshow(
                    pivot,
                    aspect="auto",
                    color_continuous_scale=("RdYlGn" if goal == "max" else "RdYlGn_r"),
                    text_auto=text_auto,
                    title=f"{m.upper()} heatmap (model x preprocessor)",
                )
            except TypeError:
                fig = px.imshow(
                    pivot,
                    aspect="auto",
                    color_continuous_scale=("RdYlGn" if goal == "max" else "RdYlGn_r"),
                    title=f"{m.upper()} heatmap (model x preprocessor)",
                )
            report_plotly(logger, title=_pref(title_prefix, f"{m}_heatmap"), series=series, figure=fig)
        except Exception:
            pass


def render_model_summary_visuals(
    logger,
    model_summary_df: pd.DataFrame,
    *,
    primary_metric: str,
    title_prefix: str | None = None,
    series: str = "comparison",
) -> None:
    if model_summary_df is None or model_summary_df.empty:
        return
    if px is None:
        return

    metric = str(primary_metric).strip().lower()
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in model_summary_df.columns:
        return

    try:
        report_table(logger, title=_pref(title_prefix, "model_summary"), df=model_summary_df, series=series)
    except Exception:
        pass

    try:
        fig = px.bar(
            model_summary_df,
            x="model",
            y=mean_col,
            error_y=(std_col if std_col in model_summary_df.columns else None),
            title=f"{metric.upper()} mean by model",
        )
        report_plotly(logger, title=_pref(title_prefix, f"{metric}_mean_by_model"), series=series, figure=fig)
    except Exception:
        pass


def render_win_summary_visuals(
    logger,
    win_summary_df: pd.DataFrame,
    *,
    title_prefix: str | None = None,
    series: str = "comparison",
) -> None:
    if win_summary_df is None or win_summary_df.empty:
        return
    if px is None:
        return
    if "model" not in win_summary_df.columns or "n_wins" not in win_summary_df.columns:
        return

    try:
        fig = px.bar(
            win_summary_df,
            x="model",
            y="n_wins",
            title="Win count by model",
        )
        report_plotly(logger, title=_pref(title_prefix, "win_count_by_model"), series=series, figure=fig)
    except Exception:
        pass

    if "win_rate" in win_summary_df.columns:
        try:
            fig = px.bar(
                win_summary_df,
                x="model",
                y="win_rate",
                title="Win rate by model",
            )
            report_plotly(logger, title=_pref(title_prefix, "win_rate_by_model"), series=series, figure=fig)
        except Exception:
            pass
