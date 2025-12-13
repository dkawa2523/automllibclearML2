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


def render_comparison_visuals(logger, metrics_df: pd.DataFrame, metric_cols=None) -> None:
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
        report_table(logger, title="comparison_metrics", df=metrics_df)
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
            report_plotly(logger, title=f"{m}_bar", series="comparison", figure=fig_bar)
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
            report_plotly(logger, title=f"{m}_hist", series="comparison", figure=fig_hist)
        except Exception:
            pass


def render_model_summary_visuals(logger, model_summary_df: pd.DataFrame, *, primary_metric: str) -> None:
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
        report_table(logger, title="model_summary", df=model_summary_df, series="summary")
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
        report_plotly(logger, title=f"{metric}_mean_by_model", series="summary", figure=fig)
    except Exception:
        pass


def render_win_summary_visuals(logger, win_summary_df: pd.DataFrame) -> None:
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
        report_plotly(logger, title="win_count_by_model", series="summary", figure=fig)
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
            report_plotly(logger, title="win_rate_by_model", series="summary", figure=fig)
        except Exception:
            pass
