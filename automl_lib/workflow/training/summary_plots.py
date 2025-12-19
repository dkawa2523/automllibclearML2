from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from automl_lib.training.interpretation import compute_shap_importance, extract_feature_importance
from automl_lib.training.visualization import build_plotly_interpolation_space, build_plotly_pred_vs_actual


def _placeholder_plotly(title: str, message: str):
    try:  # pragma: no cover - optional dependency (declared in requirements.txt)
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure()
        fig.add_annotation(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            text=str(message),
            showarrow=False,
            align="left",
        )
        fig.update_layout(title=str(title), xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    except Exception:
        return None


def log_leaderboard_bar(
    *,
    clearml_mgr: Any,
    results_df_best: pd.DataFrame,
    metric: Optional[str],
    goal: str,
) -> None:
    if not getattr(clearml_mgr, "logger", None):
        return
    metric = str(metric or "").strip()
    if not metric or results_df_best is None or results_df_best.empty or metric not in results_df_best.columns:
        return
    try:
        import plotly.express as px  # type: ignore

        df_bar = results_df_best.dropna(subset=[metric]).copy()
        df_bar = df_bar.sort_values(by=metric, ascending=(goal == "min")).head(20)
        fig = px.bar(
            df_bar,
            x="model",
            y=metric,
            hover_data=["preprocessor"],
            title=f"Leaderboard ({metric}, {goal})",
        )
        clearml_mgr.logger.report_plotly(
            title="02_Leaderboard",
            series="leaderboard",
            iteration=0,
            figure=fig,
        )
    except Exception:
        return


def log_training_summary_dashboard(
    *,
    clearml_mgr: Any,
    cfg: Any,
    problem_type: str,
    df_links: Optional[pd.DataFrame],
    df_links_ranked: Optional[pd.DataFrame],
    recommended_df: Optional[pd.DataFrame],
    recommend_metric: Optional[str],
    summary_plots_mode: str,
    full_pipeline: Any,
    X_train: Any,
    y_train: Any,
) -> None:
    if not getattr(clearml_mgr, "logger", None):
        return

    try:
        tasks_total = None
        top_k_display = 20
        if df_links is not None and not df_links.empty:
            df_tasks_all = df_links_ranked if (df_links_ranked is not None and not df_links_ranked.empty) else df_links
            try:
                tasks_total = int(len(df_tasks_all))
            except Exception:
                tasks_total = None
            try:
                df_tasks_show = df_tasks_all.head(top_k_display) if (tasks_total and tasks_total > top_k_display) else df_tasks_all
            except Exception:
                df_tasks_show = df_tasks_all
            clearml_mgr.report_table("03_Leaderboard Table", df_tasks_show, series="leaderboard")

            try:
                import plotly.express as px  # type: ignore

                metric_key = str(recommend_metric or "").strip()
                df_trade = df_tasks_all.copy()
                df_trade["_metric"] = pd.to_numeric(df_trade.get(metric_key), errors="coerce") if metric_key else pd.NA
                df_trade["_train_seconds"] = pd.to_numeric(df_trade.get("train_seconds"), errors="coerce")
                df_trade["_model_size_bytes"] = pd.to_numeric(df_trade.get("model_size_bytes"), errors="coerce")
                df_trade = df_trade[df_trade["_metric"].notna() & df_trade["_train_seconds"].notna()]
                if not df_trade.empty:
                    fig = px.scatter(
                        df_trade,
                        x="_train_seconds",
                        y="_metric",
                        size="_model_size_bytes",
                        color="model" if "model" in df_trade.columns else None,
                        hover_data=[c for c in ["preprocessor", "task_id", "model_id"] if c in df_trade.columns],
                        title=f"Tradeoff: {metric_key} vs train_seconds",
                    )
                    clearml_mgr.logger.report_plotly(
                        title="04_Tradeoff",
                        series="tradeoff",
                        iteration=0,
                        figure=fig,
                    )
            except Exception:
                pass

        if recommended_df is not None and not recommended_df.empty:
            clearml_mgr.report_table("01_Recommended Model", recommended_df, series="overview")

            if str(summary_plots_mode).strip().lower() == "best" and bool(getattr(getattr(cfg, "output", None), "generate_plots", False)):
                try:
                    rec_row = recommended_df.iloc[0].to_dict()
                    model_name = str(rec_row.get("model") or "").strip()
                    preproc_name = str(rec_row.get("preprocessor") or "").strip()

                    best_series = "recommended_model"

                    # 05: Predicted vs Actual (Plotly)
                    try:
                        if problem_type == "regression":
                            X_scatter = X_train
                            y_scatter = y_train
                            max_points = 2000
                            try:
                                n = int(len(y_train))
                            except Exception:
                                n = 0
                            if n and n > max_points:
                                rng = np.random.default_rng(0)
                                idx = rng.choice(n, size=max_points, replace=False)
                                if hasattr(X_train, "iloc"):
                                    X_scatter = X_train.iloc[idx]  # type: ignore[attr-defined]
                                else:
                                    X_scatter = np.asarray(X_train)[idx]
                                if hasattr(y_train, "iloc"):
                                    y_scatter = y_train.iloc[idx]  # type: ignore[attr-defined]
                                else:
                                    y_scatter = np.asarray(y_train)[idx]
                            y_true = np.asarray(y_scatter)
                            y_pred = np.asarray(full_pipeline.predict(X_scatter))
                            fig = build_plotly_pred_vs_actual(
                                y_true=y_true,
                                y_pred=y_pred,
                                title=f"Predicted vs Actual: {model_name} ({preproc_name})" if model_name and preproc_name else "Predicted vs Actual",
                                add_regression_line=True,
                            )
                            if fig is None:
                                fig = _placeholder_plotly("Predicted vs Actual", "plotly is not available")
                        else:
                            fig = _placeholder_plotly(
                                "Predicted vs Actual",
                                f"Not available for problem_type={problem_type}",
                            )
                        if fig is not None:
                            clearml_mgr.logger.report_plotly(
                                title="05_Scatter Plot of Recommended Model",
                                series=best_series,
                                iteration=0,
                                figure=fig,
                            )
                    except Exception:
                        fig = _placeholder_plotly(
                            "Predicted vs Actual",
                            "Predicted-vs-Actual plot generation failed.",
                        )
                        if fig is not None:
                            clearml_mgr.logger.report_plotly(
                                title="05_Scatter Plot of Recommended Model",
                                series=best_series,
                                iteration=0,
                                figure=fig,
                            )

                    # 06: Feature Importance (Plotly)
                    try:
                        fig_fi = None
                        if getattr(getattr(cfg, "interpretation", None), "compute_feature_importance", False):
                            try:
                                fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
                                fnames = fitted_preprocessor.get_feature_names_out()
                            except Exception:
                                fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
                                transformed = fitted_preprocessor.transform(X_train)
                                fnames = [f"f{i}" for i in range(transformed.shape[1])]
                            imp_df = extract_feature_importance(full_pipeline.named_steps["model"], list(fnames))
                            if imp_df is not None and not imp_df.empty:
                                try:
                                    import plotly.express as px  # type: ignore

                                    top_imp = imp_df.head(30).copy().sort_values(by="importance", ascending=True)
                                    fig_fi = px.bar(
                                        top_imp,
                                        x="importance",
                                        y="feature",
                                        orientation="h",
                                        title=f"Feature Importance: {model_name} ({preproc_name})" if model_name and preproc_name else "Feature Importance",
                                    )
                                except Exception:
                                    fig_fi = None
                        if fig_fi is None:
                            msg = (
                                "Feature importance is disabled (interpretation.compute_feature_importance=false)."
                                if not getattr(getattr(cfg, "interpretation", None), "compute_feature_importance", False)
                                else "Feature importance is not available for this model."
                            )
                            fig_fi = _placeholder_plotly("Feature Importance", msg)
                        if fig_fi is not None:
                            clearml_mgr.logger.report_plotly(
                                title="06_Feature Importance from Recommended Model",
                                series=best_series,
                                iteration=0,
                                figure=fig_fi,
                            )
                    except Exception:
                        fig_fi = _placeholder_plotly(
                            "Feature Importance",
                            "Feature importance plot generation failed.",
                        )
                        if fig_fi is not None:
                            clearml_mgr.logger.report_plotly(
                                title="06_Feature Importance from Recommended Model",
                                series=best_series,
                                iteration=0,
                                figure=fig_fi,
                            )

                    # 07: Interpolation space (Plotly)
                    interp_metric = "07_Interpolation space: Recommended Model"
                    if model_name and preproc_name:
                        interp_metric = f"07_Interpolation space: {model_name} ({preproc_name})"
                    elif model_name:
                        interp_metric = f"07_Interpolation space: {model_name}"
                    try:
                        Xt_full = full_pipeline.named_steps["preprocessor"].transform(X_train)
                        y_arr = np.asarray(y_train)
                        if y_arr.ndim != 1:
                            y_arr = y_arr.ravel()
                        fig_interp = build_plotly_interpolation_space(
                            Xt_full,
                            y_arr,
                            title=f"Interpolation space: {model_name}" if model_name else "Interpolation space",
                        )
                        if fig_interp is None:
                            fig_interp = _placeholder_plotly(
                                "Interpolation space",
                                "Interpolation-space plot is not available (PCA/SVD projection failed or plotly missing).",
                            )
                        if fig_interp is not None:
                            clearml_mgr.logger.report_plotly(
                                title=interp_metric,
                                series=best_series,
                                iteration=0,
                                figure=fig_interp,
                            )
                    except Exception:
                        fig_interp = _placeholder_plotly(
                            "Interpolation space",
                            "Interpolation-space plot generation failed.",
                        )
                        if fig_interp is not None:
                            clearml_mgr.logger.report_plotly(
                                title=interp_metric,
                                series=best_series,
                                iteration=0,
                                figure=fig_interp,
                            )

                    # 08: SHAP values (Plotly)
                    try:
                        fig_shap = None
                        if getattr(getattr(cfg, "interpretation", None), "compute_shap", False):
                            shap_df = compute_shap_importance(
                                full_pipeline,
                                X_train,
                                max_display=30,
                                sample_size=200,
                                max_features=2000,
                            )
                            if shap_df is not None and not shap_df.empty:
                                try:
                                    import plotly.express as px  # type: ignore

                                    df_bar = shap_df.sort_values(by="shap_importance", ascending=True)
                                    fig_shap = px.bar(
                                        df_bar,
                                        x="shap_importance",
                                        y="feature",
                                        orientation="h",
                                        title=f"SHAP (mean |value|): {model_name} ({preproc_name})" if model_name and preproc_name else "SHAP (mean |value|)",
                                    )
                                except Exception:
                                    fig_shap = None
                        if fig_shap is None:
                            msg = (
                                "SHAP is disabled (interpretation.compute_shap=false)."
                                if not getattr(getattr(cfg, "interpretation", None), "compute_shap", False)
                                else "SHAP summary is not available (missing dependency or too many features)."
                            )
                            fig_shap = _placeholder_plotly("SHAP values", msg)
                        if fig_shap is not None:
                            clearml_mgr.logger.report_plotly(
                                title="08_SHAP values",
                                series=best_series,
                                iteration=0,
                                figure=fig_shap,
                            )
                    except Exception:
                        fig_shap = _placeholder_plotly(
                            "SHAP values",
                            "SHAP plot generation failed.",
                        )
                        if fig_shap is not None:
                            clearml_mgr.logger.report_plotly(
                                title="08_SHAP values",
                                series=best_series,
                                iteration=0,
                                figure=fig_shap,
                            )
                except Exception:
                    pass
    except Exception:
        return

