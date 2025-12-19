from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from automl_lib.integrations.clearml.properties import set_user_properties
from automl_lib.integrations.clearml.naming import build_tags, task_name
from automl_lib.integrations.clearml.manager import ClearMLManager, _import_clearml
from automl_lib.config.schemas import TrainingConfig
from automl_lib.registry.metrics import is_loss_metric

from automl_lib.training.ensemble import build_stacking, build_voting
from automl_lib.training.interpretation import (
    compute_shap_importance,
    extract_feature_importance,
    plot_feature_importance,
)
from automl_lib.training.model_factory import _get_model_class
from automl_lib.training.reporting import report_metric_scalars, save_confusion_matrices, save_roc_pr_curves
from automl_lib.training.visualization import (
    build_plotly_histogram,
    build_plotly_interpolation_space,
    build_plotly_pred_vs_actual,
    build_plotly_residual_scatter,
    plot_interpolation_space,
    plot_predicted_vs_actual,
    plot_residual_hist,
    plot_residual_scatter,
)

from .estimators import build_estimator_with_defaults, maybe_wrap_with_target_scaler


def _safe_artifact_stem(model_label: str, preproc_name: str) -> str:
    return (
        f"{str(model_label).replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_"
        f"{str(preproc_name).replace('|', '_')}"
    )


def _is_identity_preprocessor(pipeline) -> bool:
    try:
        pre_step = pipeline.named_steps.get("preprocessor")
    except Exception:
        return False
    try:
        from sklearn.preprocessing import FunctionTransformer

        if isinstance(pre_step, FunctionTransformer):
            return True
    except Exception:
        pass
    return isinstance(pre_step, str) and pre_step == "passthrough"


def _maybe_score_values(pipeline, X_part):
    try:
        if hasattr(pipeline, "predict_proba"):
            return pipeline.predict_proba(X_part)
    except Exception:
        pass
    try:
        if hasattr(pipeline, "decision_function"):
            return pipeline.decision_function(X_part)
    except Exception:
        pass
    return None


def _compute_classification_roc_auc_ovr(
    *,
    y_true: np.ndarray,
    scores,
    classes: Optional[List[Any]],
) -> Optional[float]:
    try:
        scores_arr = np.asarray(scores)
        roc_val = None
        if scores_arr.ndim == 2:
            if scores_arr.shape[1] == 2 and classes and len(classes) >= 2:
                y_bin = (np.asarray(y_true) == classes[1]).astype(int)
                roc_val = float(roc_auc_score(y_bin, scores_arr[:, 1]))
            elif classes and len(classes) == scores_arr.shape[1]:
                mapping = {c: i for i, c in enumerate(classes)}
                y_enc = np.asarray([mapping.get(v, -1) for v in np.asarray(y_true)], dtype=int)
                mask = y_enc >= 0
                if mask.any():
                    roc_val = float(
                        roc_auc_score(
                            y_enc[mask],
                            scores_arr[mask],
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(len(classes))),
                        )
                    )
        elif scores_arr.ndim == 1 and classes and len(classes) >= 2:
            y_bin = (np.asarray(y_true) == classes[1]).astype(int)
            roc_val = float(roc_auc_score(y_bin, scores_arr))
        return roc_val
    except Exception:
        return None


def run_model_tasks(
    *,
    cfg: TrainingConfig,
    ctx,
    run_id: str,
    dataset_id_for_load: Optional[str],
    dataset_key: str,
    output_dir: Path,
    preprocessors: List[Tuple[str, object]],
    results_df_best: pd.DataFrame,
    X: Any,
    y: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    problem_type: str,
    metrics: List[str],
    primary_metric_model: Optional[str],
    clearml_mgr: ClearMLManager,
    train_models_project: str,
    feature_types: Dict[str, List[str]],
    has_preproc_contract: bool,
    preproc_bundle: Any,
    preproc_schema_src: Optional[Path],
    preproc_manifest_src: Optional[Path],
    preproc_summary_src: Optional[Path],
    preproc_recipe_src: Optional[Path],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Train and register the best pipeline per model as separate ClearML child tasks."""

    training_task_ids: List[str] = []
    model_task_records: List[Dict[str, Any]] = []

    best_rows_by_model: Dict[str, pd.Series] = {row["model"]: row for _, row in results_df_best.iterrows()}

    models_all_dir = output_dir / "models"
    models_all_dir.mkdir(exist_ok=True)

    visual_pred_dir = output_dir / "scatter_plots"
    visual_resid_scatter_dir = output_dir / "residual_scatter"
    visual_resid_hist_dir = output_dir / "residual_hist"
    visual_interp_dir = output_dir / "interpolation_space"
    visual_fi_dir = output_dir / "feature_importances_models"
    visual_shap_dir = output_dir / "shap_summaries_models"
    visual_confusion_dir = output_dir / "confusion_matrices"
    visual_roc_dir = output_dir / "roc_curves"
    visual_pr_dir = output_dir / "pr_curves"

    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.predicted_vs_actual:
        visual_pred_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_scatter:
        visual_resid_scatter_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_hist:
        visual_resid_hist_dir.mkdir(exist_ok=True)
    visual_interp_dir.mkdir(exist_ok=True)
    if problem_type == "classification" and cfg.output.generate_plots:
        visual_confusion_dir.mkdir(exist_ok=True)
        visual_roc_dir.mkdir(exist_ok=True)
        visual_pr_dir.mkdir(exist_ok=True)
    if cfg.visualizations.feature_importance:
        visual_fi_dir.mkdir(exist_ok=True)
    if cfg.visualizations.shap_summary:
        visual_shap_dir.mkdir(exist_ok=True)

    proj_csv_path = visual_interp_dir / "feature_space_projection.csv"
    wrote_projection_csv = proj_csv_path.exists()

    for model_label, row in best_rows_by_model.items():
        model_task_mgr: Optional[ClearMLManager] = None
        record: Dict[str, Any] = {}
        model_id = None

        preproc_name = None
        safe_name = None
        pipeline_best = None
        y_pred_train = None
        y_pred_test = None
        y_train_array = None
        y_pred_for_plots = None
        residuals_for_plots = None
        r2_for_plots = None
        train_metrics_display: Dict[str, Optional[float]] = {}
        train_metrics_std: Dict[str, Optional[float]] = {}
        test_metrics_display: Dict[str, Optional[float]] = {}
        test_metrics_std: Dict[str, Optional[float]] = {}
        exported_with_preproc_bundle = False
        n_features_transformed = None
        model_size_bytes = None
        train_seconds = None
        predict_seconds = None
        predict_train_seconds = None
        predict_test_seconds = None
        scores_for_plots_train = None
        scores_for_plots_test = None

        try:
            preproc_name = row["preprocessor"]
            params = row["params"] if isinstance(row.get("params"), dict) else {}
            safe_name = _safe_artifact_stem(model_label, str(preproc_name))
            record = {
                "model": model_label,
                "preprocessor": preproc_name,
                "params": params,
                "artifact_stem": safe_name,
                "status": "pending",
                "error": "",
                "train_seconds": None,
                "predict_seconds": None,
                "predict_train_seconds": None,
                "predict_test_seconds": None,
                "model_size_bytes": None,
                "num_features": None,
                "task_id": "",
                "url": "",
                "link_html": "",
            }

            transformer = None
            for name, ct in preprocessors:
                if name == preproc_name:
                    transformer = ct
                    break
            if transformer is None:
                transformer = preprocessors[0][1]

            estimator_obj = None
            if str(model_label).startswith("Stacking") or str(model_label).startswith("Voting"):
                if str(model_label).startswith("Stacking"):
                    base_names = cfg.ensembles.stacking.estimators
                    final_name = cfg.ensembles.stacking.final_estimator
                    ests: List[Tuple[str, object]] = []
                    for bn in base_names:
                        try:
                            cls_bn = _get_model_class(bn, problem_type)
                            ests.append((bn, cls_bn()))
                        except Exception:
                            pass
                    if not ests:
                        record["status"] = "skipped"
                        record["error"] = "stacking base estimators are empty"
                        continue

                    final_est = None
                    if final_name:
                        try:
                            final_cls = _get_model_class(final_name, problem_type)
                            final_est = final_cls()
                        except Exception:
                            final_est = None
                    if final_est is None:
                        try:
                            if problem_type == "regression":
                                from sklearn.linear_model import LinearRegression

                                final_est = LinearRegression()
                            else:
                                from sklearn.linear_model import LogisticRegression

                                final_est = LogisticRegression(max_iter=1000)
                        except Exception:
                            record["status"] = "skipped"
                            record["error"] = "could not build default stacking final estimator"
                            continue
                    estimator_obj = build_stacking(transformer, ests, final_est, problem_type).named_steps["model"]
                else:
                    base_names = cfg.ensembles.voting.estimators
                    voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
                    ests: List[Tuple[str, object]] = []
                    for bn in base_names:
                        try:
                            cls_bn = _get_model_class(bn, problem_type)
                            ests.append((bn, cls_bn()))
                        except Exception:
                            pass
                    if not ests:
                        record["status"] = "skipped"
                        record["error"] = "voting base estimators are empty"
                        continue
                    estimator_obj = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
            else:
                try:
                    cls = _get_model_class(str(model_label), problem_type)
                except Exception as exc:
                    record["status"] = "skipped"
                    record["error"] = f"_get_model_class failed: {exc}"
                    continue
                try:
                    estimator_obj, _ = build_estimator_with_defaults(
                        str(model_label),
                        cls,
                        params,
                        problem_type,
                        cfg,
                        len(y_train),
                    )
                except Exception as exc:
                    record["status"] = "skipped"
                    record["error"] = f"build_estimator_with_defaults failed: {exc}"
                    continue

            if problem_type == "regression":
                try:
                    estimator_obj = maybe_wrap_with_target_scaler(estimator_obj, cfg, problem_type)
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = f"maybe_wrap_with_target_scaler failed: {exc}"
                    continue

            from sklearn.pipeline import Pipeline as SKPipeline

            pipeline_best = SKPipeline(
                [
                    ("preprocessor", transformer),
                    ("model", estimator_obj),
                ]
            )

            try:
                t0 = time.perf_counter()
                pipeline_best.fit(X_train, y_train)
                train_seconds = float(time.perf_counter() - t0)
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"fit failed: {exc}"
                continue

            record["status"] = "ok"
            record["error"] = ""

            y_train_array = np.asarray(y_train)

            try:
                t1 = time.perf_counter()
                y_pred_train = np.asarray(pipeline_best.predict(X_train))
                predict_train_seconds = float(time.perf_counter() - t1)
                predict_seconds = predict_train_seconds
            except Exception:
                y_pred_train = None

            metrics_requested = [str(m).strip().lower() for m in (metrics or []) if str(m).strip()]

            if y_pred_train is not None:
                if problem_type == "regression":
                    try:
                        y_pred_for_plots = y_pred_train
                        residuals_for_plots = y_train_array - y_pred_train
                        r2_for_plots = float(r2_score(y_train_array, y_pred_train))
                    except Exception:
                        y_pred_for_plots = None
                        residuals_for_plots = None
                        r2_for_plots = None
                    mse_val = None
                    rmse_val = None
                    mae_val = None
                    if y_pred_for_plots is not None:
                        try:
                            resid_tmp = residuals_for_plots if residuals_for_plots is not None else (y_train_array - y_pred_for_plots)
                            mse_val = float(np.mean(resid_tmp ** 2))
                            rmse_val = float(np.sqrt(mse_val))
                            mae_val = float(mean_absolute_error(y_train_array, y_pred_for_plots))
                        except Exception:
                            pass
                    train_metrics_display = {"R2": r2_for_plots, "MSE": mse_val, "RMSE": rmse_val, "MAE": mae_val}
                    train_metrics_std = {"r2": r2_for_plots, "mse": mse_val, "rmse": rmse_val, "mae": mae_val}
                else:
                    y_true_train = y_train_array
                    y_pred_labels_train = y_pred_train
                    if "accuracy" in metrics_requested:
                        try:
                            val = float(accuracy_score(y_true_train, y_pred_labels_train))
                            train_metrics_display["accuracy"] = val
                            train_metrics_std["accuracy"] = val
                        except Exception:
                            pass
                    if "precision_macro" in metrics_requested:
                        try:
                            val = float(precision_score(y_true_train, y_pred_labels_train, average="macro", zero_division=0))
                            train_metrics_display["precision_macro"] = val
                            train_metrics_std["precision_macro"] = val
                        except Exception:
                            pass
                    if "recall_macro" in metrics_requested:
                        try:
                            val = float(recall_score(y_true_train, y_pred_labels_train, average="macro", zero_division=0))
                            train_metrics_display["recall_macro"] = val
                            train_metrics_std["recall_macro"] = val
                        except Exception:
                            pass
                    if "f1_macro" in metrics_requested:
                        try:
                            val = float(f1_score(y_true_train, y_pred_labels_train, average="macro", zero_division=0))
                            train_metrics_display["f1_macro"] = val
                            train_metrics_std["f1_macro"] = val
                        except Exception:
                            pass
                    if "roc_auc_ovr" in metrics_requested or "roc_auc" in metrics_requested:
                        try:
                            scores = _maybe_score_values(pipeline_best, X_train)
                            if scores is not None:
                                scores_for_plots_train = scores
                                classes = None
                                try:
                                    model_step = pipeline_best.named_steps.get("model")
                                    classes = getattr(model_step, "classes_", None)
                                    classes = list(classes) if classes is not None else None
                                except Exception:
                                    classes = None
                                roc_val = _compute_classification_roc_auc_ovr(
                                    y_true=np.asarray(y_true_train),
                                    scores=scores,
                                    classes=classes,
                                )
                                if roc_val is not None:
                                    train_metrics_display["roc_auc_ovr"] = roc_val
                                    train_metrics_std["roc_auc_ovr"] = roc_val
                        except Exception:
                            pass

            if X_test is not None and len(X_test) > 0:
                try:
                    t2 = time.perf_counter()
                    y_pred_test = np.asarray(pipeline_best.predict(X_test))
                    predict_test_seconds = float(time.perf_counter() - t2)
                    if problem_type == "regression":
                        mse_test = float(mean_squared_error(y_test, y_pred_test))
                        test_metrics_display = {
                            "R2": float(r2_score(y_test, y_pred_test)),
                            "MSE": mse_test,
                            "RMSE": float(np.sqrt(mse_test)),
                            "MAE": float(mean_absolute_error(y_test, y_pred_test)),
                        }
                        test_metrics_std = {
                            "r2": test_metrics_display.get("R2"),
                            "mse": test_metrics_display.get("MSE"),
                            "rmse": test_metrics_display.get("RMSE"),
                            "mae": test_metrics_display.get("MAE"),
                        }
                    else:
                        y_true_test = np.asarray(y_test)
                        y_pred_labels_test = y_pred_test
                        if "accuracy" in metrics_requested:
                            try:
                                val = float(accuracy_score(y_true_test, y_pred_labels_test))
                                test_metrics_display["accuracy"] = val
                                test_metrics_std["accuracy"] = val
                            except Exception:
                                pass
                        if "precision_macro" in metrics_requested:
                            try:
                                val = float(precision_score(y_true_test, y_pred_labels_test, average="macro", zero_division=0))
                                test_metrics_display["precision_macro"] = val
                                test_metrics_std["precision_macro"] = val
                            except Exception:
                                pass
                        if "recall_macro" in metrics_requested:
                            try:
                                val = float(recall_score(y_true_test, y_pred_labels_test, average="macro", zero_division=0))
                                test_metrics_display["recall_macro"] = val
                                test_metrics_std["recall_macro"] = val
                            except Exception:
                                pass
                        if "f1_macro" in metrics_requested:
                            try:
                                val = float(f1_score(y_true_test, y_pred_labels_test, average="macro", zero_division=0))
                                test_metrics_display["f1_macro"] = val
                                test_metrics_std["f1_macro"] = val
                            except Exception:
                                pass
                        if "roc_auc_ovr" in metrics_requested or "roc_auc" in metrics_requested:
                            try:
                                scores = _maybe_score_values(pipeline_best, X_test)
                                if scores is not None:
                                    scores_for_plots_test = scores
                                    classes = None
                                    try:
                                        model_step = pipeline_best.named_steps.get("model")
                                        classes = getattr(model_step, "classes_", None)
                                        classes = list(classes) if classes is not None else None
                                    except Exception:
                                        classes = None
                                    roc_val = _compute_classification_roc_auc_ovr(
                                        y_true=np.asarray(y_true_test),
                                        scores=scores,
                                        classes=classes,
                                    )
                                    if roc_val is not None:
                                        test_metrics_display["roc_auc_ovr"] = roc_val
                                        test_metrics_std["roc_auc_ovr"] = roc_val
                            except Exception:
                                pass
                except Exception:
                    test_metrics_display = {}
                    test_metrics_std = {}

            if problem_type == "regression" and cfg.output.generate_plots:
                if cfg.visualizations.predicted_vs_actual and y_pred_for_plots is not None:
                    scatter_path = visual_pred_dir / f"{safe_name}.png"
                    try:
                        plot_predicted_vs_actual(
                            pipeline_best,
                            X_train,
                            y_train,
                            scatter_path,
                            title=f"{model_label} ({preproc_name})",
                            predictions=y_pred_for_plots,
                            r2_override=r2_for_plots,
                            add_regression_line=True,
                        )
                        if residuals_for_plots is not None:
                            pd.DataFrame(
                                {
                                    "actual": y_train_array,
                                    "predicted": y_pred_for_plots,
                                    "residual": residuals_for_plots,
                                    "split": "train",
                                }
                            ).to_csv(scatter_path.with_suffix(".csv"), index=False)
                    except Exception:
                        pass

                if cfg.visualizations.residual_scatter and y_pred_for_plots is not None and residuals_for_plots is not None:
                    resid_scatter_path = visual_resid_scatter_dir / f"{safe_name}.png"
                    try:
                        plot_residual_scatter(
                            pipeline_best,
                            X_train,
                            y_train,
                            resid_scatter_path,
                            title=f"Residual scatter: {model_label} ({preproc_name})",
                            predictions=y_pred_for_plots,
                            residuals=residuals_for_plots,
                        )
                        pd.DataFrame(
                            {
                                "predicted": y_pred_for_plots,
                                "residual": residuals_for_plots,
                                "split": "train",
                            }
                        ).to_csv(resid_scatter_path.with_suffix(".csv"), index=False)
                    except Exception:
                        pass

                if cfg.visualizations.residual_hist and residuals_for_plots is not None:
                    resid_hist_path = visual_resid_hist_dir / f"{safe_name}.png"
                    try:
                        plot_residual_hist(
                            pipeline_best,
                            X_train,
                            y_train,
                            resid_hist_path,
                            title=f"Residual histogram: {model_label} ({preproc_name})",
                            residuals=residuals_for_plots,
                        )
                        pd.DataFrame({"residual": residuals_for_plots, "split": "train"}).to_csv(
                            resid_hist_path.with_suffix(".csv"),
                            index=False,
                        )
                    except Exception:
                        pass

            if problem_type == "classification" and cfg.output.generate_plots and y_pred_train is not None:
                try:
                    y_true_cm = np.asarray(y_test) if y_pred_test is not None else y_train_array
                    y_pred_cm = np.asarray(y_pred_test) if y_pred_test is not None else np.asarray(y_pred_train)

                    labels = None
                    try:
                        model_step = pipeline_best.named_steps.get("model")
                        classes = getattr(model_step, "classes_", None)
                        if classes is not None:
                            labels = list(classes)
                    except Exception:
                        labels = None
                    if not labels:
                        try:
                            labels = list(pd.unique(pd.Series(list(y_true_cm) + list(y_pred_cm))))
                        except Exception:
                            labels = None

                    save_confusion_matrices(
                        y_true=y_true_cm,
                        y_pred=y_pred_cm,
                        labels=labels,
                        out_dir=visual_confusion_dir,
                        base_name=safe_name,
                        title_prefix=f"{model_label} ({preproc_name}) - ",
                    )

                    if "roc_auc_ovr" in metrics_requested or "roc_auc" in metrics_requested:
                        use_test = bool(y_pred_test is not None and X_test is not None)
                        X_curve = X_test if use_test else X_train
                        scores_curve = scores_for_plots_test if use_test else scores_for_plots_train
                        if scores_curve is None:
                            scores_curve = _maybe_score_values(pipeline_best, X_curve)
                        if scores_curve is not None:
                            save_roc_pr_curves(
                                y_true=y_true_cm,
                                scores=scores_curve,
                                classes=labels,
                                out_roc_dir=visual_roc_dir,
                                out_pr_dir=visual_pr_dir,
                                base_name=safe_name,
                                title_prefix=f"{model_label} ({preproc_name}) - ",
                            )
                except Exception:
                    pass

            try:
                Xt_full = pipeline_best.named_steps["preprocessor"].transform(X_train)
                try:
                    if hasattr(Xt_full, "shape") and len(Xt_full.shape) > 1:
                        n_features_transformed = int(Xt_full.shape[1])
                except Exception:
                    n_features_transformed = None
                try:
                    interp_path = visual_interp_dir / f"{safe_name}.png"
                    plot_interpolation_space(Xt_full, y_train_array, interp_path, title=f"Interpolation space: {model_label}")
                except Exception:
                    pass
                if not wrote_projection_csv:
                    try:
                        from sklearn.decomposition import PCA

                        comps = PCA(n_components=2).fit_transform(Xt_full)
                        proj_df = pd.DataFrame(
                            {
                                "pc1": comps[:, 0],
                                "pc2": comps[:, 1] if comps.shape[1] > 1 else 0,
                                "target": y_train_array,
                                "model": model_label,
                            }
                        )
                        proj_df.to_csv(proj_csv_path, index=False)
                        wrote_projection_csv = True
                    except Exception:
                        pass
            except Exception:
                pass

            model_path = models_all_dir / f"{safe_name}.joblib"
            pipeline_export = pipeline_best
            if _is_identity_preprocessor(pipeline_best) and has_preproc_contract:
                if preproc_bundle is None:
                    raise RuntimeError(
                        "Preprocessed dataset contract detected but 'preprocessing/bundle.joblib' could not be loaded. "
                        "Re-run preprocessing to regenerate the dataset contract."
                    )
                from sklearn.pipeline import Pipeline as SkPipeline

                pipeline_export = SkPipeline(
                    [
                        ("preprocessor", preproc_bundle),
                        ("model", pipeline_best.named_steps["model"]),
                    ]
                )
                exported_with_preproc_bundle = True

            dump(pipeline_export, model_path)
            try:
                if model_path.exists():
                    model_size_bytes = int(model_path.stat().st_size)
            except Exception:
                model_size_bytes = None

            model_task_name = task_name("training_child", ctx, model=model_label, preproc=preproc_name)
            child_tags = build_tags(ctx, phase="training", model=model_label, preproc=preproc_name, extra=[problem_type])

            task_obj = None
            if cfg.clearml and cfg.clearml.enabled:
                _, _, TaskCls, TaskTypesCls = _import_clearml()
                if TaskCls is not None and TaskTypesCls is not None:
                    try:
                        task_obj = TaskCls.create(
                            project_name=train_models_project,
                            task_name=model_task_name,
                            task_type=getattr(TaskTypesCls, "training", None),
                            # These child tasks are for logging (not remote execution). Keep creation lightweight/stable.
                            add_task_init_call=False,
                            detect_repository=False,
                        )
                        if clearml_mgr.task:
                            try:
                                if hasattr(task_obj, "add_parent"):
                                    task_obj.add_parent(clearml_mgr.task.id)
                                else:
                                    task_obj.set_parent(clearml_mgr.task.id)
                            except Exception:
                                try:
                                    task_obj.set_parent(clearml_mgr.task.id)
                                except Exception:
                                    pass
                        try:
                            task_obj.add_tags(child_tags)
                        except Exception:
                            pass
                        if cfg.clearml.queue and not cfg.clearml.run_tasks_locally:
                            try:
                                task_obj.set_parameter("requested_queue", cfg.clearml.queue)
                            except Exception:
                                pass
                    except Exception:
                        task_obj = None

            # Important: if we fail to create a dedicated child task, do NOT fall back to
            # reusing Task.current_task() (which would pollute the training-summary task).
            child_clearml_cfg = cfg.clearml if task_obj is not None else None
            model_task_mgr = ClearMLManager(
                child_clearml_cfg,
                task_name=model_task_name,
                task_type="training",
                default_project=train_models_project,
                project=train_models_project,
                parent=clearml_mgr.task.id if clearml_mgr.task else None,
                existing_task=task_obj,
                extra_tags=child_tags,
            )

            try:
                model_task_mgr.connect_params_sections(
                    {
                        "Input": {
                            "dataset_id": dataset_id_for_load or "",
                            "target_column": cfg.data.target_column or getattr(y, "name", "target") or "target",
                            "preprocessor": str(preproc_name),
                        },
                        "Model": {
                            "name": model_label,
                            "class": f"{type(estimator_obj).__module__}.{type(estimator_obj).__name__}",
                            "params": params,
                        },
                        "Training": {
                            "problem_type": str(problem_type),
                            "test_size": float(cfg.data.test_size),
                            "random_seed": int(cfg.data.random_seed),
                            "cv_folds": int(cfg.cross_validation.n_folds) if cfg.cross_validation.n_folds else None,
                            "cv_shuffle": bool(cfg.cross_validation.shuffle),
                        },
                    }
                )
            except Exception:
                pass

            try:
                if model_task_mgr.task:
                    dataset_role = (
                        "preprocessed"
                        if has_preproc_contract
                        else "raw"
                    )
                    numeric_cols = list(feature_types.get("numeric") or [])
                    categorical_cols = list(feature_types.get("categorical") or [])
                    feature_types_summary = {
                        "numeric": {"n": len(numeric_cols), "sample": numeric_cols[:20]},
                        "categorical": {"n": len(categorical_cols), "sample": categorical_cols[:20]},
                    }
                    dataset_conf: Dict[str, Any] = {
                        "dataset_id": dataset_id_for_load or "",
                        "dataset_key": dataset_key,
                        "dataset_role": dataset_role,
                        "csv_path_used": str(getattr(cfg.data, "csv_path", "") or ""),
                        "target_column": cfg.data.target_column or getattr(y, "name", "target") or "target",
                        "problem_type": str(problem_type),
                        "n_rows": int(len(X)) if X is not None else 0,
                        "split": {
                            "n_train": int(len(X_train)) if X_train is not None else 0,
                            "n_test": int(len(X_test)) if X_test is not None else 0,
                            "test_size": float(cfg.data.test_size),
                            "random_seed": int(cfg.data.random_seed),
                        },
                        "feature_types": feature_types_summary,
                        "n_features_transformed": n_features_transformed,
                    }
                    model_task_mgr.connect_configuration(dataset_conf, name="Dataset")

                    preproc_conf: Dict[str, Any] = {
                        "selected_preprocessor": str(preproc_name),
                        "has_preprocessing_contract": bool(has_preproc_contract),
                        "bundle_available": bool(preproc_bundle is not None),
                    }
                    try:
                        if preproc_manifest_src and preproc_manifest_src.exists():
                            preproc_conf["manifest"] = json.loads(preproc_manifest_src.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                    try:
                        if preproc_schema_src and preproc_schema_src.exists():
                            preproc_conf["schema"] = json.loads(preproc_schema_src.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                    try:
                        if preproc_recipe_src and preproc_recipe_src.exists():
                            preproc_conf["recipe"] = json.loads(preproc_recipe_src.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                    try:
                        if preproc_summary_src and preproc_summary_src.exists():
                            preproc_conf["summary_md"] = preproc_summary_src.read_text(encoding="utf-8")
                    except Exception:
                        pass
                    model_task_mgr.connect_configuration(preproc_conf, name="Preprocessing")
            except Exception:
                pass

            try:
                for m_name, m_val in (train_metrics_display or {}).items():
                    if m_val is None:
                        continue
                    model_task_mgr.report_scalar("train", str(m_name), float(m_val), iteration=0)
                for m_name, m_val in (test_metrics_display or {}).items():
                    if m_val is None:
                        continue
                    model_task_mgr.report_scalar("test", str(m_name), float(m_val), iteration=0)
            except Exception:
                pass

            try:
                has_test_metrics = bool(
                    isinstance(test_metrics_std, dict) and any(v is not None for v in test_metrics_std.values())
                )
                preferred_metrics = (test_metrics_std or {}) if has_test_metrics else (train_metrics_std or {})
                train_metrics_only = (train_metrics_std or {}) if has_test_metrics else None
                report_metric_scalars(
                    model_task_mgr,
                    train_metrics=preferred_metrics,
                    test_metrics=train_metrics_only,
                    iteration=0,
                    train_prefix="metric",
                    test_prefix="metric_train",
                )
            except Exception:
                pass

            try:
                if train_seconds is not None:
                    model_task_mgr.report_scalar("time/train_seconds", "value", float(train_seconds), iteration=0)
                if predict_seconds is not None:
                    model_task_mgr.report_scalar("time/predict_seconds", "value", float(predict_seconds), iteration=0)
                if predict_train_seconds is not None:
                    model_task_mgr.report_scalar("time/predict_train_seconds", "value", float(predict_train_seconds), iteration=0)
                if predict_test_seconds is not None:
                    model_task_mgr.report_scalar("time/predict_test_seconds", "value", float(predict_test_seconds), iteration=0)
            except Exception:
                pass

            try:
                n_rows_train = int(len(X_train)) if X_train is not None else 0
                n_rows_test = int(len(X_test)) if X_test is not None else 0
                n_features_raw = int(X_train.shape[1]) if hasattr(X_train, "shape") and len(X_train.shape) > 1 else None
                model_task_mgr.report_scalar("model/num_rows_train", "value", float(n_rows_train), iteration=0)
                model_task_mgr.report_scalar("model/num_rows_test", "value", float(n_rows_test), iteration=0)
                if n_features_raw is not None:
                    model_task_mgr.report_scalar("model/num_features_raw", "value", float(n_features_raw), iteration=0)
                if n_features_transformed is not None:
                    model_task_mgr.report_scalar("model/num_features", "value", float(n_features_transformed), iteration=0)
                if model_size_bytes is not None:
                    model_task_mgr.report_scalar("model/size_bytes", "value", float(model_size_bytes), iteration=0)
            except Exception:
                pass

            try:
                model_id = model_task_mgr.register_output_model(model_path, name=model_label)
            except Exception:
                model_id = None

            try:
                if model_task_mgr.task:
                    set_user_properties(
                        model_task_mgr.task,
                        {
                            "run_id": run_id,
                            "dataset_id": dataset_id_for_load or "",
                            "dataset_key": dataset_key,
                            "model_name": model_label,
                            "preprocessor": preproc_name,
                            "model_input": "raw",
                            "exported_with_preproc_bundle": bool(exported_with_preproc_bundle),
                            "primary_metric": str(primary_metric_model or ""),
                            "model_id": str(model_id or ""),
                        },
                    )
            except Exception:
                pass

            if cfg.visualizations.feature_importance:
                try:
                    try:
                        fitted_preprocessor = pipeline_best.named_steps["preprocessor"]
                        fnames = fitted_preprocessor.get_feature_names_out()
                    except Exception:
                        fitted_preprocessor = pipeline_best.named_steps["preprocessor"]
                        xt = fitted_preprocessor.transform(X_train)
                        fnames = [f"f{i}" for i in range(xt.shape[1])]
                    imp_df = extract_feature_importance(pipeline_best.named_steps["model"], list(fnames))
                    if imp_df is not None and not imp_df.empty:
                        fi_plot_path = visual_fi_dir / f"{safe_name}.png"
                        plot_feature_importance(
                            imp_df,
                            fi_plot_path,
                            title=f"Feature Importance: {model_label} ({preproc_name})",
                        )
                        try:
                            imp_df.to_csv(visual_fi_dir / f"{safe_name}.csv", index=False)
                        except Exception:
                            pass
                        if model_task_mgr.logger:
                            try:
                                import plotly.express as px  # type: ignore

                                top_imp = imp_df.head(30).copy().sort_values(by="importance", ascending=True)
                                fig = px.bar(
                                    top_imp,
                                    x="importance",
                                    y="feature",
                                    orientation="h",
                                    title=f"Feature Importance: {model_label} ({preproc_name})",
                                )
                                model_task_mgr.logger.report_plotly(
                                    title="02_Explain/01_Feature_Importance",
                                    series="feature_importance",
                                    iteration=0,
                                    figure=fig,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

            if cfg.visualizations.shap_summary:
                shap_imp_df = None
                try:
                    shap_imp_df = compute_shap_importance(
                        pipeline_best,
                        X_train,
                        max_display=30,
                        sample_size=200,
                    )
                except Exception:
                    shap_imp_df = None
                if shap_imp_df is not None and not shap_imp_df.empty:
                    try:
                        shap_imp_df.to_csv(visual_shap_dir / f"{safe_name}_importance.csv", index=False)
                    except Exception:
                        pass
                    if model_task_mgr.logger:
                        try:
                            import plotly.express as px  # type: ignore

                            df_bar = shap_imp_df.sort_values(by="shap_importance", ascending=True)
                            fig = px.bar(
                                df_bar,
                                x="shap_importance",
                                y="feature",
                                orientation="h",
                                title=f"SHAP (mean |value|): {model_label} ({preproc_name})",
                            )
                            model_task_mgr.logger.report_plotly(
                                title="02_Explain/02_SHAP_Summary",
                                series="shap_summary",
                                iteration=0,
                                figure=fig,
                            )
                        except Exception:
                            pass

            if model_task_mgr.logger and problem_type == "regression" and cfg.output.generate_plots:
                try:
                    if cfg.visualizations.predicted_vs_actual and y_pred_for_plots is not None:
                        fig = build_plotly_pred_vs_actual(
                            y_train_array,
                            y_pred_for_plots,
                            title=f"{model_label} ({preproc_name})",
                            add_regression_line=True,
                        )
                        if fig is not None:
                            model_task_mgr.logger.report_plotly(
                                title="01_Performance/01_Pred_vs_True",
                                series="performance",
                                iteration=0,
                                figure=fig,
                            )
                    if cfg.visualizations.residual_scatter and y_pred_for_plots is not None and residuals_for_plots is not None:
                        fig = build_plotly_residual_scatter(
                            y_pred_for_plots,
                            residuals_for_plots,
                            title=f"Residuals: {model_label} ({preproc_name})",
                        )
                        if fig is not None:
                            model_task_mgr.logger.report_plotly(
                                title="01_Performance/02_Residuals",
                                series="performance",
                                iteration=0,
                                figure=fig,
                            )
                    if cfg.visualizations.residual_hist and residuals_for_plots is not None:
                        fig = build_plotly_histogram(
                            residuals_for_plots.tolist(),
                            metric_name="residual",
                            title=f"Residual Histogram: {model_label} ({preproc_name})",
                        )
                        if fig is not None:
                            model_task_mgr.logger.report_plotly(
                                title="01_Performance/03_Residual_Hist",
                                series="performance",
                                iteration=0,
                                figure=fig,
                            )
                except Exception:
                    pass

            if model_task_mgr.logger:
                try:
                    try:
                        if Xt_full is not None:
                            fig = build_plotly_interpolation_space(
                                Xt_full,
                                y_train_array,
                                title=f"Interpolation space: {model_label}",
                            )
                            if fig is not None:
                                model_task_mgr.logger.report_plotly(
                                    title="03_Diagnostics/01_InterpolationSpace",
                                    series="diagnostics",
                                    iteration=0,
                                    figure=fig,
                                )
                    except Exception:
                        pass
                except Exception:
                    pass

            related_paths = [
                model_path,
                visual_pred_dir / f"{safe_name}.png",
                visual_pred_dir / f"{safe_name}.csv",
                visual_resid_scatter_dir / f"{safe_name}.png",
                visual_resid_scatter_dir / f"{safe_name}.csv",
                visual_resid_hist_dir / f"{safe_name}.png",
                visual_resid_hist_dir / f"{safe_name}.csv",
                visual_interp_dir / f"{safe_name}.png",
                visual_confusion_dir / f"{safe_name}.png",
                visual_confusion_dir / f"{safe_name}.csv",
                visual_confusion_dir / f"{safe_name}_normalized.png",
                visual_confusion_dir / f"{safe_name}_normalized.csv",
                visual_roc_dir / f"{safe_name}.png",
                visual_roc_dir / f"{safe_name}.csv",
                visual_pr_dir / f"{safe_name}.png",
                visual_pr_dir / f"{safe_name}.csv",
                visual_fi_dir / f"{safe_name}.png",
                visual_fi_dir / f"{safe_name}.csv",
                visual_shap_dir / f"{safe_name}_importance.csv",
            ]
            try:
                model_task_mgr.upload_artifacts([p for p in related_paths if isinstance(p, Path) and p.exists()])
            except Exception:
                pass

            try:
                task_url = ""
                if model_task_mgr.task and hasattr(model_task_mgr.task, "get_output_log_web_page"):
                    task_url = str(model_task_mgr.task.get_output_log_web_page() or "")
                record.update(
                    {
                        "train_seconds": train_seconds,
                        "predict_seconds": predict_seconds,
                        "predict_train_seconds": predict_train_seconds,
                        "predict_test_seconds": predict_test_seconds,
                        "model_size_bytes": model_size_bytes,
                        "num_features": n_features_transformed,
                        "model_id": str(model_id or ""),
                        "model_input": "raw",
                        "exported_with_preproc_bundle": bool(exported_with_preproc_bundle),
                        "task_id": model_task_mgr.task.id if model_task_mgr.task else "",
                        "url": task_url,
                        "link_html": f'<a href="{task_url}">{model_label} ({preproc_name})</a>' if task_url else "",
                        "status": "ok",
                        "error": "",
                    }
                )
                has_test_metrics = bool(
                    isinstance(test_metrics_std, dict) and any(v is not None for v in test_metrics_std.values())
                )
                record["metric_source"] = "test" if has_test_metrics else "train"
                record.update((test_metrics_std or {}) if has_test_metrics else (train_metrics_std or {}))
            except Exception:
                pass

            try:
                if model_task_mgr.task:
                    training_task_ids.append(model_task_mgr.task.id)
            except Exception:
                pass

        except Exception as exc:
            try:
                record["status"] = "failed"
                record["error"] = f"unexpected error: {exc}"
            except Exception:
                record = {"model": model_label, "status": "failed", "error": f"unexpected error: {exc}"}
        finally:
            if model_task_mgr is not None:
                model_task_mgr.close()
            try:
                err = record.get("error")
                if err is not None:
                    err_s = str(err)
                    if len(err_s) > 1000:
                        record["error"] = err_s[:1000] + "..."
            except Exception:
                pass
            if isinstance(record, dict) and record:
                model_task_records.append(record)

        try:
            if safe_name and y_train_array is not None:
                preds_rows = []
                if y_pred_train is not None:
                    for idx, (yt, yp) in enumerate(zip(y_train_array, y_pred_train)):
                        row = {"sample": idx, "split": "train", "y_true": yt, "y_pred": yp}
                        if problem_type == "regression":
                            try:
                                row["residual"] = float(yt - yp)
                            except Exception:
                                pass
                        else:
                            try:
                                row["correct"] = bool(yt == yp)
                            except Exception:
                                pass
                        preds_rows.append(row)
                if y_pred_test is not None:
                    for idx, (yt, yp) in enumerate(zip(y_test, y_pred_test)):
                        row = {"sample": idx, "split": "test", "y_true": yt, "y_pred": yp}
                        if problem_type == "regression":
                            try:
                                row["residual"] = float(yt - yp)
                            except Exception:
                                pass
                        else:
                            try:
                                row["correct"] = bool(yt == yp)
                            except Exception:
                                pass
                        preds_rows.append(row)
                if preds_rows:
                    pd.DataFrame(preds_rows).to_csv(output_dir / f"predictions_{safe_name}.csv", index=False)

                metrics_records = []
                for k, v in (train_metrics_display or {}).items():
                    metrics_records.append({"metric": str(k), "split": "train", "value": v})
                for k, v in (test_metrics_display or {}).items():
                    metrics_records.append({"metric": str(k), "split": "test", "value": v})
                pd.DataFrame(metrics_records).to_csv(output_dir / f"metrics_{safe_name}.csv", index=False)
        except Exception:
            pass

    return model_task_records, training_task_ids
