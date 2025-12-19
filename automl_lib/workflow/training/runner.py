"""Training runner (workflow orchestration).

Entry-point: `run_automl()`.

Responsibilities (kept intentionally small):
- Load/validate config (+ ClearML overrides / HyperParameters edits)
- Resolve dataset sources + preprocessing contract sources
- Evaluate model/preprocessor combinations (CV) and write `results_summary.csv/json`
- Launch per-model training child tasks and aggregate their results
- Produce a clean training-summary dashboard (Plots 01–08 only)
"""

from __future__ import annotations

# ruff: noqa: E402

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from automl_lib.config.loaders import load_training_config
from automl_lib.integrations.clearml.bootstrap import ensure_clearml_config_file
from automl_lib.integrations.clearml.context import (
    build_run_context,
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    run_scoped_output_dir,
    sanitize_name_token,
    set_run_id_env,
)
from automl_lib.integrations.clearml.naming import build_project_path, build_tags, task_name
from automl_lib.integrations.clearml.manager import ClearMLManager, _import_clearml
from automl_lib.registry.metrics import is_loss_metric
from automl_lib.training.data_loader import get_feature_types, infer_problem_type, load_dataset, split_data
from automl_lib.workflow.training.artifacts import collect_training_summary_artifacts, write_model_task_records
from automl_lib.workflow.training.clearml_summary import (
    connect_child_tasks_overview,
    connect_dataset_and_preprocessing_objects,
    connect_training_hyperparams,
    fail_training_summary,
    finalize_task,
    normalize_summary_plots_mode,
    persist_full_config,
    set_min_user_properties,
    set_training_summary_user_properties,
)
from automl_lib.workflow.training.dataset_sources import resolve_training_dataset_sources
from automl_lib.workflow.training.model_tasks import run_model_tasks
from automl_lib.workflow.training.preprocessor_selection import select_preprocessors
from automl_lib.workflow.training.recommendation import build_recommendation_and_leaderboard
from automl_lib.workflow.training.summary_evaluation import (
    build_model_instances,
    evaluate_models_with_ensembles,
    fit_pipeline_for_row,
    pick_best_row_global,
    resolve_primary_metric,
    select_best_rows_per_model,
)
from automl_lib.workflow.training.summary_plots import log_leaderboard_bar, log_training_summary_dashboard

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if __package__ in {None, ""}:
    root_str = str(PROJECT_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)
tabpfn_home = PROJECT_ROOT / ".tabpfn_home"
tabpfn_home.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TABPFN_HOME", str(tabpfn_home))
os.environ.setdefault("TABPFN_STATE_DIR", str(tabpfn_home))
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(tabpfn_home / "model_cache"))

ensure_clearml_config_file()

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message="The optimal value found for dimension 0 of parameter length_scale is close to the specified lower bound 1e-05",
)
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message="Stochastic Optimizer: Maximum iterations",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Could not find the number of physical cores",
)


def run_automl(
    config_path: Path,
    *,
    dataset_id: Optional[str] = None,
    parent_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run training (summary task + per-model child tasks)."""

    config_path = Path(config_path)
    try:
        cfg = load_training_config(config_path)
    except Exception as exc:
        raise ValueError(f"Invalid training config: {exc}") from exc

    # ClearML clone-mode overrides + HyperParameters edits (clone -> edit -> run)
    # should override YAML config before any processing starts.
    try:
        cfg_dump = cfg.model_dump()
        try:
            from automl_lib.integrations.clearml.overrides import apply_overrides, get_task_overrides

            overrides = get_task_overrides()
            if overrides:
                cfg_dump = apply_overrides(cfg_dump, overrides)
        except Exception:
            pass
        try:
            if cfg.clearml and bool(getattr(cfg.clearml, "enabled", False)):
                from automl_lib.integrations.clearml.hyperparams import (
                    apply_training_hyperparams,
                    get_current_task_hyperparams,
                )

                params = get_current_task_hyperparams(cast=True)
                if isinstance(params, dict):
                    cfg_dump = apply_training_hyperparams(cfg_dump, params)
        except Exception:
            pass
        if cfg_dump != cfg.model_dump():
            cfg = type(cfg).model_validate(cfg_dump)
    except Exception as exc:
        raise ValueError(f"Invalid ClearML HyperParameters override for training: {exc}") from exc

    run_id = resolve_run_id(from_config=getattr(cfg.run, "id", None), from_env=get_run_id_env())
    set_run_id_env(run_id)
    base_output_dir = Path(cfg.output.output_dir)
    output_dir = run_scoped_output_dir(base_output_dir, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = resolve_training_dataset_sources(
        cfg=cfg,
        dataset_id_override=(str(dataset_id) if dataset_id else None),
        base_output_dir=base_output_dir,
    )
    dataset_id_for_load = sources.dataset_id_for_load
    if sources.csv_override:
        cfg.data.csv_path = str(sources.csv_override)

    preproc_summary_src = sources.preproc_summary_src
    preproc_recipe_src = sources.preproc_recipe_src
    preproc_schema_src = sources.preproc_schema_src
    preproc_manifest_src = sources.preproc_manifest_src
    preproc_bundle_src = sources.preproc_bundle_src
    preproc_bundle = sources.preproc_bundle
    has_preproc_contract = sources.has_preproc_contract

    dataset_key = resolve_dataset_key(
        explicit=getattr(cfg.run, "dataset_key", None),
        dataset_id=str(dataset_id_for_load) if dataset_id_for_load else None,
        csv_path=getattr(cfg.data, "csv_path", None),
    )
    ctx = build_run_context(
        run_id=run_id,
        dataset_key=dataset_key,
        project_root=(cfg.clearml.project_name if cfg.clearml else None),
        dataset_project=(cfg.clearml.dataset_project if cfg.clearml else None),
        user=getattr(cfg.run, "user", None),
    )

    naming_cfg = getattr(cfg.clearml, "naming", None) if cfg.clearml else None
    project_mode = getattr(naming_cfg, "project_mode", "root")
    train_suffix = getattr(naming_cfg, "train_models_suffix", "train_models")
    project_for_summary = build_project_path(ctx, project_mode=project_mode)
    train_models_project = build_project_path(ctx, project_mode=project_mode, suffix=train_suffix)

    base_summary_name = cfg.clearml.task_name if cfg.clearml and cfg.clearml.task_name else None
    summary_task_name = (
        f"{sanitize_name_token(base_summary_name, max_len=64)} ds:{sanitize_name_token(ctx.dataset_key, max_len=64)} "
        f"run:{sanitize_name_token(ctx.run_id, max_len=64)}"
        if base_summary_name
        else task_name("training_summary", ctx)
    )

    summary_task_obj = None
    if cfg.clearml and cfg.clearml.enabled and os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        # In PipelineController function steps, a step task already exists (Task.current_task()).
        # Reuse it as the user-facing training-summary task to avoid duplication/confusion.
        _, _, TaskCls, _ = _import_clearml()
        if TaskCls is not None:
            try:
                summary_task_obj = TaskCls.current_task()
            except Exception:
                summary_task_obj = None
            if summary_task_obj is not None:
                try:
                    if hasattr(summary_task_obj, "move_to_project"):
                        summary_task_obj.move_to_project(new_project_name=project_for_summary)
                except Exception:
                    pass
                try:
                    if hasattr(summary_task_obj, "set_name"):
                        summary_task_obj.set_name(str(summary_task_name))
                    elif hasattr(summary_task_obj, "rename"):
                        summary_task_obj.rename(str(summary_task_name))
                except Exception:
                    pass

    clearml_mgr = ClearMLManager(
        cfg.clearml,
        task_name=summary_task_name,
        task_type="training",
        default_project=project_for_summary,
        project=project_for_summary,
        parent=parent_task_id,
        existing_task=summary_task_obj,
        extra_tags=build_tags(ctx, phase="training"),
    )

    # Ensure the task is searchable even if we fail early.
    set_min_user_properties(
        clearml_mgr,
        run_id=run_id,
        dataset_id=dataset_id_for_load,
        dataset_key=dataset_key,
    )

    summary_plots_mode = normalize_summary_plots_mode(cfg)

    full_config_path = persist_full_config(
        clearml_mgr,
        cfg=cfg,
        run_id=run_id,
        dataset_key=dataset_key,
        dataset_id_for_load=(str(dataset_id_for_load) if dataset_id_for_load else None),
        output_dir=output_dir,
    )
    connect_training_hyperparams(clearml_mgr, cfg=cfg, dataset_id_for_load=(dataset_id_for_load or None))

    preproc_summary_path: Optional[Path] = None
    preproc_recipe_path: Optional[Path] = None
    preproc_schema_path: Optional[Path] = None
    preproc_manifest_path: Optional[Path] = None
    try:
        if preproc_summary_src and preproc_summary_src.exists():
            preproc_summary_path = output_dir / "preprocessing_summary.md"
            preproc_summary_path.write_text(preproc_summary_src.read_text(encoding="utf-8"), encoding="utf-8")
        if preproc_recipe_src and preproc_recipe_src.exists():
            preproc_recipe_path = output_dir / "preprocessing_recipe.json"
            preproc_recipe_path.write_text(preproc_recipe_src.read_text(encoding="utf-8"), encoding="utf-8")
        if preproc_schema_src and preproc_schema_src.exists():
            preproc_schema_path = output_dir / "preprocessing_schema.json"
            preproc_schema_path.write_text(preproc_schema_src.read_text(encoding="utf-8"), encoding="utf-8")
        if preproc_manifest_src and preproc_manifest_src.exists():
            preproc_manifest_path = output_dir / "preprocessing_manifest.json"
            preproc_manifest_path.write_text(preproc_manifest_src.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    try:
        X, y = load_dataset(cfg.data)
        problem_type = infer_problem_type(y, cfg.data.problem_type)
        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            test_size=cfg.data.test_size,
            random_seed=cfg.data.random_seed,
            shuffle=cfg.cross_validation.shuffle,
        )
        feature_types = get_feature_types(X_train)

        preprocessors = select_preprocessors(
            cfg=cfg,
            feature_types=feature_types,
            X_train=X_train,
            has_preproc_contract=has_preproc_contract,
            preproc_manifest_src=preproc_manifest_src,
        )
        if not preprocessors:
            raise RuntimeError("No preprocessing pipelines were generated. Check preprocessing configuration.")

        metrics = (
            cfg.evaluation.regression_metrics
            if problem_type == "regression"
            else cfg.evaluation.classification_metrics
        )
        model_instances = build_model_instances(
            cfg=cfg,
            preprocessors=preprocessors,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            metrics=metrics,
        )
        if not model_instances:
            raise RuntimeError("No valid models were instantiated. Check your configuration.")

        results_df = evaluate_models_with_ensembles(
            cfg=cfg,
            preprocessors=preprocessors,
            model_instances=model_instances,
            X_train=X_train,
            y_train=y_train,
            problem_type=problem_type,
            metrics=metrics,
        )
        primary_metric_model = resolve_primary_metric(cfg=cfg, results_df=results_df, problem_type=problem_type)
        results_df_best = select_best_rows_per_model(
            results_df=results_df,
            primary_metric=primary_metric_model,
            problem_type=problem_type,
        )
        best_row_global = pick_best_row_global(
            results_df_best=results_df_best,
            primary_metric=primary_metric_model,
            problem_type=problem_type,
        )

        results_path = output_dir / cfg.output.results_csv
        results_df_best.to_csv(results_path, index=False)
        results_df_best.to_json(results_path.with_suffix(".json"), orient="records", indent=2)
    except Exception as exc:
        fail_training_summary(
            clearml_mgr,
            output_dir=output_dir,
            run_id=run_id,
            dataset_id=(str(dataset_id_for_load) if dataset_id_for_load else None),
            dataset_key=dataset_key,
            exc=exc,
        )
        raise

    training_task_ids: List[str] = []
    model_task_records: List[Dict[str, Any]] = []
    try:
        model_task_records, training_task_ids = run_model_tasks(
            cfg=cfg,
            ctx=ctx,
            run_id=run_id,
            dataset_id_for_load=dataset_id_for_load,
            dataset_key=dataset_key,
            output_dir=output_dir,
            preprocessors=preprocessors,
            results_df_best=results_df_best,
            X=X,
            y=y,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            problem_type=problem_type,
            metrics=metrics,
            primary_metric_model=primary_metric_model,
            clearml_mgr=clearml_mgr,
            train_models_project=train_models_project,
            feature_types=feature_types,
            has_preproc_contract=has_preproc_contract,
            preproc_bundle=preproc_bundle,
            preproc_schema_src=preproc_schema_src,
            preproc_manifest_src=preproc_manifest_src,
            preproc_summary_src=preproc_summary_src,
            preproc_recipe_src=preproc_recipe_src,
        )
    except Exception as exc:
        fail_training_summary(
            clearml_mgr,
            output_dir=output_dir,
            run_id=run_id,
            dataset_id=(str(dataset_id_for_load) if dataset_id_for_load else None),
            dataset_key=dataset_key,
            exc=exc,
        )
        raise

    try:
        df_links = write_model_task_records(output_dir, model_task_records)
        rec_meta = build_recommendation_and_leaderboard(
            cfg=cfg,
            output_dir=output_dir,
            df_links=df_links,
            primary_metric_model=primary_metric_model,
            problem_type=problem_type,
            run_id=run_id,
            dataset_key=dataset_key,
            summary_best_model_id=None,
        )
        recommended_df = rec_meta["recommended_df"]
        df_links_ranked = rec_meta["df_links_ranked"]
        training_primary_metric = rec_meta["training_primary_metric"]
        recommend_metric = rec_meta["recommend_metric"]
        recommend_goal = rec_meta["recommend_goal"]
    except Exception as exc:
        fail_training_summary(
            clearml_mgr,
            output_dir=output_dir,
            run_id=run_id,
            dataset_id=(str(dataset_id_for_load) if dataset_id_for_load else None),
            dataset_key=dataset_key,
            exc=exc,
        )
        raise

    connect_child_tasks_overview(
        clearml_mgr,
        df_links=(df_links if isinstance(df_links, pd.DataFrame) else None),
        df_links_ranked=(df_links_ranked if isinstance(df_links_ranked, pd.DataFrame) else None),
        recommend_metric=(str(recommend_metric) if recommend_metric is not None else None),
    )

    # Fit pipeline for plots from the *recommended* row when available, otherwise fallback to the global best row.
    full_pipeline = None
    try:
        row_for_plots = None
        if recommended_df is not None and isinstance(recommended_df, pd.DataFrame) and not recommended_df.empty:
            row_for_plots = recommended_df.iloc[0]
        else:
            row_for_plots = best_row_global
        full_pipeline, _, _, _ = fit_pipeline_for_row(
            cfg=cfg,
            preprocessors=preprocessors,
            best_row=row_for_plots,
            problem_type=problem_type,
            X_train=X_train,
            y_train=y_train,
        )
    except Exception:
        full_pipeline = None

    # Local/offline usability: persist a single best pipeline for inference convenience.
    try:
        if cfg.output.save_models and full_pipeline is not None:
            from joblib import dump  # type: ignore

            model_dir = output_dir / "models"
            model_dir.mkdir(exist_ok=True)
            dump(full_pipeline, model_dir / "best_model.joblib")
    except Exception:
        pass

    try:
        # 02 + 01/03/04 (+ 05–08 for recommended model)
        df_for_bar = (
            df_links_ranked
            if (df_links_ranked is not None and isinstance(df_links_ranked, pd.DataFrame) and not df_links_ranked.empty)
            else df_links
        )
        metric_for_bar = recommend_metric or primary_metric_model
        goal_for_bar = recommend_goal
        if not goal_for_bar and metric_for_bar:
            goal_for_bar = "min" if is_loss_metric(str(metric_for_bar), problem_type=problem_type) else "max"
        log_leaderboard_bar(
            clearml_mgr=clearml_mgr,
            results_df_best=(df_for_bar if isinstance(df_for_bar, pd.DataFrame) else results_df_best),
            metric=metric_for_bar,
            goal=str(goal_for_bar or "max"),
        )
        log_training_summary_dashboard(
            clearml_mgr=clearml_mgr,
            cfg=cfg,
            problem_type=problem_type,
            df_links=df_links if isinstance(df_links, pd.DataFrame) else None,
            df_links_ranked=df_links_ranked if isinstance(df_links_ranked, pd.DataFrame) else None,
            recommended_df=recommended_df if isinstance(recommended_df, pd.DataFrame) else None,
            recommend_metric=metric_for_bar,
            summary_plots_mode=summary_plots_mode,
            full_pipeline=full_pipeline,
            X_train=X_train,
            y_train=y_train,
        )
    except Exception:
        pass

    dataset_role = "preprocessed" if has_preproc_contract else "raw"
    target_column = str(cfg.data.target_column or getattr(y, "name", "target") or "target")
    set_training_summary_user_properties(
        clearml_mgr,
        run_id=run_id,
        dataset_id_for_load=(str(dataset_id_for_load) if dataset_id_for_load else None),
        dataset_key=dataset_key,
        dataset_role=dataset_role,
        target_column=target_column,
        primary_metric=str(training_primary_metric or ""),
        preproc_manifest_src=preproc_manifest_src,
        preproc_recipe_src=preproc_recipe_src,
        recommended_df=(recommended_df if isinstance(recommended_df, pd.DataFrame) else None),
        metric_for_bar=(str(metric_for_bar) if metric_for_bar is not None else None),
    )

    model_candidates = [str(s.name) for s in getattr(cfg, "models", []) if getattr(s, "enable", True)]
    connect_dataset_and_preprocessing_objects(
        clearml_mgr,
        dataset_id_for_load=(str(dataset_id_for_load) if dataset_id_for_load else None),
        dataset_key=dataset_key,
        problem_type=str(problem_type),
        X=X,
        X_train=X_train,
        X_test=X_test,
        preprocessors=preprocessors,
        feature_types=(feature_types if isinstance(feature_types, dict) else {}),
        has_preproc_contract=bool(has_preproc_contract),
        preproc_bundle_available=bool(preproc_bundle is not None),
        preproc_manifest_src=preproc_manifest_src,
        preproc_schema_src=preproc_schema_src,
        preproc_recipe_src=preproc_recipe_src,
        preproc_summary_src=preproc_summary_src,
        model_candidates=model_candidates,
    )

    try:
        artifacts_to_upload = collect_training_summary_artifacts(
            output_dir=output_dir,
            results_path=(output_dir / cfg.output.results_csv),
            full_config_path=None if clearml_mgr.task else full_config_path,
            preproc_summary_path=preproc_summary_path,
            preproc_recipe_path=preproc_recipe_path,
            preproc_schema_path=preproc_schema_path,
            preproc_manifest_path=preproc_manifest_path,
        )
        clearml_mgr.upload_artifacts([p for p in artifacts_to_upload if p.exists()])
    except Exception:
        pass
    finally:
        finalize_task(clearml_mgr)

    metrics_for_comparison = model_task_records
    try:
        metrics_for_comparison = [
            r
            for r in model_task_records
            if isinstance(r, dict) and (not str(r.get("status") or "").strip() or str(r.get("status") or "").strip().lower() == "ok")
        ]
    except Exception:
        metrics_for_comparison = model_task_records

    return {
        "dataset_id": dataset_id_for_load,
        "summary_task_id": clearml_mgr.task.id if clearml_mgr.task else None,
        "training_task_ids": training_task_ids,
        "metrics": metrics_for_comparison,
    }
