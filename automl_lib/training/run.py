"""Entry point for running the custom AutoML pipeline.

This script coordinates the end‑to‑end workflow: reading the configuration
file, loading the data, generating preprocessing pipelines and model
instances, evaluating all combinations via cross‑validation, writing
results to disk, and producing visualizations. It can be executed as a
stand‑alone Python module or imported and called from other code.

Usage
-----
From the command line:

```
python -m automl_lib.cli.run_training --config path/to/config.yaml
```

The configuration YAML specifies all behavior including data paths,
preprocessing options, models to evaluate, cross‑validation settings and
output preferences. See ``config.yaml`` for a template.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_MODELS_PROJECT = "train_models"

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

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from automl_lib.clearml.bootstrap import ensure_clearml_config_file
from automl_lib.clearml.context import (
    build_run_context,
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    run_scoped_output_dir,
    sanitize_name_token,
    set_run_id_env,
)
from automl_lib.clearml.naming import build_project_path, build_tags, task_name
from automl_lib.config.loaders import load_training_config
from automl_lib.config.schemas import TrainingConfig
from automl_lib.preprocessing.preprocessors import generate_preprocessors
from automl_lib.registry.metrics import add_derived_metrics, is_loss_metric

from .clearml_integration import ClearMLManager, _import_clearml, ensure_local_dataset_copy, find_first_csv
from .data_loader import get_feature_types, infer_problem_type, load_dataset, split_data
from .ensemble import build_stacking, build_voting
from .evaluation import _get_cv_splitter, _get_scoring, evaluate_model_combinations
from .interpretation import compute_shap_importance, extract_feature_importance, plot_feature_importance, plot_shap_summary
from .model_factory import ModelInstance, prepare_tabpfn_params
from .search import generate_param_combinations
from .tabpfn_utils import OfflineTabPFNRegressor
from .reporting import (
    build_plot_artifacts_table,
    report_metric_scalars,
    save_confusion_matrices,
    save_roc_pr_curves,
)
from .visualization import (
    plot_bar_comparison,
    plot_predicted_vs_actual,
    plot_residual_scatter,
    plot_residual_hist,
    plot_metric_heatmap,
    plot_metric_histogram,
    plot_interpolation_space,
    build_plotly_pred_vs_actual,
    build_plotly_residual_scatter,
    build_plotly_histogram,
    build_plotly_interpolation_space,
)
ensure_clearml_config_file()
try:  # optional clearml plotting helpers
    from clearml import Logger
    from clearml.automation.metrics import Scatter2D  # type: ignore
except Exception:
    Logger = None
    Scatter2D = None


def _tune_lightgbm_params(params: Dict[str, Any], train_size: int, problem_type: str) -> Dict[str, Any]:
    """Ensure LightGBM receives sensible defaults for small datasets."""

    tuned = dict(params)
    tuned.setdefault("force_row_wise", True)
    if problem_type == "regression":
        tuned.setdefault("objective", "regression_l2")
    if "min_child_samples" not in tuned and "min_data_in_leaf" not in tuned:
        if train_size > 0:
            candidate = max(1, train_size // 5)
            tuned["min_child_samples"] = candidate
    return tuned


def _instantiate_estimator(
    model_name: str,
    estimator_cls,
    init_params: Dict[str, Any],
):
    """Instantiate an estimator, handling TabPFN fallbacks when necessary."""

    params_for_init = dict(init_params)
    fallback = params_for_init.pop("use_fallback_tabpfn", False)
    if fallback:
        return OfflineTabPFNRegressor(**params_for_init)
    return estimator_cls(**params_for_init)


def _maybe_wrap_with_target_scaler(
    estimator: Any,
    cfg: TrainingConfig,
    problem_type: str,
):
    """Optionally wrap estimator with target standardization for regression."""

    if problem_type.lower() != "regression":
        return estimator
    if not getattr(cfg.preprocessing, "target_standardize", False):
        return estimator
    if isinstance(estimator, TransformedTargetRegressor):
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator,
        transformer=StandardScaler(),
        check_inverse=False,
    )


def _build_estimator_with_defaults(
    model_name: str,
    estimator_cls,
    init_params: Dict[str, Any] | None,
    problem_type: str,
    cfg: TrainingConfig,
    train_size: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate an estimator while applying evaluation-time defaults."""

    params: Dict[str, Any] = dict(init_params or {})
    name_lower = model_name.lower()
    module_name_lower = estimator_cls.__module__.lower()

    if name_lower in {"gaussianprocess", "gaussianprocessregressor", "gaussianprocessclassifier"}:
        if "kernel" in params and isinstance(params["kernel"], str):
            kernel_str = params["kernel"]
            try:
                from sklearn.gaussian_process import kernels as gpkernels  # type: ignore

                kernel_cls = getattr(gpkernels, kernel_str)
                params["kernel"] = kernel_cls()
            except Exception:
                pass
        if problem_type == "regression":
            if "kernel" not in params:
                from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

                params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3),
                ) + WhiteKernel(
                    noise_level=1.0,
                    noise_level_bounds=(1e-5, 1e5),
                )
            params.setdefault("alpha", 1e-2)
            params.setdefault("normalize_y", True)
            params.setdefault("n_restarts_optimizer", 10)

    if "catboost" in module_name_lower:
        params.setdefault("verbose", 0)
        params.setdefault("random_seed", cfg.data.random_seed)
        params.setdefault("allow_writing_files", False)

    if "lightgbm" in module_name_lower:
        params = _tune_lightgbm_params(params, train_size, problem_type)
        if "verbose" not in params and "verbosity" not in params:
            params["verbose"] = -1
        params.setdefault("random_state", cfg.data.random_seed)

    if "xgboost" in module_name_lower:
        params.setdefault("random_state", cfg.data.random_seed)
        params.setdefault("n_jobs", -1)

    if "pytorch_tabnet" in module_name_lower:
        params.setdefault("device_name", "cpu")
        params.setdefault("verbose", 0)

    if name_lower == "mlp":
        params.setdefault("random_state", cfg.data.random_seed)
        if problem_type == "regression":
            params.setdefault("max_iter", 2000)
            params.setdefault("early_stopping", True)
            params.setdefault("n_iter_no_change", 20)
            params.setdefault("validation_fraction", 0.1)

    if name_lower == "tabpfn":
        tabpfn_params = prepare_tabpfn_params(problem_type, params)
        if tabpfn_params is None:
            raise ValueError("TabPFN weights are unavailable")
        params = tabpfn_params

    if "gaussian_process" in module_name_lower and problem_type == "regression":
        if "kernel" not in params:
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

            params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e3),
            ) + WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1e-5, 1e5),
            )
        params.setdefault("alpha", 1e-2)
        params.setdefault("normalize_y", True)
        params.setdefault("n_restarts_optimizer", 10)

    estimator = _instantiate_estimator(model_name, estimator_cls, params)

    if name_lower in {"gaussianprocess", "gaussianprocessregressor"} and problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            check_inverse=False,
        )

    if name_lower in {"tabnet", "mlp"} and problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            check_inverse=False,
        )

    estimator = _maybe_wrap_with_target_scaler(estimator, cfg, problem_type)

    return estimator, params

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

def run_automl(config_path: Path) -> None:
    """Execute the AutoML pipeline with the given configuration.

    This function orchestrates data loading, preprocessing, hyperparameter
    search, model evaluation, ensemble construction, result visualization,
    and interpretability analysis. It reads the configuration, applies
    various search strategies (grid, random, bayesian), evaluates models
    using cross‑validation, generates plots, selects the best model, and
    optionally computes feature importances and SHAP values.
    """
    try:
        cfg = load_training_config(config_path)
    except Exception as exc:
        raise ValueError(f"Invalid training config: {exc}") from exc
    run_id = resolve_run_id(from_config=getattr(cfg.run, "id", None), from_env=get_run_id_env())
    set_run_id_env(run_id)
    base_output_dir = Path(cfg.output.output_dir)
    training_task_ids: List[str] = []
    raw_dataset_id = cfg.clearml.raw_dataset_id if cfg.clearml else None
    preprocessed_dataset_id = cfg.clearml.preprocessed_dataset_id if cfg.clearml else None
    env_raw = os.environ.get("AUTO_ML_RAW_DATASET_ID")
    env_preproc = os.environ.get("AUTO_ML_PREPROCESSED_DATASET_ID")
    if env_raw and not raw_dataset_id:
        raw_dataset_id = env_raw
    if env_preproc and not preprocessed_dataset_id:
        preprocessed_dataset_id = env_preproc
    dataset_id_for_load = (
        os.environ.get("AUTO_ML_DATASET_ID")
        or preprocessed_dataset_id
        or raw_dataset_id
        or getattr(cfg.data, "dataset_id", None)
    )
    if dataset_id_for_load:
        dataset_id_for_load = _normalize_dataset_id(str(dataset_id_for_load), cfg)
    if dataset_id_for_load:
        local_copy = ensure_local_dataset_copy(dataset_id_for_load, base_output_dir / "clearml_dataset")
        if cfg.clearml and cfg.clearml.enabled and not local_copy:
            raise ValueError(f"Failed to download ClearML Dataset (dataset_id={dataset_id_for_load})")
        csv_override = find_first_csv(local_copy) if local_copy else None
        if csv_override:
            cfg.data.csv_path = str(csv_override)

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
    # In ClearML PipelineController steps, the step task lifecycle is owned by the controller.
    # Create a dedicated training-summary task so users can find it under the normal project,
    # while keeping the step task untouched.
    if cfg.clearml and cfg.clearml.enabled and os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        DatasetCls, OutputModelCls, TaskCls, TaskTypesCls = _import_clearml()
        if TaskCls is not None and TaskTypesCls is not None:
            try:
                summary_task_obj = TaskCls.create(
                    project_name=project_for_summary,
                    task_name=summary_task_name,
                    task_type=getattr(TaskTypesCls, "training", None),
                )
            except Exception:
                summary_task_obj = None
    clearml_mgr = ClearMLManager(
        cfg.clearml,
        task_name=summary_task_name,
        task_type="training",
        default_project=project_for_summary,
        project=project_for_summary,
        existing_task=summary_task_obj,
        extra_tags=build_tags(ctx, phase="training"),
    )
    summary_plots_mode = "best"
    try:
        summary_plots_mode = str(getattr(cfg.clearml, "summary_plots", "best") if cfg.clearml else "none").strip().lower()
    except Exception:
        summary_plots_mode = "best"
    if summary_plots_mode not in {"none", "best", "all"}:
        summary_plots_mode = "best"

    # Load data
    X, y = load_dataset(cfg.data)
    # Decide problem type early for downstream defaults
    problem_type = infer_problem_type(y, cfg.data.problem_type)
    # Split data into training and optional hold‑out test set
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=cfg.data.test_size,
        random_seed=cfg.data.random_seed,
        shuffle=cfg.cross_validation.shuffle,
    )
    # Determine feature types and generate preprocessors
    feature_types = get_feature_types(X_train)
    preprocessors = generate_preprocessors(cfg.preprocessing, feature_types)
    if not preprocessors:
        raise RuntimeError("No preprocessing pipelines were generated. Check preprocessing config.")
    # Connect training context info to ClearML for visibility (single place)
    try:
        config_params: Dict[str, Any] = {
            "dataset": {
                "id": dataset_id_for_load or "",
                "project": cfg.clearml.dataset_project if cfg.clearml else "",
                "name": "",
                "version": "",
                "input_features": cfg.data.feature_columns or list(X.columns),
                "target_columns": [cfg.data.target_column or getattr(y, "name", "target") or "target"],
                "n_train_samples": int(len(X_train)),
                "n_valid_samples": 0,
                "n_test_samples": int(len(X_test)) if X_test is not None else 0,
            },
            "training_common": {
                "cv.n_splits": cfg.cross_validation.n_folds,
                "scoring.metrics": ["R2", "MSE", "RMSE", "MAE"],
                "global_random_seed": cfg.data.random_seed,
                "queue.name": (cfg.clearml.queue if cfg.clearml else None),
                "problem_type": problem_type,
            },
            "preprocessing": {
                "pipeline": [p[0] for p in preprocessors],
            },
            "training_data": {
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)) if X_test is not None else 0,
                "n_features": int(X_train.shape[1]),
                "split_type": "random",
                "random_state": cfg.data.random_seed,
            },
        }
        for spec in cfg.models:
            key = f"models.{spec.name}"
            class_path = ""
            try:
                from .model_factory import _get_model_class
                cls_tmp = _get_model_class(spec.name, problem_type)
                class_path = f"{cls_tmp.__module__}.{cls_tmp.__name__}"
            except Exception:
                pass
            config_params[key] = {
                "enabled": getattr(spec, "enable", True),
                "class": class_path,
                "params": getattr(spec, "params", {}),
        }
        cfg_dump = cfg.model_dump()
        try:
            cfg_dump.setdefault("run", {})
            cfg_dump["run"]["id"] = run_id
            cfg_dump["run"]["dataset_key"] = dataset_key
        except Exception:
            pass
        try:
            cfg_dump.setdefault("data", {})
            if dataset_id_for_load:
                cfg_dump["data"]["dataset_id"] = str(dataset_id_for_load)
            if getattr(cfg.data, "csv_path", None):
                cfg_dump["data"]["csv_path"] = str(cfg.data.csv_path)
        except Exception:
            pass
        sections = {
            "Run": cfg_dump.get("run") or {},
            "Data": cfg_dump.get("data") or {},
            "Preprocessing": cfg_dump.get("preprocessing") or {},
            "ModelCandidates": {"models": cfg_dump.get("models") or [], "ensembles": cfg_dump.get("ensembles") or {}},
            "CrossValidation": cfg_dump.get("cross_validation") or {},
            "Evaluation": cfg_dump.get("evaluation") or {},
            "Optimization": cfg_dump.get("optimization") or {},
            "Interpretation": cfg_dump.get("interpretation") or {},
            "Visualizations": cfg_dump.get("visualizations") or {},
            "Output": cfg_dump.get("output") or {},
            "ClearML": cfg_dump.get("clearml") or {},
        }
        clearml_mgr.connect_params_sections(sections)
        # Extra curated context (kept for backward compatibility / visibility).
        clearml_mgr.connect_params(config_params)
    except Exception:
        pass

    # Log raw dataset snapshot and register dataset if requested
    try:
        df_for_logging = X.copy()
        target_name = cfg.data.target_column or getattr(y, "name", "target") or "target"
        df_for_logging[target_name] = y
        clearml_mgr.log_dataset_overview(df_for_logging, "raw_dataset", source=cfg.data.csv_path)
    except Exception:
        pass
    if cfg.clearml and cfg.clearml.register_raw_dataset and not raw_dataset_id:
        try:
            raw_dataset_id = clearml_mgr.register_dataset_from_path(
                name="raw-dataset",
                path=Path(cfg.data.csv_path),
                dataset_project=cfg.clearml.dataset_project,
                parent_ids=None,
                tags=["raw-csv"],
            )
        except Exception:
            raw_dataset_id = None
    # Preprocessing / model configuration tables
    try:
        preproc_rows = [{"preprocessor": prep[0]} for prep in preprocessors]
        if preproc_rows:
            clearml_mgr.report_table("preprocessing_selected", pd.DataFrame(preproc_rows), series="config")
        model_rows = [{"model": spec.name, "params": getattr(spec, "params", {})} for spec in cfg.models]
        if model_rows:
            clearml_mgr.report_table("models_selected", pd.DataFrame(model_rows), series="config")
    except Exception:
        pass
    # Determine metrics list based on problem type
    metrics = cfg.evaluation.regression_metrics if problem_type == "regression" else cfg.evaluation.classification_metrics
    # Hyperparameter search and model instantiation
    model_instances: List[ModelInstance] = []
    # Local import for model class resolution
    from .model_factory import _get_model_class
    # --- Model instantiation -------------------------------------------------
    # Iterate over each model specification and generate parameter combinations
    # using the configured search strategy. For each combination we parse the
    # parameter values (e.g. convert strings like "(64,)" to tuples) and
    # instantiate the corresponding estimator. Any instantiation errors are
    # gracefully skipped to allow the rest of the pipeline to continue.
    import ast
    for spec in cfg.models:
        # Skip models that are explicitly disabled
        if hasattr(spec, "enable") and not spec.enable:
            continue
        combos = generate_param_combinations(
            spec=spec,
            problem_type=problem_type,
            optimization_config=cfg.optimization,
            preprocessors=preprocessors,
            X=X_train,
            y=y_train,
            cv_config=cfg.cross_validation,
            metrics=metrics,
            target_standardize=cfg.preprocessing.target_standardize if problem_type == "regression" else False,
        )
        for params in combos:
            try:
                cls = _get_model_class(spec.name, problem_type)
            except Exception as exc:
                print(f"Warning: {exc}")
                continue
            # Normalize parameter values
            init_params: Dict[str, Any] = {}
            if params:
                for key, value in params.items():
                    val = value
                    # Attempt to parse string literals (e.g. "(64,)")
                    if isinstance(val, str):
                        try:
                            val = ast.literal_eval(val)
                        except Exception:
                            # Leave as string if parsing fails
                            pass
                    # Convert lists for hidden_layer_sizes into tuples
                    if key.lower() == "hidden_layer_sizes":
                        # If parsed value is a list containing a single element, unwrap it
                        if isinstance(val, list):
                            # Handle lists like [(64,)] or [[64]]
                            if len(val) == 1:
                                inner = val[0]
                                # If the inner element is itself a list or tuple of ints, convert directly
                                if isinstance(inner, (list, tuple)):
                                    val = tuple(inner)
                                else:
                                    # Fallback: treat list as the sequence of layer sizes
                                    val = tuple(val)
                            else:
                                # List of ints -> tuple
                                val = tuple(val)
                        elif isinstance(val, tuple):
                            # Already tuple, keep as is
                            val = val
                        init_params[key] = val
                        continue
                    init_params[key] = val
            try:
                estimator, applied_params = _build_estimator_with_defaults(
                    spec.name,
                    cls,
                    init_params,
                    problem_type,
                    cfg,
                    len(y_train),
                )
            except ValueError as exc:
                print(f"Warning: could not instantiate {spec.name}: {exc}")
                continue
            except Exception as exc:
                print(f"Warning: could not instantiate {spec.name} with {init_params}: {exc}")
                continue
            model_instances.append(ModelInstance(name=spec.name, params=applied_params, estimator=estimator))
    if not model_instances:
        raise RuntimeError("No valid models were instantiated. Check your configuration.")
    # Evaluate all base model combinations
    results_df = evaluate_model_combinations(
        preprocessors,
        model_instances,
        X_train,
        y_train,
        cfg.cross_validation,
        problem_type,
        metrics,
    )
    # Evaluate ensembles if configured
    ensemble_records = []
    # Stacking ensembles
    if cfg.ensembles.stacking.enable:
        base_names = cfg.ensembles.stacking.estimators
        final_name = cfg.ensembles.stacking.final_estimator
        for preproc_name, transformer in preprocessors:
            estimators_list: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls = _get_model_class(bn, problem_type)
                    base_est = cls()
                except Exception:
                    continue
                estimators_list.append((bn, base_est))
            if not estimators_list:
                continue
            # Final estimator
            if final_name:
                try:
                    final_cls = _get_model_class(final_name, problem_type)
                    final_est = final_cls()
                except Exception:
                    final_est = None
            else:
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
                    continue
            stack_pipeline = build_stacking(
                preprocessor=transformer,
                estimators=estimators_list,
                final_estimator=final_est,
                problem_type=problem_type,
            )
            stack_for_eval = stack_pipeline
            if problem_type == "regression":
                stack_for_eval = _maybe_wrap_with_target_scaler(stack_pipeline, cfg, problem_type)
            try:
                from sklearn.model_selection import cross_validate
                cv = _get_cv_splitter(problem_type, len(y_train), cfg.cross_validation, y_train)
                scoring = _get_scoring(problem_type, metrics)
                cv_res = cross_validate(
                    stack_for_eval,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                )
                record = {
                    "preprocessor": preproc_name,
                    "model": f"Stacking({' + '.join(base_names)})",
                    "params": {},
                }
                for metric_name, scores in cv_res.items():
                    if not metric_name.startswith("test_"):
                        continue
                    simple = metric_name.replace("test_", "")
                    mean_score = np.mean(scores)
                    if is_loss_metric(simple, problem_type=problem_type):
                        mean_score = -mean_score
                    record[simple] = mean_score
                add_derived_metrics(record, problem_type=problem_type, requested_metrics=metrics)
                ensemble_records.append(record)
            except Exception as exc:
                print(f"Warning: stacking ensemble failed for {preproc_name}: {exc}")
    # Voting ensembles
    if cfg.ensembles.voting.enable:
        base_names = cfg.ensembles.voting.estimators
        voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
        for preproc_name, transformer in preprocessors:
            est_list: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls = _get_model_class(bn, problem_type)
                    est = cls()
                except Exception:
                    continue
                est_list.append((bn, est))
            if not est_list:
                continue
            vote_pipeline = build_voting(
                preprocessor=transformer,
                estimators=est_list,
                voting=voting_scheme,
                problem_type=problem_type,
            )
            vote_for_eval = vote_pipeline
            if problem_type == "regression":
                vote_for_eval = _maybe_wrap_with_target_scaler(vote_pipeline, cfg, problem_type)
            try:
                from sklearn.model_selection import cross_validate
                cv = _get_cv_splitter(problem_type, len(y_train), cfg.cross_validation, y_train)
                scoring = _get_scoring(problem_type, metrics)
                cv_res = cross_validate(
                    vote_for_eval,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                )
                rec = {
                    "preprocessor": preproc_name,
                    "model": f"Voting({' + '.join(base_names)})",
                    "params": {},
                }
                for metric_name, scores in cv_res.items():
                    if not metric_name.startswith("test_"):
                        continue
                    simple = metric_name.replace("test_", "")
                    mean_score = np.mean(scores)
                    if is_loss_metric(simple, problem_type=problem_type):
                        mean_score = -mean_score
                    rec[simple] = mean_score
                add_derived_metrics(rec, problem_type=problem_type, requested_metrics=metrics)
                ensemble_records.append(rec)
            except Exception as exc:
                print(f"Warning: voting ensemble failed for {preproc_name}: {exc}")
    if ensemble_records:
        results_df = pd.concat([results_df, pd.DataFrame(ensemble_records)], ignore_index=True)
    # Prepare output directory
    output_dir = run_scoped_output_dir(base_output_dir, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save only the best result per model. Determine primary metric to rank within each model
    primary_metric_model = cfg.evaluation.primary_metric or ("r2" if problem_type == "regression" else "accuracy")
    if primary_metric_model not in results_df.columns:
        available_cols = [c for c in results_df.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_model = available_cols[0] if available_cols else None
    # Identify best row for each model (including ensembles) based on primary metric
    best_rows: Dict[str, pd.Series] = {}
    if primary_metric_model:
        goal_primary = "min" if is_loss_metric(primary_metric_model, problem_type=problem_type) else "max"
        for _, row in results_df.iterrows():
            if pd.isna(row.get(primary_metric_model)):
                continue
            model_label = row["model"]
            score = row[primary_metric_model]
            if model_label not in best_rows:
                best_rows[model_label] = row
                continue
            try:
                current = best_rows[model_label][primary_metric_model]
            except Exception:
                current = None
            if pd.isna(current):
                best_rows[model_label] = row
                continue
            better = score < current if goal_primary == "min" else score > current
            if better:
                best_rows[model_label] = row
    else:
        for _, row in results_df.iterrows():
            model_label = row["model"]
            best_rows.setdefault(model_label, row)
    # Construct DataFrame of best results
    if best_rows:
        results_df_best = pd.DataFrame(list(best_rows.values()))
    else:
        results_df_best = results_df.copy()
    # Save filtered results (best per model) to CSV and JSON
    results_path = output_dir / cfg.output.results_csv
    results_df_best.to_csv(results_path, index=False)
    results_df_best.to_json(results_path.with_suffix(".json"), orient="records", indent=2)
    clearml_mgr.report_table("02_leaderboard/cv_best_per_model", results_df_best, series="leaderboard")
    # bar chart for metrics comparison
    try:
        import plotly.express as px  # type: ignore

        metrics_list = ["r2", "mse", "rmse"]
        metric_cols = [m for m in metrics_list if m in results_df_best.columns]
        if metric_cols:
            df_bar = results_df_best[["model"] + metric_cols].melt(id_vars=["model"], var_name="metric", value_name="value")
            fig = px.bar(df_bar, x="model", y="value", color="metric", barmode="group", title="Model metric comparison")
            if clearml_mgr.logger:
                clearml_mgr.logger.report_plotly(title="02_leaderboard/metrics_bar", series="leaderboard", iteration=0, figure=fig)
    except Exception:
        pass
    # Generate visualizations if enabled
    if cfg.output.generate_plots:
        # Determine metric columns for the filtered DataFrame
        metric_columns = [c for c in results_df_best.columns if c not in {"preprocessor", "model", "params", "error"}]
        # Generate bar charts for each metric using top performers
        for metric in metric_columns:
            plot_path = output_dir / f"top_{metric}.png"
            df_metric = results_df_best.dropna(subset=[metric])
            if df_metric.empty:
                continue
            top_n = len(df_metric) if len(df_metric) <= 20 else 20
            df_sorted = df_metric.sort_values(by=metric, ascending=False).head(top_n)
            try:
                plot_bar_comparison(
                    results_df=df_sorted,
                    metric=metric,
                    top_n=top_n,
                    output_path=plot_path,
                    title=f"Top models by {metric}",
                )
            except Exception as e:
                print(f"Warning: failed to create bar plot for {metric}: {e}")
            try:
                df_sorted.to_csv(plot_path.with_suffix(".csv"), index=False)
            except Exception as e:
                print(f"Warning: failed to write CSV for {metric}: {e}")
        # Comparative heatmap for all models and metrics
        if cfg.visualizations.comparative_heatmap:
            try:
                heatmap_path = output_dir / "model_metric_heatmap.png"
                plot_metric_heatmap(
                    results_df=results_df_best,
                    metrics=metric_columns,
                    output_path=heatmap_path,
                    primary_metric=primary_metric_model,
                    title="Model Comparison Heatmap",
                )
            except Exception as e:
                print(f"Warning: failed to create comparative heatmap: {e}")

    # ---------------------------------------------------------------------
    # Optional: embed comparison aggregation into training-summary task
    # ---------------------------------------------------------------------
    embed_comparison = bool(cfg.clearml and getattr(cfg.clearml, "comparison_mode", "disabled") == "embedded")
    if embed_comparison:
        try:
            from automl_lib.phases.comparison.meta import build_comparison_metadata
            from automl_lib.phases.comparison.visualization import (
                render_comparison_visuals,
                render_model_summary_visuals,
                render_win_summary_visuals,
            )

            # Prefer an explicit comparison config (pipeline/controller passes it), then
            # fall back to config_comparison.yaml if present, else derive defaults from the training config.
            cmp_cfg = None
            ranking_cfg = None
            try:
                from automl_lib.config.loaders import load_comparison_config

                cmp_path = os.environ.get("AUTO_ML_COMPARISON_CONFIG_PATH")
                if cmp_path:
                    cmp_cfg = load_comparison_config(Path(str(cmp_path)))
                else:
                    default_cmp = Path("config_comparison.yaml")
                    if default_cmp.exists():
                        cmp_cfg = load_comparison_config(default_cmp)
                    else:
                        cmp_cfg = load_comparison_config(config_path)
                ranking_cfg = getattr(cmp_cfg, "ranking", None)
            except Exception:
                cmp_cfg = None
                ranking_cfg = None

            cmp_clearml_cfg = getattr(cmp_cfg, "clearml", None) if cmp_cfg else None
            if not cmp_clearml_cfg:
                cmp_clearml_cfg = cfg.clearml

            desired_metrics: List[str] = []
            try:
                if ranking_cfg and getattr(ranking_cfg, "metrics", None):
                    desired_metrics = [str(m).strip().lower() for m in (ranking_cfg.metrics or []) if str(m).strip()]
            except Exception:
                desired_metrics = []
            if not desired_metrics:
                try:
                    if cmp_clearml_cfg and getattr(cmp_clearml_cfg, "comparison_metrics", None):
                        desired_metrics = [
                            str(m).strip().lower()
                            for m in (cmp_clearml_cfg.comparison_metrics or [])
                            if str(m).strip()
                        ]
                except Exception:
                    desired_metrics = []
            if not desired_metrics:
                try:
                    desired_metrics = [str(m).strip().lower() for m in (metrics or []) if str(m).strip()]
                except Exception:
                    desired_metrics = []
            if not desired_metrics:
                desired_metrics = ["r2", "rmse", "mae"]

            primary_metric_cmp = None
            try:
                if ranking_cfg and getattr(ranking_cfg, "primary_metric", None):
                    primary_metric_cmp = str(ranking_cfg.primary_metric).strip().lower()
            except Exception:
                primary_metric_cmp = None
            if not primary_metric_cmp and primary_metric_model:
                try:
                    primary_metric_cmp = str(primary_metric_model).strip().lower()
                except Exception:
                    primary_metric_cmp = None
            if (not primary_metric_cmp) and desired_metrics:
                primary_metric_cmp = desired_metrics[0]
            if primary_metric_cmp and primary_metric_cmp not in {"composite_score"} and primary_metric_cmp not in desired_metrics:
                desired_metrics.append(primary_metric_cmp)

            goal_cmp = None
            try:
                if ranking_cfg and getattr(ranking_cfg, "goal", None):
                    goal_cmp = str(ranking_cfg.goal).strip().lower()
            except Exception:
                goal_cmp = None

            top_k_cmp = None
            try:
                if ranking_cfg and getattr(ranking_cfg, "top_k", None) is not None:
                    top_k_cmp = int(ranking_cfg.top_k)
            except Exception:
                top_k_cmp = None
            if top_k_cmp is None:
                top_k_cmp = 50

            composite_cfg = getattr(ranking_cfg, "composite", None) if ranking_cfg else None
            composite_enabled = True
            composite_metrics = None
            composite_weights = None
            composite_require_all = False
            if composite_cfg:
                try:
                    composite_enabled = bool(getattr(composite_cfg, "enabled", True))
                except Exception:
                    composite_enabled = True
                try:
                    composite_require_all = bool(getattr(composite_cfg, "require_all_metrics", False))
                except Exception:
                    composite_require_all = False
                try:
                    composite_metrics = list(getattr(composite_cfg, "metrics", []) or [])
                except Exception:
                    composite_metrics = None
                try:
                    composite_weights = dict(getattr(composite_cfg, "weights", {}) or {})
                except Exception:
                    composite_weights = None
            if composite_metrics:
                for m in composite_metrics:
                    try:
                        cand = str(m).strip().lower()
                    except Exception:
                        continue
                    if cand and cand not in desired_metrics and cand not in {"composite_score"}:
                        desired_metrics.append(cand)

            # Keep output location compatible with the standalone comparison phase.
            try:
                comparison_base_dir = Path(cmp_cfg.output.output_dir) if cmp_cfg else Path("outputs/comparison")
            except Exception:
                comparison_base_dir = Path("outputs/comparison")
            comparison_output_dir = run_scoped_output_dir(comparison_base_dir, run_id)
            try:
                df_cmp = results_df.copy()
                df_cmp = df_cmp.assign(run_id=run_id, dataset_key=dataset_key)
                rows_cmp = df_cmp.to_dict(orient="records")
            except Exception:
                rows_cmp = []

            meta_cmp = build_comparison_metadata(
                rows_cmp,
                output_dir=comparison_output_dir,
                metric_cols=desired_metrics,
                primary_metric=primary_metric_cmp,
                goal=goal_cmp,
                group_col="preprocessor",
                top_k=top_k_cmp,
                composite_enabled=composite_enabled,
                composite_metrics=composite_metrics,
                composite_weights=composite_weights,
                composite_require_all=composite_require_all,
            )

            if clearml_mgr.logger:
                try:
                    topk_df = meta_cmp.get("ranked_topk_df")
                    if isinstance(topk_df, pd.DataFrame) and not topk_df.empty:
                        clearml_mgr.report_table("03_comparison/topk", topk_df, series="comparison")
                except Exception:
                    pass
                try:
                    best_by_model_df = meta_cmp.get("best_by_model_df")
                    if isinstance(best_by_model_df, pd.DataFrame) and not best_by_model_df.empty:
                        clearml_mgr.report_table("03_comparison/best_by_model", best_by_model_df, series="comparison")
                except Exception:
                    pass
                try:
                    win_summary_df = meta_cmp.get("win_summary_df")
                    if isinstance(win_summary_df, pd.DataFrame) and not win_summary_df.empty:
                        clearml_mgr.report_table("03_comparison/win_summary", win_summary_df, series="comparison")
                        render_win_summary_visuals(clearml_mgr.logger, win_summary_df, title_prefix="03_comparison")
                except Exception:
                    pass
                try:
                    model_summary_df = meta_cmp.get("model_summary_df")
                    if (
                        isinstance(model_summary_df, pd.DataFrame)
                        and not model_summary_df.empty
                        and primary_metric_model
                    ):
                        render_model_summary_visuals(
                            clearml_mgr.logger,
                            model_summary_df,
                            primary_metric=str(primary_metric_model).strip().lower(),
                            title_prefix="03_comparison",
                        )
                except Exception:
                    pass
                try:
                    best_by_group_model_df = meta_cmp.get("best_by_group_model_df")
                    ranked_df = meta_cmp.get("ranked_df")
                    df_vis = None
                    if isinstance(best_by_group_model_df, pd.DataFrame) and not best_by_group_model_df.empty:
                        df_vis = best_by_group_model_df
                    elif isinstance(ranked_df, pd.DataFrame) and not ranked_df.empty:
                        df_vis = ranked_df
                    if isinstance(df_vis, pd.DataFrame) and not df_vis.empty:
                        render_comparison_visuals(
                            clearml_mgr.logger,
                            df_vis,
                            metric_cols=desired_metrics,
                            title_prefix="03_comparison",
                        )
                except Exception:
                    pass

            try:
                if clearml_mgr.task and isinstance(meta_cmp.get("artifacts"), list):
                    clearml_mgr.upload_artifacts(
                        [Path(p) for p in meta_cmp["artifacts"] if isinstance(p, str) and str(p).strip()]
                    )
            except Exception:
                pass
        except Exception:
            pass

    # Choose the overall best model across models using the filtered results
    primary_metric_global = primary_metric_model
    if not primary_metric_global or primary_metric_global not in results_df_best.columns:
        cols_available = [c for c in results_df_best.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_global = cols_available[0] if cols_available else None
    if primary_metric_global:
        goal_global = "min" if is_loss_metric(primary_metric_global, problem_type=problem_type) else "max"
        best_row = (
            results_df_best.loc[results_df_best[primary_metric_global].idxmin()]
            if goal_global == "min"
            else results_df_best.loc[results_df_best[primary_metric_global].idxmax()]
        )
    else:
        best_row = results_df_best.iloc[0]
    # Retrieve preprocessor and estimator for best model (if ensemble, skip saving)
    best_preproc_name = best_row["preprocessor"]
    best_model_name = best_row["model"]
    best_params = best_row["params"] if isinstance(best_row.get("params"), dict) else {}
    # Find transformer
    transformer = None
    for name, ct in preprocessors:
        if name == best_preproc_name:
            transformer = ct
            break
    if transformer is None:
        transformer = preprocessors[0][1]
    # Construct estimator for the best model
    if best_model_name.startswith("Stacking") or best_model_name.startswith("Voting"):
        # Build ensemble pipeline fresh
        if best_model_name.startswith("Stacking"):
            base_names = cfg.ensembles.stacking.estimators
            final_name = cfg.ensembles.stacking.final_estimator
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            # Determine final estimator
            if final_name:
                try:
                    final_cls = _get_model_class(final_name, problem_type)
                    final_est = final_cls()
                except Exception:
                    final_est = None
            else:
                final_est = None
            from sklearn.linear_model import LinearRegression, LogisticRegression
            if final_est is None:
                final_est = LinearRegression() if problem_type == "regression" else LogisticRegression(max_iter=1000)
            estimator = build_stacking(transformer, ests, final_est, problem_type).named_steps["model"]
            if problem_type == "regression":
                estimator = _maybe_wrap_with_target_scaler(estimator, cfg, problem_type)
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
            estimator = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
            if problem_type == "regression":
                estimator = _maybe_wrap_with_target_scaler(estimator, cfg, problem_type)
    else:
        # Base model: instantiate from class and params
        try:
            cls_best = _get_model_class(best_model_name, problem_type)
        except Exception:
            estimator = None
        else:
            try:
                estimator, _ = _build_estimator_with_defaults(
                    best_model_name,
                    cls_best,
                    best_params,
                    problem_type,
                    cfg,
                    len(y_train),
                )
            except Exception:
                estimator = None
    # Fit pipeline on full training data
    from sklearn.pipeline import Pipeline as SKPipeline
    full_pipeline = SKPipeline([
        ("preprocessor", transformer),
        ("model", estimator),
    ])
    full_pipeline.fit(X_train, y_train)
    # Register preprocessed dataset derived from raw dataset
    if cfg.clearml and cfg.clearml.enable_preprocessing and not preprocessed_dataset_id:
        try:
            transformed = transformer.transform(X_train)
            try:
                feature_names = transformer.get_feature_names_out()
            except Exception:
                feature_names = [f"f{i}" for i in range(transformed.shape[1])]
            try:
                import scipy.sparse as sp  # type: ignore

                if sp.issparse(transformed):  # type: ignore[attr-defined]
                    transformed = transformed.toarray()
            except Exception:
                pass
            df_preprocessed = pd.DataFrame(transformed, columns=feature_names)
            df_preprocessed["target"] = y_train.reset_index(drop=True)
            preprocessed_dataset_id = clearml_mgr.register_dataframe_dataset(
                name="preprocessed-dataset",
                df=df_preprocessed,
                output_dir=output_dir,
                filename="preprocessed_features.csv",
                dataset_project=cfg.clearml.dataset_project if cfg.clearml else None,
                parent_ids=[raw_dataset_id] if raw_dataset_id else None,
                tags=["preprocessed"],
            )
        except Exception:
            preprocessed_dataset_id = None
    # Save best model
    if cfg.output.save_models:
        model_dir = output_dir / "models"
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / "best_model.joblib"
        dump(full_pipeline, model_file)
        clearml_mgr.register_output_model(model_file, name=best_model_name)
    # Predicted vs actual for regression
    if cfg.output.generate_plots and problem_type == "regression" and cfg.data.test_size and len(X_test) > 0:
        pred_plot = output_dir / "predicted_vs_actual.png"
        try:
            plot_predicted_vs_actual(full_pipeline, X_test, y_test, pred_plot)
        except Exception as e:
            print(f"Warning: could not generate predicted vs actual plot: {e}")
    # Feature importance
    if cfg.interpretation.compute_feature_importance:
        try:
            fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
            fnames = fitted_preprocessor.get_feature_names_out()
        except Exception:
            fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
            transformed = fitted_preprocessor.transform(X_train)
            fnames = [f"f{i}" for i in range(transformed.shape[1])]
        imp_df = extract_feature_importance(full_pipeline.named_steps["model"], list(fnames))
        if imp_df is not None:
            fi_path = output_dir / "feature_importance.png"
            try:
                plot_feature_importance(imp_df, fi_path, title="Feature Importance")
            except Exception as e:
                print(f"Warning: could not plot feature importance: {e}")
    # SHAP analysis
    if cfg.interpretation.compute_shap:
        shap_path = output_dir / "shap_summary.png"
        try:
            plot_shap_summary(full_pipeline, X_train, shap_path)
        except Exception as e:
            print(f"Warning: could not generate SHAP summary: {e}")

    # ---------------------------------------------------------------------
    # Fit and save the best pipeline for each model
    # ---------------------------------------------------------------------
    # Build dictionary of best rows from results_df_best keyed by model name
    best_rows: Dict[str, pd.Series] = {
        row["model"]: row for _, row in results_df_best.iterrows()
    }
    # Create directories for per‑model visualizations
    models_all_dir = output_dir / "models"
    models_all_dir.mkdir(exist_ok=True)
    visual_pred_dir = output_dir / "scatter_plots"
    visual_resid_scatter_dir = output_dir / "residual_scatter"
    visual_resid_hist_dir = output_dir / "residual_hist"
    visual_metric_hist_dir = output_dir / "metric_hists"
    visual_interp_dir = output_dir / "interpolation_space"
    visual_fi_dir = output_dir / "feature_importances_models"
    visual_shap_dir = output_dir / "shap_summaries_models"
    visual_confusion_dir = output_dir / "confusion_matrices"
    visual_roc_dir = output_dir / "roc_curves"
    visual_pr_dir = output_dir / "pr_curves"
    # Create only if enabled and regression
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.predicted_vs_actual:
        visual_pred_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_scatter:
        visual_resid_scatter_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_hist:
        visual_resid_hist_dir.mkdir(exist_ok=True)
    visual_metric_hist_dir.mkdir(exist_ok=True)
    visual_interp_dir.mkdir(exist_ok=True)
    if problem_type == "classification" and cfg.output.generate_plots:
        visual_confusion_dir.mkdir(exist_ok=True)
        visual_roc_dir.mkdir(exist_ok=True)
        visual_pr_dir.mkdir(exist_ok=True)
    if cfg.visualizations.feature_importance:
        visual_fi_dir.mkdir(exist_ok=True)
    if cfg.visualizations.shap_summary:
        visual_shap_dir.mkdir(exist_ok=True)
    model_task_records: List[Dict[str, Any]] = []
    # Iterate over best models (per model training tasks)
    for model_label, row in best_rows.items():
        model_task_mgr = None
        record: Dict[str, Any] = {}
        try:
            preproc_name = row["preprocessor"]
            params = row["params"] if isinstance(row.get("params"), dict) else {}
            safe_name = (
                f"{model_label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_"
                f"{str(preproc_name).replace('|', '_')}"
            )
            record = {
                "model": model_label,
                "preprocessor": preproc_name,
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
            # Lookup transformer
            transformer = None
            for name, ct in preprocessors:
                if name == preproc_name:
                    transformer = ct
                    break
            if transformer is None:
                transformer = preprocessors[0][1]
            # Instantiate estimator
            estimator_obj = None
            if model_label.startswith("Stacking") or model_label.startswith("Voting"):
                # Ensemble
                if model_label.startswith("Stacking"):
                    base_names = cfg.ensembles.stacking.estimators
                    final_name = cfg.ensembles.stacking.final_estimator
                    ests: List[Tuple[str, object]] = []
                    for bn in base_names:
                        try:
                            cls_bn = _get_model_class(bn, problem_type)
                            ests.append((bn, cls_bn()))
                        except Exception:
                            pass
                    # Determine final estimator
                    if final_name:
                        try:
                            final_cls = _get_model_class(final_name, problem_type)
                            final_est = final_cls()
                        except Exception:
                            final_est = None
                    else:
                        final_est = None
                    from sklearn.linear_model import LinearRegression, LogisticRegression
                    if final_est is None:
                        final_est = LinearRegression() if problem_type == "regression" else LogisticRegression(max_iter=1000)
                    try:
                        estimator_obj = build_stacking(transformer, ests, final_est, problem_type).named_steps["model"]
                    except Exception as exc:
                        record["status"] = "failed"
                        record["error"] = f"build_stacking failed: {exc}"
                        continue
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
                    try:
                        estimator_obj = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
                    except Exception as exc:
                        record["status"] = "failed"
                        record["error"] = f"build_voting failed: {exc}"
                        continue
            else:
                # Base model
                try:
                    cls = _get_model_class(model_label, problem_type)
                except Exception as exc:
                    record["status"] = "skipped"
                    record["error"] = f"_get_model_class failed: {exc}"
                    continue
                try:
                    estimator_obj, _ = _build_estimator_with_defaults(
                        model_label,
                        cls,
                        params,
                        problem_type,
                        cfg,
                        len(y_train),
                    )
                except Exception as exc:
                    record["status"] = "skipped"
                    record["error"] = f"_build_estimator_with_defaults failed: {exc}"
                    continue
            if problem_type == "regression":
                try:
                    estimator_obj = _maybe_wrap_with_target_scaler(estimator_obj, cfg, problem_type)
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = f"_maybe_wrap_with_target_scaler failed: {exc}"
                    continue
            # Build and fit pipeline
            from sklearn.pipeline import Pipeline as SKPipeline
            pipeline_best = SKPipeline([
                ("preprocessor", transformer),
                ("model", estimator_obj),
            ])
            train_seconds = None
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
            need_plot_data = (
                problem_type == "regression"
                and cfg.output.generate_plots
                and (
                    cfg.visualizations.predicted_vs_actual
                    or cfg.visualizations.residual_scatter
                    or cfg.visualizations.residual_hist
                )
            )
            y_train_array = np.asarray(y_train)
            y_pred_train = None
            train_metrics_display: Dict[str, Optional[float]] = {}
            train_metrics_std: Dict[str, Optional[float]] = {}
            y_pred_for_plots: Optional[np.ndarray] = None
            residuals_for_plots: Optional[np.ndarray] = None
            r2_for_plots: Optional[float] = None
            predict_seconds = None
            predict_train_seconds = None
            predict_test_seconds = None
            # Compute train predictions/metrics
            try:
                t1 = time.perf_counter()
                y_pred_train = np.asarray(pipeline_best.predict(X_train))
                predict_train_seconds = float(time.perf_counter() - t1)
                predict_seconds = predict_train_seconds
            except Exception:
                y_pred_train = None

            metrics_requested = [str(m).strip().lower() for m in (metrics or []) if str(m).strip()]

            mse_val = None
            rmse_val = None
            mae_val = None
            n_classes = None
            scores_for_plots_train = None
            scores_for_plots_test = None

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
                    if y_pred_for_plots is not None:
                        try:
                            resid_tmp = (
                                residuals_for_plots
                                if residuals_for_plots is not None
                                else (y_train_array - y_pred_for_plots)
                            )
                            mse_val = float(np.mean(resid_tmp ** 2))
                            rmse_val = float(np.sqrt(mse_val))
                            mae_val = float(mean_absolute_error(y_train_array, y_pred_for_plots))
                        except Exception:
                            pass

                    train_metrics_display = {
                        "R2": r2_for_plots,
                        "MSE": mse_val,
                        "RMSE": rmse_val,
                        "MAE": mae_val,
                    }
                    train_metrics_std = {
                        "r2": r2_for_plots,
                        "mse": mse_val,
                        "rmse": rmse_val,
                        "mae": mae_val,
                    }
                else:
                    y_true_train = y_train_array
                    y_pred_labels_train = y_pred_train
                    try:
                        n_classes = int(len(np.unique(y_true_train)))
                    except Exception:
                        n_classes = None

                    def _score_values(X_part):
                        try:
                            if hasattr(pipeline_best, "predict_proba"):
                                return pipeline_best.predict_proba(X_part)
                        except Exception:
                            pass
                        try:
                            if hasattr(pipeline_best, "decision_function"):
                                return pipeline_best.decision_function(X_part)
                        except Exception:
                            pass
                        return None

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
                            scores = _score_values(X_train)
                            if scores is not None:
                                scores_for_plots_train = scores
                                model_step = None
                                try:
                                    model_step = pipeline_best.named_steps.get("model")
                                except Exception:
                                    model_step = None
                                classes = getattr(model_step, "classes_", None)
                                if classes is not None:
                                    classes = list(classes)
                                scores_arr = np.asarray(scores)
                                roc_val = None
                                if scores_arr.ndim == 2:
                                    if scores_arr.shape[1] == 2 and classes and len(classes) >= 2:
                                        y_bin = (np.asarray(y_true_train) == classes[1]).astype(int)
                                        roc_val = float(roc_auc_score(y_bin, scores_arr[:, 1]))
                                    elif classes and len(classes) == scores_arr.shape[1]:
                                        mapping = {c: i for i, c in enumerate(classes)}
                                        y_enc = np.asarray([mapping.get(v, -1) for v in np.asarray(y_true_train)], dtype=int)
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
                                    y_bin = (np.asarray(y_true_train) == classes[1]).astype(int)
                                    roc_val = float(roc_auc_score(y_bin, scores_arr))
                                if roc_val is not None:
                                    train_metrics_display["roc_auc_ovr"] = roc_val
                                    train_metrics_std["roc_auc_ovr"] = roc_val
                        except Exception:
                            pass
            # Optionally compute test predictions/metrics
            test_metrics: Dict[str, Optional[float]] = {}
            test_metrics_std: Dict[str, Optional[float]] = {}
            y_pred_test = None
            if X_test is not None and len(X_test) > 0:
                try:
                    t2 = time.perf_counter()
                    y_pred_test = np.asarray(pipeline_best.predict(X_test))
                    predict_test_seconds = float(time.perf_counter() - t2)
                    if problem_type == "regression":
                        mse_test = float(mean_squared_error(y_test, y_pred_test))
                        test_metrics = {
                            "R2": float(r2_score(y_test, y_pred_test)),
                            "MSE": mse_test,
                            "RMSE": float(np.sqrt(mse_test)),
                            "MAE": float(mean_absolute_error(y_test, y_pred_test)),
                        }
                        test_metrics_std = {
                            "r2": test_metrics.get("R2"),
                            "mse": test_metrics.get("MSE"),
                            "rmse": test_metrics.get("RMSE"),
                            "mae": test_metrics.get("MAE"),
                        }
                    else:
                        y_true_test = np.asarray(y_test)
                        y_pred_labels_test = y_pred_test
                        if "accuracy" in metrics_requested:
                            try:
                                val = float(accuracy_score(y_true_test, y_pred_labels_test))
                                test_metrics["accuracy"] = val
                                test_metrics_std["accuracy"] = val
                            except Exception:
                                pass
                        if "precision_macro" in metrics_requested:
                            try:
                                val = float(precision_score(y_true_test, y_pred_labels_test, average="macro", zero_division=0))
                                test_metrics["precision_macro"] = val
                                test_metrics_std["precision_macro"] = val
                            except Exception:
                                pass
                        if "recall_macro" in metrics_requested:
                            try:
                                val = float(recall_score(y_true_test, y_pred_labels_test, average="macro", zero_division=0))
                                test_metrics["recall_macro"] = val
                                test_metrics_std["recall_macro"] = val
                            except Exception:
                                pass
                        if "f1_macro" in metrics_requested:
                            try:
                                val = float(f1_score(y_true_test, y_pred_labels_test, average="macro", zero_division=0))
                                test_metrics["f1_macro"] = val
                                test_metrics_std["f1_macro"] = val
                            except Exception:
                                pass
                        if "roc_auc_ovr" in metrics_requested or "roc_auc" in metrics_requested:
                            try:
                                scores = None
                                try:
                                    if hasattr(pipeline_best, "predict_proba"):
                                        scores = pipeline_best.predict_proba(X_test)
                                except Exception:
                                    scores = None
                                if scores is None:
                                    try:
                                        if hasattr(pipeline_best, "decision_function"):
                                            scores = pipeline_best.decision_function(X_test)
                                    except Exception:
                                        scores = None
                                if scores is not None:
                                    scores_for_plots_test = scores
                                    model_step = None
                                    try:
                                        model_step = pipeline_best.named_steps.get("model")
                                    except Exception:
                                        model_step = None
                                    classes = getattr(model_step, "classes_", None)
                                    if classes is not None:
                                        classes = list(classes)
                                    scores_arr = np.asarray(scores)
                                    roc_val = None
                                    if scores_arr.ndim == 2:
                                        if scores_arr.shape[1] == 2 and classes and len(classes) >= 2:
                                            y_bin = (np.asarray(y_true_test) == classes[1]).astype(int)
                                            roc_val = float(roc_auc_score(y_bin, scores_arr[:, 1]))
                                        elif classes and len(classes) == scores_arr.shape[1]:
                                            mapping = {c: i for i, c in enumerate(classes)}
                                            y_enc = np.asarray([mapping.get(v, -1) for v in np.asarray(y_true_test)], dtype=int)
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
                                        y_bin = (np.asarray(y_true_test) == classes[1]).astype(int)
                                        roc_val = float(roc_auc_score(y_bin, scores_arr))
                                    if roc_val is not None:
                                        test_metrics["roc_auc_ovr"] = roc_val
                                        test_metrics_std["roc_auc_ovr"] = roc_val
                            except Exception:
                                pass
                except Exception:
                    test_metrics = {}
                    test_metrics_std = {}
            # Classification diagnostics: confusion matrix / ROC / PR (saved as PNG/CSV, best-effort).
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

                    # ROC / PR curves (only when score metric is requested; best-effort).
                    try:
                        if "roc_auc_ovr" in metrics_requested or "roc_auc" in metrics_requested:
                            use_test = bool(y_pred_test is not None and X_test is not None)
                            X_curve = X_test if use_test else X_train
                            scores_curve = scores_for_plots_test if use_test else scores_for_plots_train
                            if scores_curve is None:
                                try:
                                    if hasattr(pipeline_best, "predict_proba"):
                                        scores_curve = pipeline_best.predict_proba(X_curve)
                                except Exception:
                                    scores_curve = None
                                if scores_curve is None:
                                    try:
                                        if hasattr(pipeline_best, "decision_function"):
                                            scores_curve = pipeline_best.decision_function(X_curve)
                                    except Exception:
                                        scores_curve = None
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
                except Exception:
                    pass

            # Interpolation / feature space plot (PCA on transformed features)
            interp_path = visual_interp_dir / f"{safe_name}.png"
            proj_csv_path = visual_interp_dir / "feature_space_projection.csv"
            n_features_transformed = None
            try:
                Xt_full = pipeline_best.named_steps["preprocessor"].transform(X_train)
                try:
                    if hasattr(Xt_full, "shape") and len(Xt_full.shape) > 1:
                        n_features_transformed = int(Xt_full.shape[1])
                except Exception:
                    n_features_transformed = None
                plot_interpolation_space(Xt_full, y_train_array, interp_path, title=f"Interpolation space: {model_label}")
                # store projection CSV (first two components)
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
                except Exception:
                    pass
            except Exception:
                pass
            # Save pipeline
            model_path = models_all_dir / f"{safe_name}.joblib"
            try:
                dump(pipeline_best, model_path)
            except Exception:
                pass
            model_size_bytes = None
            try:
                if model_path.exists():
                    model_size_bytes = int(model_path.stat().st_size)
            except Exception:
                model_size_bytes = None
            # ClearML per-model training task
            model_task_name = task_name("training_child", ctx, model=model_label, preproc=preproc_name)
            child_tags = build_tags(ctx, phase="training", model=model_label, preproc=preproc_name, extra=[problem_type])
            task_obj = None
            if cfg.clearml and cfg.clearml.enabled:
                DatasetCls, OutputModelCls, TaskCls, TaskTypesCls = _import_clearml()
                if TaskCls is not None and TaskTypesCls is not None:
                    try:
                        task_obj = TaskCls.create(
                            project_name=train_models_project,
                            task_name=model_task_name,
                            task_type=getattr(TaskTypesCls, "training", None),
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
            model_task_mgr = ClearMLManager(
                cfg.clearml,
                task_name=model_task_name,
                task_type="training",
                default_project=train_models_project,
                project=train_models_project,
                parent=clearml_mgr.task.id if clearml_mgr.task else None,
                existing_task=task_obj,
                extra_tags=child_tags,
            )
            # Prefer connect_params; connect_configuration kept minimal
            try:
                feat_cols_model = cfg.data.feature_columns or (
                    list(X_train.columns) if hasattr(X_train, "columns") else list(range(X_train.shape[1]))
                )
                params_connect = {
                    "dataset": {
                        "id": dataset_id_for_load or "",
                        "name": "",
                        "version": "",
                        "input_features": feat_cols_model,
                        "target_columns": [cfg.data.target_column or getattr(y, "name", "target") or "target"],
                        "n_train_samples": int(len(X_train)),
                        "n_valid_samples": 0,
                        "n_test_samples": int(len(X_test)) if X_test is not None else 0,
                    },
                    "split": {
                        "type": "random",
                        "ratios.train": 1.0 - cfg.data.test_size,
                        "ratios.valid": 0.0,
                        "ratios.test": cfg.data.test_size,
                        "random_state": cfg.data.random_seed,
                    },
                    "preprocessing": {
                        "pipeline": [preproc_name],
                    },
                    "model": {
                        "name": model_label,
                        "class": f"{type(estimator_obj).__module__}.{type(estimator_obj).__name__}",
                        "params": params,
                    },
                    "training": {},
                }
                model_task_mgr.connect_params_sections(
                    {
                        "Run": {"id": run_id, "dataset_key": dataset_key},
                        "Data": {"dataset": params_connect.get("dataset") or {}, "split": params_connect.get("split") or {}},
                        "Preprocessing": params_connect.get("preprocessing") or {},
                        "Model": params_connect.get("model") or {},
                    }
                )
            except Exception:
                pass
            try:
                for m_name, m_val in (train_metrics_display or {}).items():
                    if m_val is None:
                        continue
                    model_task_mgr.report_scalar("train", str(m_name), float(m_val), iteration=0)
                for m_name, m_val in (test_metrics or {}).items():
                    if m_val is None:
                        continue
                    model_task_mgr.report_scalar("test", str(m_name), float(m_val), iteration=0)
            except Exception:
                pass
            # Standardized compare-friendly scalars (title-based keys).
            try:
                has_test_metrics = False
                try:
                    has_test_metrics = bool(
                        isinstance(test_metrics_std, dict) and any(v is not None for v in test_metrics_std.values())
                    )
                except Exception:
                    has_test_metrics = False

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
                    model_task_mgr.report_scalar(
                        "time/predict_train_seconds", "value", float(predict_train_seconds), iteration=0
                    )
                if predict_test_seconds is not None:
                    model_task_mgr.report_scalar(
                        "time/predict_test_seconds", "value", float(predict_test_seconds), iteration=0
                    )
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
                    model_task_mgr.report_scalar(
                        "model/num_features", "value", float(n_features_transformed), iteration=0
                    )
                if n_classes is not None:
                    model_task_mgr.report_scalar("model/num_classes", "value", float(n_classes), iteration=0)
                if model_size_bytes is not None:
                    model_task_mgr.report_scalar("model/size_bytes", "value", float(model_size_bytes), iteration=0)
            except Exception:
                pass
            model_task_mgr.register_output_model(model_path, name=model_label)

            # -----------------------------------------------------------------
            # Interpretation (per-model): Feature Importance / SHAP
            # - Report as Plotly into the per-model task (ClearML Plots)
            # - Save PNG/CSV under output_dir for artifacts
            # -----------------------------------------------------------------
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

                                top_imp = imp_df.head(30).copy()
                                top_imp = top_imp.sort_values(by="importance", ascending=True)
                                fig = px.bar(
                                    top_imp,
                                    x="importance",
                                    y="feature",
                                    orientation="h",
                                    title=f"Feature Importance: {model_label} ({preproc_name})",
                                )
                                model_task_mgr.logger.report_plotly(
                                    title="02_interpretability",
                                    series="feature_importance",
                                    iteration=0,
                                    figure=fig,
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

            if cfg.visualizations.shap_summary:
                # Prefer Plotly summary (mean abs SHAP per feature). Only fall back to PNG when that fails.
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
                                title="02_interpretability",
                                series="shap_summary",
                                iteration=0,
                                figure=fig,
                            )
                        except Exception:
                            pass
            # Ensure plot artifacts exist before uploading/logging to ClearML.
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
                            resid_hist_path.with_suffix(".csv"), index=False
                        )
                    except Exception:
                        pass

            try:
                related_paths = [
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
                    visual_shap_dir / f"{safe_name}.png",
                    visual_shap_dir / f"{safe_name}_importance.csv",
                ]
                model_task_mgr.upload_artifacts([p for p in related_paths if p.exists()])
            except Exception:
                pass
            # Metrics table in ClearML plots
            try:
                metrics_rows = []
                for k, v in (train_metrics_display or {}).items():
                    metrics_rows.append({"metric": str(k), "split": "train", "value": v})
                for k, v in (test_metrics or {}).items():
                    metrics_rows.append({"metric": str(k), "split": "test", "value": v})
                model_task_mgr.report_table("metrics_table", pd.DataFrame(metrics_rows), series="metrics")
            except Exception:
                pass
            # Dataset/config table per model
            try:
                feat_cols = cfg.data.feature_columns or (
                    list(X_train.columns) if hasattr(X_train, "columns") else list(range(X_train.shape[1]))
                )
                ds_info = pd.DataFrame(
                    [
                        {
                            "dataset_id": dataset_id_for_load or "",
                            "input_csv": cfg.data.csv_path,
                            "target_column": cfg.data.target_column or getattr(y, "name", "target") or "target",
                            "feature_columns": ", ".join(feat_cols),
                            "train_rows": int(len(X_train)),
                            "test_rows": int(len(X_test)) if X_test is not None else 0,
                            "preprocessor": preproc_name,
                            "model": model_label,
                            "params": params,
                        }
                    ]
                )
                model_task_mgr.report_table("dataset_info", ds_info, series="config")
            except Exception:
                pass
            # Log key plots into the per-model task
            if model_task_mgr.logger:
                for img_path, series_name in [
                    (visual_pred_dir / f"{safe_name}.png", "pred_vs_actual"),
                    (visual_resid_scatter_dir / f"{safe_name}.png", "residual_scatter"),
                    (visual_resid_hist_dir / f"{safe_name}.png", "residual_hist"),
                    (visual_interp_dir / f"{safe_name}.png", "interpolation_space"),
                    (visual_confusion_dir / f"{safe_name}.png", "confusion_matrix"),
                    (visual_confusion_dir / f"{safe_name}_normalized.png", "confusion_matrix_normalized"),
                    (visual_roc_dir / f"{safe_name}.png", "roc_curve"),
                    (visual_pr_dir / f"{safe_name}.png", "pr_curve"),
                    (visual_fi_dir / f"{safe_name}.png", "feature_importance"),
                    (visual_shap_dir / f"{safe_name}.png", "shap_summary"),
                ]:
                    if img_path.exists():
                        try:
                            title = (
                                "02_interpretability"
                                if series_name in {"feature_importance", "shap_summary"}
                                else "01_performance"
                            )
                            model_task_mgr.logger.report_image(
                                title=title,
                                series=series_name,
                                local_path=str(img_path),
                            )
                            # Also push as plotly if applicable
                            if series_name == "pred_vs_actual" and y_pred_for_plots is not None:
                                fig = build_plotly_pred_vs_actual(
                                    y_train_array,
                                    y_pred_for_plots,
                                    title=f"{model_label} ({preproc_name})",
                                    add_regression_line=True,
                                )
                                if fig is not None:
                                    model_task_mgr.logger.report_plotly(
                                        title=title,
                                        series=series_name,
                                        iteration=0,
                                        figure=fig,
                                    )
                            if series_name == "residual_scatter" and y_pred_for_plots is not None and residuals_for_plots is not None:
                                fig = build_plotly_residual_scatter(
                                    y_pred_for_plots,
                                    residuals_for_plots,
                                    title=f"{model_label} ({preproc_name})",
                                )
                                if fig is not None:
                                    model_task_mgr.logger.report_plotly(
                                        title=title,
                                        series=series_name,
                                        iteration=0,
                                        figure=fig,
                                    )
                            if series_name == "residual_hist" and residuals_for_plots is not None:
                                fig = build_plotly_histogram(
                                    residuals_for_plots.tolist(),
                                    metric_name="residual",
                                    title=f"{model_label} ({preproc_name})",
                                )
                                if fig is not None:
                                    model_task_mgr.logger.report_plotly(
                                        title=title,
                                        series=series_name,
                                        iteration=0,
                                        figure=fig,
                                    )
                            if series_name == "interpolation_space":
                                try:
                                    Xt_full = pipeline_best.named_steps["preprocessor"].transform(X_train)
                                    fig = build_plotly_interpolation_space(
                                        Xt_full,
                                        y_train_array,
                                        title=f"{model_label} ({preproc_name})",
                                    )
                                    if fig is not None:
                                        model_task_mgr.logger.report_plotly(
                                            title=title,
                                            series=series_name,
                                            iteration=0,
                                            figure=fig,
                                        )
                                except Exception:
                                    pass
                        except Exception:
                            pass
            # Debugsamples: link back to summary task + plot files
            try:
                debug_rows = []
                summary_url = None
                if clearml_mgr.task and hasattr(clearml_mgr.task, "get_output_log_web_page"):
                    summary_url = clearml_mgr.task.get_output_log_web_page()
                if summary_url:
                    debug_rows.append({"name": "training-summary", "url": summary_url})
                plot_rows = []
                plot_files = [
                    visual_pred_dir / f"{safe_name}.png",
                    visual_resid_scatter_dir / f"{safe_name}.png",
                    visual_resid_hist_dir / f"{safe_name}.png",
                    visual_interp_dir / f"{safe_name}.png",
                    visual_confusion_dir / f"{safe_name}.png",
                    visual_confusion_dir / f"{safe_name}_normalized.png",
                    visual_roc_dir / f"{safe_name}.png",
                    visual_pr_dir / f"{safe_name}.png",
                    visual_fi_dir / f"{safe_name}.png",
                    visual_shap_dir / f"{safe_name}.png",
                ]
                for pf in plot_files:
                    if pf.exists():
                        plot_rows.append({"file": pf.name, "path": str(pf)})
                if plot_rows:
                    debug_rows.extend(plot_rows)
                if debug_rows:
                    model_task_mgr.report_table("99_debug/debug_samples", pd.DataFrame(debug_rows), series="debug")
            except Exception:
                pass
            # Record task link for summary table
            try:
                task_url = None
                try:
                    if model_task_mgr.task and hasattr(model_task_mgr.task, "get_output_log_web_page"):
                        task_url = model_task_mgr.task.get_output_log_web_page()
                except Exception:
                    task_url = None
                record.update(
                    {
                        "train_seconds": train_seconds,
                        "predict_seconds": predict_seconds,
                        "predict_train_seconds": predict_train_seconds,
                        "predict_test_seconds": predict_test_seconds,
                        "model_size_bytes": model_size_bytes,
                        "num_features": n_features_transformed,
                        "task_id": model_task_mgr.task.id if model_task_mgr.task else "",
                        "url": task_url or "",
                        "link_html": f'<a href="{task_url}">{model_label} ({preproc_name})</a>' if task_url else "",
                        "status": "ok",
                        "error": "",
                    }
                )
                try:
                    has_test_metrics = False
                    try:
                        has_test_metrics = bool(
                            isinstance(test_metrics_std, dict) and any(v is not None for v in test_metrics_std.values())
                        )
                    except Exception:
                        has_test_metrics = False
                    metrics_source = "test" if has_test_metrics else "train"
                    record["metric_source"] = metrics_source
                    metrics_for_record = (test_metrics_std or {}) if has_test_metrics else (train_metrics_std or {})
                    record.update({k: v for k, v in metrics_for_record.items()})
                except Exception:
                    pass
            except Exception:
                pass
            try:
                if model_task_mgr and getattr(model_task_mgr, "task", None):
                    training_task_ids.append(model_task_mgr.task.id)
            except Exception:
                pass
        except Exception as exc:
            try:
                record["status"] = "failed"
                record["error"] = f"unexpected error: {exc}"
            except Exception:
                record = {"model": model_label, "status": "failed", "error": f"unexpected error: {exc}"}
            continue
        finally:
            if model_task_mgr:
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
        # Visualizations (also mirrored to training-summary task for summary)
        if summary_plots_mode == "all" and problem_type == "regression" and cfg.output.generate_plots:
            # Predicted vs actual
            if cfg.visualizations.predicted_vs_actual:
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
                    if y_pred_for_plots is not None and residuals_for_plots is not None:
                        scatter_df = pd.DataFrame(
                            {
                                "actual": y_train_array,
                                "predicted": y_pred_for_plots,
                                "residual": residuals_for_plots,
                                "split": "train",
                            }
                        )
                        scatter_df.to_csv(scatter_path.with_suffix(".csv"), index=False)
                    if clearml_mgr.logger and scatter_path.exists():
                        try:
                            clearml_mgr.logger.report_image(
                                title=f"{model_label} ({preproc_name})",
                                series="pred_vs_actual",
                                local_path=str(scatter_path),
                            )
                            if y_pred_for_plots is not None:
                                fig = build_plotly_pred_vs_actual(
                                    y_train_array,
                                    y_pred_for_plots,
                                    title=f"{model_label} ({preproc_name})",
                                    add_regression_line=True,
                                )
                                if fig is not None:
                                    clearml_mgr.logger.report_plotly(
                                        title=f"{model_label} ({preproc_name})",
                                        series="pred_vs_actual",
                                        iteration=0,
                                        figure=fig,
                                    )
                        except Exception:
                            pass
                    # Summary task also gets plotly scatter
                    if clearml_mgr.logger and y_pred_for_plots is not None:
                        try:
                            fig = build_plotly_pred_vs_actual(
                                y_train_array,
                                y_pred_for_plots,
                                title=f"{model_label} ({preproc_name})",
                                add_regression_line=True,
                            )
                            if fig is not None:
                                clearml_mgr.logger.report_plotly(
                                    title=f"{model_label} ({preproc_name})",
                                    series="pred_vs_actual_models",
                                    iteration=0,
                                    figure=fig,
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
            # Residual scatter
            if cfg.visualizations.residual_scatter:
                resid_scatter_path = visual_resid_scatter_dir / f"{safe_name}.png"
                try:
                    plot_residual_scatter(
                        pipeline_best,
                        X_train,
                        y_train,
                        resid_scatter_path,
                        title=f"Residuals: {model_label} ({preproc_name})",
                        predictions=y_pred_for_plots,
                        residuals=residuals_for_plots,
                    )
                    if y_pred_for_plots is not None and residuals_for_plots is not None:
                        resid_scatter_df = pd.DataFrame(
                            {
                                "predicted": y_pred_for_plots,
                                "residual": residuals_for_plots,
                                "split": "train",
                            }
                        )
                        resid_scatter_df.to_csv(resid_scatter_path.with_suffix(".csv"), index=False)
                    if clearml_mgr.logger and resid_scatter_path.exists():
                        try:
                            clearml_mgr.logger.report_image(
                                title=f"residual_scatter_{model_label}",
                                series="residual_scatter",
                                local_path=str(resid_scatter_path),
                            )
                            if y_pred_for_plots is not None and residuals_for_plots is not None:
                                fig = build_plotly_residual_scatter(
                                    y_pred_for_plots,
                                    residuals_for_plots,
                                    title=f"Residuals: {model_label} ({preproc_name})",
                                )
                                if fig is not None:
                                    clearml_mgr.logger.report_plotly(
                                        title=f"residual_scatter_{model_label}",
                                        series="residual_scatter",
                                        iteration=0,
                                        figure=fig,
                                    )
                        except Exception:
                            pass
                    if clearml_mgr.logger and y_pred_for_plots is not None and residuals_for_plots is not None:
                        try:
                            fig = build_plotly_residual_scatter(
                                y_pred_for_plots,
                                residuals_for_plots,
                                title=f"Residuals: {model_label} ({preproc_name})",
                            )
                            if fig is not None:
                                clearml_mgr.logger.report_plotly(
                                    title=f"residual_scatter_{model_label}",
                                    series="residual_scatter_models",
                                    iteration=0,
                                    figure=fig,
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
            # Residual histogram
            if cfg.visualizations.residual_hist:
                resid_hist_path = visual_resid_hist_dir / f"{safe_name}.png"
                try:
                    plot_residual_hist(
                        pipeline_best,
                        X_train,
                        y_train,
                        resid_hist_path,
                        title=f"Residual Histogram: {model_label} ({preproc_name})",
                        predictions=y_pred_for_plots,
                        residuals=residuals_for_plots,
                    )
                    if residuals_for_plots is not None:
                        resid_hist_df = pd.DataFrame({"residual": residuals_for_plots, "split": "train"})
                        resid_hist_df.to_csv(resid_hist_path.with_suffix(".csv"), index=False)
                    if clearml_mgr.logger and resid_hist_path.exists():
                        try:
                            clearml_mgr.logger.report_image(
                                title=f"residual_hist_{model_label}",
                                series="residual_hist",
                                local_path=str(resid_hist_path),
                            )
                            if residuals_for_plots is not None:
                                fig = build_plotly_histogram(
                                    residuals_for_plots.tolist(),
                                    metric_name="residual",
                                    title=f"Residual Histogram: {model_label} ({preproc_name})",
                                )
                                if fig is not None:
                                    clearml_mgr.logger.report_plotly(
                                        title=f"residual_hist_{model_label}",
                                        series="residual_hist",
                                        iteration=0,
                                        figure=fig,
                                    )
                        except Exception:
                            pass
                    if clearml_mgr.logger and residuals_for_plots is not None:
                        try:
                            fig = build_plotly_histogram(
                                residuals_for_plots.tolist(),
                                metric_name="residual",
                                title=f"Residual Histogram: {model_label} ({preproc_name})",
                            )
                            if fig is not None:
                                clearml_mgr.logger.report_plotly(
                                    title=f"residual_hist_{model_label}",
                                    series="residual_hist_models",
                                    iteration=0,
                                    figure=fig,
                                )
                        except Exception:
                            pass
                except Exception:
                    pass
        # Save predictions and metrics CSV artifacts
        try:
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
                preds_df = pd.DataFrame(preds_rows)
                preds_path = output_dir / f"predictions_{safe_name}.csv"
                preds_df.to_csv(preds_path, index=False)
            metrics_records = []
            for k, v in (train_metrics_display or {}).items():
                metrics_records.append({"metric": str(k), "split": "train", "value": v})
            for k, v in (test_metrics or {}).items():
                metrics_records.append({"metric": str(k), "split": "test", "value": v})
            metrics_df = pd.DataFrame(metrics_records)
            metrics_path = output_dir / f"metrics_{safe_name}.csv"
            metrics_df.to_csv(metrics_path, index=False)
        except Exception:
            pass
    # Record links to per-model tasks for navigation (ClearML on/off).
    df_links = pd.DataFrame(model_task_records) if model_task_records else pd.DataFrame()
    df_links_ranked: Optional[pd.DataFrame] = None
    if not df_links.empty:
        try:
            df_links.to_csv(output_dir / "model_metrics.csv", index=False)
        except Exception:
            pass
        try:
            df_fail = df_links.copy()
            status_bad = pd.Series([False] * len(df_fail))
            error_bad = pd.Series([False] * len(df_fail))
            try:
                if "status" in df_fail.columns:
                    st = df_fail["status"].fillna("").astype(str).str.strip().str.lower()
                    status_bad = ~st.isin(["", "ok"])
            except Exception:
                status_bad = pd.Series([False] * len(df_fail))
            try:
                if "error" in df_fail.columns:
                    er = df_fail["error"].fillna("").astype(str).str.strip()
                    error_bad = er != ""
            except Exception:
                error_bad = pd.Series([False] * len(df_fail))
            df_fail = df_fail[status_bad | error_bad]
            if not df_fail.empty:
                df_fail.to_csv(output_dir / "model_task_failures.csv", index=False)
        except Exception:
            pass

    # Persist recommended model summary to output_dir (ClearML on/off).
    recommended_df = None
    training_primary_metric = str(primary_metric_model).strip().lower() if primary_metric_model else ""
    recommend_metric = training_primary_metric
    recommend_goal = (
        "min" if (recommend_metric and is_loss_metric(recommend_metric, problem_type=problem_type)) else "max"
    )
    recommend_source = "training_primary_metric"
    recommendation_mode = "auto"
    try:
        if cfg.clearml and getattr(cfg.clearml, "recommendation_mode", None):
            recommendation_mode = str(cfg.clearml.recommendation_mode).strip().lower()
    except Exception:
        recommendation_mode = "auto"
    if recommendation_mode not in {"auto", "training", "comparison"}:
        recommendation_mode = "auto"
    use_comparison_ranking = bool(
        recommendation_mode == "comparison" or (recommendation_mode == "auto" and embed_comparison)
    )
    ranking_metrics: List[str] = []
    composite_enabled = True
    composite_metrics = None
    composite_weights = None
    composite_require_all = False
    try:
        # If comparison is embedded, reuse comparison ranking settings for recommendation
        # (e.g., composite_score) so the dashboard is consistent.
        if use_comparison_ranking:
            try:
                from automl_lib.config.loaders import load_comparison_config

                cmp_cfg = None
                cmp_path = os.environ.get("AUTO_ML_COMPARISON_CONFIG_PATH")
                if cmp_path:
                    cmp_cfg = load_comparison_config(Path(str(cmp_path)))
                else:
                    default_cmp = Path("config_comparison.yaml")
                    if default_cmp.exists():
                        cmp_cfg = load_comparison_config(default_cmp)
                    else:
                        cmp_cfg = load_comparison_config(config_path)
                ranking_cfg = getattr(cmp_cfg, "ranking", None) if cmp_cfg else None

                try:
                    if ranking_cfg and getattr(ranking_cfg, "metrics", None):
                        ranking_metrics = [
                            str(m).strip().lower() for m in (ranking_cfg.metrics or []) if str(m).strip()
                        ]
                except Exception:
                    ranking_metrics = []
                if not ranking_metrics:
                    try:
                        if (
                            cmp_cfg
                            and getattr(cmp_cfg, "clearml", None)
                            and getattr(cmp_cfg.clearml, "comparison_metrics", None)
                        ):
                            ranking_metrics = [
                                str(m).strip().lower()
                                for m in (cmp_cfg.clearml.comparison_metrics or [])
                                if str(m).strip()
                            ]
                    except Exception:
                        ranking_metrics = []

                try:
                    pm = str(getattr(ranking_cfg, "primary_metric", "") or "").strip().lower()
                    if pm:
                        recommend_metric = pm
                        recommend_source = "comparison_ranking"
                except Exception:
                    pass

                try:
                    explicit_goal = getattr(ranking_cfg, "goal", None)
                    if explicit_goal:
                        recommend_goal = str(explicit_goal).strip().lower()
                except Exception:
                    pass
                if recommend_metric and recommend_goal not in {"min", "max"}:
                    recommend_goal = (
                        "min" if is_loss_metric(recommend_metric, problem_type=problem_type) else "max"
                    )

                composite_cfg = getattr(ranking_cfg, "composite", None) if ranking_cfg else None
                if composite_cfg:
                    try:
                        composite_enabled = bool(getattr(composite_cfg, "enabled", True))
                    except Exception:
                        composite_enabled = True
                    try:
                        composite_require_all = bool(getattr(composite_cfg, "require_all_metrics", False))
                    except Exception:
                        composite_require_all = False
                    try:
                        composite_metrics = list(getattr(composite_cfg, "metrics", []) or [])
                    except Exception:
                        composite_metrics = None
                    try:
                        composite_weights = dict(getattr(composite_cfg, "weights", {}) or {})
                    except Exception:
                        composite_weights = None
            except Exception:
                pass

        if not df_links.empty and recommend_metric:
            df_rank = df_links.copy()
            # Compute composite_score (if needed) using the same logic as comparison phase.
            try:
                from automl_lib.phases.comparison.meta import build_comparison_metadata

                if (recommend_metric == "composite_score") or composite_enabled:
                    meta_rank = build_comparison_metadata(
                        df_rank.to_dict(orient="records"),
                        output_dir=None,
                        metric_cols=ranking_metrics or None,
                        primary_metric=(recommend_metric or None),
                        goal=(recommend_goal if recommend_source == "comparison_ranking" else None),
                        group_col=None,
                        top_k=None,
                        composite_enabled=composite_enabled,
                        composite_metrics=composite_metrics,
                        composite_weights=composite_weights,
                        composite_require_all=composite_require_all,
                    )
                    if isinstance(meta_rank.get("ranked_df"), pd.DataFrame):
                        df_rank = meta_rank["ranked_df"]
            except Exception:
                pass

            # If composite_score is requested but cannot be computed, fall back.
            if recommend_metric == "composite_score" and "composite_score" not in df_rank.columns:
                if training_primary_metric and training_primary_metric in df_rank.columns:
                    recommend_metric = training_primary_metric
                    recommend_source = "training_primary_metric_fallback"
                    recommend_goal = (
                        "min"
                        if is_loss_metric(training_primary_metric, problem_type=problem_type)
                        else "max"
                    )
            if recommend_metric not in df_rank.columns:
                if training_primary_metric and training_primary_metric in df_rank.columns:
                    recommend_metric = training_primary_metric
                    recommend_source = "training_primary_metric_fallback"
                    recommend_goal = (
                        "min"
                        if is_loss_metric(training_primary_metric, problem_type=problem_type)
                        else "max"
                    )

            if recommend_metric in df_rank.columns:
                try:
                    if "status" in df_rank.columns:
                        status_norm = df_rank["status"].fillna("").astype(str).str.strip().str.lower()
                        df_rank = df_rank[status_norm.isin(["ok", ""])]
                except Exception:
                    pass
                df_rank["_metric"] = pd.to_numeric(df_rank[recommend_metric], errors="coerce")
                df_rank["_train_seconds"] = pd.to_numeric(df_rank.get("train_seconds"), errors="coerce")
                df_rank["_predict_seconds"] = pd.to_numeric(df_rank.get("predict_seconds"), errors="coerce")
                df_rank["_model_size_bytes"] = pd.to_numeric(df_rank.get("model_size_bytes"), errors="coerce")
                df_rank["_num_features"] = pd.to_numeric(df_rank.get("num_features"), errors="coerce")
                df_rank = df_rank[df_rank["_metric"].notna()]
                if not df_rank.empty:
                    ascending = [recommend_goal == "min", True, True, True, True]
                    sort_cols = ["_metric", "_train_seconds", "_predict_seconds", "_model_size_bytes", "_num_features"]
                    df_rank = df_rank.sort_values(by=sort_cols, ascending=ascending, na_position="last")
                    best_row = df_rank.iloc[0].to_dict()
                    try:
                        df_links_ranked = df_rank.copy().reset_index(drop=True)
                        helper_cols = [c for c in df_links_ranked.columns if str(c).startswith("_")]
                        if helper_cols:
                            df_links_ranked = df_links_ranked.drop(columns=helper_cols, errors="ignore")
                        df_links_ranked.insert(0, "rank", range(1, len(df_links_ranked) + 1))

                        rec_task_id = str(best_row.get("task_id") or "").strip()
                        rec_model = str(best_row.get("model") or "").strip()
                        rec_preproc = str(best_row.get("preprocessor") or "").strip()
                        is_rec = pd.Series([False] * len(df_links_ranked))
                        try:
                            if rec_task_id and "task_id" in df_links_ranked.columns:
                                is_rec = df_links_ranked["task_id"].astype(str) == rec_task_id
                            elif rec_model and "model" in df_links_ranked.columns and "preprocessor" in df_links_ranked.columns:
                                is_rec = (df_links_ranked["model"].astype(str) == rec_model) & (
                                    df_links_ranked["preprocessor"].astype(str) == rec_preproc
                                )
                        except Exception:
                            is_rec = pd.Series([False] * len(df_links_ranked))
                        df_links_ranked.insert(1, "is_recommended", is_rec.astype(bool))

                        preferred_cols = [
                            "rank",
                            "is_recommended",
                            "model",
                            "preprocessor",
                            "metric_source",
                            recommend_metric,
                            "composite_score",
                            training_primary_metric if training_primary_metric else "",
                            "train_seconds",
                            "predict_seconds",
                            "model_size_bytes",
                            "num_features",
                            "task_id",
                            "url",
                        ]
                        ordered_cols = []
                        for c in preferred_cols:
                            if c and c in df_links_ranked.columns and c not in ordered_cols:
                                ordered_cols.append(c)
                        for c in df_links_ranked.columns:
                            if c not in ordered_cols:
                                ordered_cols.append(c)
                        df_links_ranked = df_links_ranked[ordered_cols]
                        try:
                            df_links_ranked.to_csv(output_dir / "model_tasks_ranked.csv", index=False)
                        except Exception:
                            pass
                    except Exception:
                        df_links_ranked = None
                    recommended = {
                        "run_id": run_id,
                        "dataset_key": dataset_key,
                        "training_primary_metric": training_primary_metric,
                        "primary_metric": recommend_metric,
                        "goal": recommend_goal,
                        "recommendation_mode": recommendation_mode,
                        "ranking_source": recommend_source,
                        "metric_source": best_row.get("metric_source", ""),
                        "model": best_row.get("model", ""),
                        "preprocessor": best_row.get("preprocessor", ""),
                        recommend_metric: best_row.get(recommend_metric),
                        "train_seconds": best_row.get("train_seconds"),
                        "predict_seconds": best_row.get("predict_seconds"),
                        "model_size_bytes": best_row.get("model_size_bytes"),
                        "num_features": best_row.get("num_features"),
                        "task_id": best_row.get("task_id", ""),
                        "url": best_row.get("url", ""),
                        "link_html": best_row.get("link_html", ""),
                    }
                    # Include ranking metrics + composite_score for transparency.
                    metrics_to_show = list(ranking_metrics or [])
                    if "composite_score" in best_row and "composite_score" not in metrics_to_show:
                        metrics_to_show.append("composite_score")
                    for m in metrics_to_show:
                        key = str(m).strip().lower()
                        if not key or key in recommended:
                            continue
                        if key in best_row:
                            recommended[key] = best_row.get(key)
                    recommended_df = pd.DataFrame([recommended])
                    try:
                        recommended_df.to_csv(output_dir / "recommended_model.csv", index=False)
                    except Exception:
                        pass
                    # Persist recommendation rationale (config + selected row) for transparency.
                    try:
                        rationale_row = recommended_df.iloc[0].to_dict()
                    except Exception:
                        rationale_row = {}
                    rationale = {
                        "run_id": run_id,
                        "dataset_key": dataset_key,
                        "comparison_mode": (getattr(cfg.clearml, "comparison_mode", "disabled") if cfg.clearml else "disabled"),
                        "recommendation_mode": recommendation_mode,
                        "training_primary_metric": training_primary_metric,
                        "recommend_metric": recommend_metric,
                        "recommend_goal": recommend_goal,
                        "ranking_source": recommend_source,
                        "ranking_metrics": ranking_metrics,
                        "composite_enabled": composite_enabled,
                        "composite_metrics": composite_metrics,
                        "composite_weights": composite_weights,
                        "composite_require_all": composite_require_all,
                        "selected": rationale_row,
                        "comparison_config_path": os.environ.get("AUTO_ML_COMPARISON_CONFIG_PATH") or "",
                    }
                    try:
                        import json as _json

                        (output_dir / "recommendation_rationale.json").write_text(
                            _json.dumps(rationale, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
                    try:
                        md_lines = []
                        md_lines.append("# Recommendation rationale")
                        md_lines.append("")
                        md_lines.append(f"- recommendation_mode: `{recommendation_mode}`")
                        md_lines.append(f"- ranking_source: `{recommend_source}`")
                        md_lines.append(f"- recommend_metric: `{recommend_metric}` ({recommend_goal})")
                        if training_primary_metric:
                            md_lines.append(f"- training_primary_metric: `{training_primary_metric}`")
                        if ranking_metrics:
                            md_lines.append(f"- ranking_metrics: {', '.join(f'`{m}`' for m in ranking_metrics)}")
                        if composite_weights:
                            md_lines.append("")
                            md_lines.append("## Composite weights")
                            for k, v in composite_weights.items():
                                md_lines.append(f"- `{k}`: {v}")
                        md_lines.append("")
                        md_lines.append("## Selected model")
                        for key in [
                            "model",
                            "preprocessor",
                            "task_id",
                            "metric_source",
                            recommend_metric,
                            "composite_score",
                            "train_seconds",
                            "predict_seconds",
                            "model_size_bytes",
                            "num_features",
                            "url",
                        ]:
                            if key in rationale_row and rationale_row.get(key) not in (None, ""):
                                md_lines.append(f"- {key}: {rationale_row.get(key)}")
                        (output_dir / "recommendation_rationale.md").write_text(
                            "\n".join(md_lines).strip() + "\n",
                            encoding="utf-8",
                        )
                    except Exception:
                        pass
    except Exception:
        recommended_df = None

    # ClearML dashboard: show model tasks + recommendation.
    if clearml_mgr.logger:
        try:
            df_tasks_show = None
            tasks_total = None
            tasks_shown = None
            top_k_display = 20
            if not df_links.empty:
                df_tasks_all = df_links_ranked if (df_links_ranked is not None and not df_links_ranked.empty) else df_links
                try:
                    tasks_total = int(len(df_tasks_all))
                except Exception:
                    tasks_total = None
                try:
                    df_tasks_show = df_tasks_all.head(top_k_display) if (tasks_total and tasks_total > top_k_display) else df_tasks_all
                    tasks_shown = int(len(df_tasks_show))
                except Exception:
                    df_tasks_show = df_tasks_all
                    try:
                        tasks_shown = int(len(df_tasks_show))
                    except Exception:
                        tasks_shown = None
                clearml_mgr.report_table("02_leaderboard/model_tasks", df_tasks_show, series="leaderboard")
            if recommended_df is not None and not recommended_df.empty:
                clearml_mgr.report_table("01_overview/recommended_model", recommended_df, series="overview")
                try:
                    val_best = float(pd.to_numeric(recommended_df.iloc[0].get(recommend_metric), errors="coerce"))
                    if val_best == val_best:  # not NaN
                        clearml_mgr.report_scalar(f"01_overview/best_{recommend_metric}", "value", val_best, iteration=0)
                except Exception:
                    pass
                try:
                    rationale_md = output_dir / "recommendation_rationale.md"
                    if rationale_md.exists():
                        text = rationale_md.read_text(encoding="utf-8")
                        clearml_mgr.logger.report_text(
                            text.replace("\n", "<br/>"),
                            title="01_overview/recommendation_rationale",
                        )
                except Exception:
                    pass
            # Failures table (helps explain missing/filtered candidates)
            try:
                if not df_links.empty:
                    df_fail = df_links.copy()
                    status_bad = pd.Series([False] * len(df_fail))
                    error_bad = pd.Series([False] * len(df_fail))
                    try:
                        if "status" in df_fail.columns:
                            st = df_fail["status"].fillna("").astype(str).str.strip().str.lower()
                            status_bad = ~st.isin(["", "ok"])
                    except Exception:
                        status_bad = pd.Series([False] * len(df_fail))
                    try:
                        if "error" in df_fail.columns:
                            er = df_fail["error"].fillna("").astype(str).str.strip()
                            error_bad = er != ""
                    except Exception:
                        error_bad = pd.Series([False] * len(df_fail))
                    df_fail = df_fail[status_bad | error_bad]
                    if not df_fail.empty:
                        try:
                            df_fail.to_csv(output_dir / "model_task_failures.csv", index=False)
                        except Exception:
                            pass
                        clearml_mgr.report_table("99_debug/model_task_failures", df_fail.head(50), series="debug")
            except Exception:
                pass
            try:
                link_lines = []
                df_links_src = df_tasks_show if isinstance(df_tasks_show, pd.DataFrame) else None
                if df_links_src is None or df_links_src.empty:
                    df_links_src = df_links_ranked if isinstance(df_links_ranked, pd.DataFrame) else None
                if df_links_src is None or df_links_src.empty:
                    df_links_src = df_links
                if isinstance(df_links_src, pd.DataFrame) and not df_links_src.empty:
                    for _, row in df_links_src.iterrows():
                        link = str(row.get("link_html") or "").strip()
                        if not link:
                            continue
                        try:
                            rank = int(row.get("rank")) if row.get("rank") is not None else None
                        except Exception:
                            rank = None
                        link_lines.append(f"{rank}. {link}" if rank is not None else link)
                links_html = "<br/>".join(link_lines)
                if links_html:
                    suffix = ""
                    try:
                        if tasks_total and tasks_shown and tasks_shown < tasks_total:
                            suffix = f"<br/>(showing top {tasks_shown} of {tasks_total})"
                    except Exception:
                        suffix = ""
                    clearml_mgr.logger.report_text(
                        f"Model task links:<br/>{links_html}{suffix}",
                        title="01_overview/model_task_links",
                    )
            except Exception:
                pass
        except Exception:
            pass
        try:
            rec_html = ""
            if recommended_df is not None and not recommended_df.empty:
                try:
                    rec_row = recommended_df.iloc[0].to_dict()
                    rec_html = str(rec_row.get("link_html") or "")
                    if not rec_html:
                        rec_html = f"{rec_row.get('model', '')} ({rec_row.get('preprocessor', '')})"
                except Exception:
                    rec_html = ""
            summary_lines = [
                f"run_id: {run_id}",
                f"dataset_key: {dataset_key}",
                f"dataset_id: {dataset_id_for_load or ''}",
                f"csv_path: {cfg.data.csv_path or ''}",
                f"problem_type: {problem_type}",
                f"train_rows: {int(len(X_train))}",
                f"test_rows: {int(len(X_test)) if X_test is not None else 0}",
                f"training_primary_metric: {training_primary_metric} ({'min' if (training_primary_metric and is_loss_metric(training_primary_metric, problem_type=problem_type)) else 'max'})",
                f"recommend_metric: {recommend_metric} ({recommend_goal}) mode={recommendation_mode} source={recommend_source}",
                f"comparison_mode: {getattr(cfg.clearml, 'comparison_mode', 'disabled') if cfg.clearml else 'disabled'}",
                f"output_dir: {str(output_dir)}",
            ]
            if rec_html:
                summary_lines.append(f"recommended: {rec_html}")
            clearml_mgr.logger.report_text("<br/>".join(summary_lines), title="01_overview/run_summary")
        except Exception:
            pass
        if summary_plots_mode == "best":
            try:
                if recommended_df is not None and not recommended_df.empty:
                    rec = recommended_df.iloc[0].to_dict()
                    rec_model = str(rec.get("model") or "")
                    rec_preproc = str(rec.get("preprocessor") or "")
                    rec_safe_name = (
                        f"{rec_model.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_"
                        f"{rec_preproc.replace('|', '_')}"
                    )
                    candidates = [
                        (visual_pred_dir / f"{rec_safe_name}.png", "pred_vs_actual"),
                        (visual_resid_scatter_dir / f"{rec_safe_name}.png", "residual_scatter"),
                        (visual_resid_hist_dir / f"{rec_safe_name}.png", "residual_hist"),
                        (visual_confusion_dir / f"{rec_safe_name}.png", "confusion_matrix"),
                        (visual_confusion_dir / f"{rec_safe_name}_normalized.png", "confusion_matrix_normalized"),
                        (visual_roc_dir / f"{rec_safe_name}.png", "roc_curve"),
                        (visual_pr_dir / f"{rec_safe_name}.png", "pr_curve"),
                    ]
                    for img_path, series_name in candidates:
                        if img_path.exists():
                            try:
                                clearml_mgr.logger.report_image(
                                    title=f"04_best_model/{series_name}",
                                    series="best_model",
                                    local_path=str(img_path),
                                )
                            except Exception:
                                pass
            except Exception:
                pass

    # Debugsamples: plot artifact listing for summary task
    try:
        plot_df = build_plot_artifacts_table(output_dir)
        if not plot_df.empty:
            try:
                plot_df.to_csv(output_dir / "plot_artifacts.csv", index=False)
            except Exception:
                pass
            if clearml_mgr.logger:
                clearml_mgr.report_table("99_debug/plot_artifacts", plot_df, series="debug")
    except Exception:
        pass
    # Upload key artifacts to ClearML
    try:
        artifacts_to_upload = [results_path, results_path.with_suffix(".json")]
        rec_csv = output_dir / "recommended_model.csv"
        if rec_csv.exists():
            artifacts_to_upload.append(rec_csv)
        rationale_json = output_dir / "recommendation_rationale.json"
        if rationale_json.exists():
            artifacts_to_upload.append(rationale_json)
        rationale_md = output_dir / "recommendation_rationale.md"
        if rationale_md.exists():
            artifacts_to_upload.append(rationale_md)
        model_metrics_csv = output_dir / "model_metrics.csv"
        if model_metrics_csv.exists():
            artifacts_to_upload.append(model_metrics_csv)
        ranked_tasks_csv = output_dir / "model_tasks_ranked.csv"
        if ranked_tasks_csv.exists():
            artifacts_to_upload.append(ranked_tasks_csv)
        failures_csv = output_dir / "model_task_failures.csv"
        if failures_csv.exists():
            artifacts_to_upload.append(failures_csv)
        plot_artifacts_csv = output_dir / "plot_artifacts.csv"
        if plot_artifacts_csv.exists():
            artifacts_to_upload.append(plot_artifacts_csv)
        # feature space projection artifact (PCA on first model) if exists
        proj_csv = output_dir / "interpolation_space" / "feature_space_projection.csv"
        if proj_csv.exists():
            artifacts_to_upload.append(proj_csv)
        if cfg.output.save_models:
            artifacts_to_upload.extend((output_dir / "models").glob("*.joblib"))
        if cfg.output.generate_plots:
            artifacts_to_upload.extend(output_dir.glob("*.png"))
            artifacts_to_upload.extend((output_dir / "scatter_plots").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "residual_scatter").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "residual_hist").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "metric_hists").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "interpolation_space").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "confusion_matrices").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "confusion_matrices").glob("*.csv"))
            artifacts_to_upload.extend((output_dir / "roc_curves").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "roc_curves").glob("*.csv"))
            artifacts_to_upload.extend((output_dir / "pr_curves").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "pr_curves").glob("*.csv"))
            artifacts_to_upload.extend((output_dir / "feature_importances_models").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "shap_summaries_models").glob("*.png"))
            artifacts_to_upload.extend((output_dir / "scatter_plots").glob("*.csv"))
            artifacts_to_upload.extend((output_dir / "residual_scatter").glob("*.csv"))
            artifacts_to_upload.extend((output_dir / "residual_hist").glob("*.csv"))
            artifacts_to_upload.extend(output_dir.glob("predictions_*.csv"))
            artifacts_to_upload.extend(output_dir.glob("metrics_*.csv"))
        clearml_mgr.upload_artifacts([p for p in artifacts_to_upload if p.exists()])
    except Exception:
        pass
    finally:
        try:
            if clearml_mgr.task:
                clearml_mgr.task.flush(wait_for_uploads=True)
        except Exception:
            pass
        # PipelineController owns the step task lifecycle, but training-summary can be a separate task.
        # Close only when it is safe (non-step task) so it appears as completed in ClearML UI.
        should_close = os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1"
        if not should_close and clearml_mgr.task:
            step_task_id = (os.environ.get("CLEARML_TASK_ID") or "").strip()
            try:
                should_close = (not step_task_id) or (str(clearml_mgr.task.id) != step_task_id)
            except Exception:
                should_close = False
        if should_close:
            clearml_mgr.close()
    metrics_for_comparison = model_task_records
    try:
        filtered = []
        for r in model_task_records:
            if not isinstance(r, dict):
                continue
            st = str(r.get("status") or "").strip().lower()
            if not st or st == "ok":
                filtered.append(r)
        metrics_for_comparison = filtered
    except Exception:
        metrics_for_comparison = model_task_records
    return {
        "dataset_id": dataset_id_for_load,
        "summary_task_id": clearml_mgr.task.id if clearml_mgr.task else None,
        "training_task_ids": training_task_ids,
        "metrics": metrics_for_comparison,  # 各モデルのメトリクス/リンクを比較フェーズに渡す
    }


def _resolve_dataset_id(cfg: TrainingConfig) -> Optional[str]:
    """
    Decide which ClearML Dataset ID to use:
    1) preprocessed_dataset_id / edited_dataset_id / raw_dataset_id (in that order)
    2) data.dataset_id if provided
    3) latest dataset in dataset_project (optional dataset_name matching if dataset_id is not a 32-char id)
    """
    # explicit fields take priority
    for cand in [
        getattr(cfg.clearml, "preprocessed_dataset_id", None) if cfg.clearml else None,
        getattr(cfg.clearml, "edited_dataset_id", None) if cfg.clearml else None,
        getattr(cfg.clearml, "raw_dataset_id", None) if cfg.clearml else None,
        getattr(cfg.data, "dataset_id", None),
    ]:
        if cand:
            norm = _normalize_dataset_id(cand, cfg)
            if norm:
                return norm
    # fallback: latest dataset in project
    try:
        from clearml import Dataset  # type: ignore

        ds_list = Dataset.list_datasets(dataset_project=cfg.clearml.dataset_project if cfg.clearml else None)
        if ds_list:
            return ds_list[0].get("id")
    except Exception:
        return None
    return None


def _normalize_dataset_id(raw_id: str, cfg: TrainingConfig) -> Optional[str]:
    """If raw_id looks like a name, try resolving by name+project. Otherwise return as-is when valid."""
    if not raw_id:
        return None
    raw_str = str(raw_id)
    try:
        from clearml import Dataset  # type: ignore

        # if it's a typical task/dataset id, just return
        if len(raw_str) == 32:
            return raw_str
        # otherwise try to resolve by name within project
        ds_obj = Dataset.get(dataset_name=raw_str, dataset_project=cfg.clearml.dataset_project if cfg.clearml else None)
        if ds_obj:
            return ds_obj.id
    except Exception:
        pass
    return raw_str if raw_str else None
