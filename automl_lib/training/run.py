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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from automl_lib.clearml.bootstrap import ensure_clearml_config_file
from automl_lib.config.loaders import load_training_config
from automl_lib.config.schemas import TrainingConfig
from automl_lib.preprocessing.preprocessors import generate_preprocessors
from automl_lib.registry.metrics import add_derived_metrics, is_loss_metric

from .clearml_integration import ClearMLManager, _import_clearml, ensure_local_dataset_copy, find_first_csv
from .data_loader import get_feature_types, infer_problem_type, load_dataset, split_data
from .ensemble import build_stacking, build_voting
from .evaluation import _get_cv_splitter, _get_scoring, evaluate_model_combinations
from .interpretation import extract_feature_importance, plot_feature_importance, plot_shap_summary
from .model_factory import ModelInstance, prepare_tabpfn_params
from .search import generate_param_combinations
from .tabpfn_utils import OfflineTabPFNRegressor
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
    training_task_ids: List[str] = []
    parent_project = cfg.clearml.project_name if cfg.clearml else None
    clearml_mgr = ClearMLManager(
        cfg.clearml,
        task_name=cfg.clearml.task_name if cfg.clearml and cfg.clearml.task_name else "training-summary",
        task_type="training",
        default_project=cfg.clearml.project_name if cfg.clearml and cfg.clearml.project_name else "AutoML",
    )
    # derive train_models project under the same parent
    train_models_project = f"{parent_project}/train_models" if parent_project else f"train_models"
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
        local_copy = ensure_local_dataset_copy(dataset_id_for_load, Path(cfg.output.output_dir) / "clearml_dataset")
        if cfg.clearml and cfg.clearml.enabled and not local_copy:
            raise ValueError(f"Failed to download ClearML Dataset (dataset_id={dataset_id_for_load})")
        csv_override = find_first_csv(local_copy) if local_copy else None
        if csv_override:
            cfg.data.csv_path = str(csv_override)

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
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save only the best result per model. Determine primary metric to rank within each model
    primary_metric_model = cfg.evaluation.primary_metric or ("r2" if problem_type == "regression" else "accuracy")
    if primary_metric_model not in results_df.columns:
        available_cols = [c for c in results_df.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_model = available_cols[0] if available_cols else None
    # Identify best row for each model (including ensembles) based on primary metric
    best_rows: Dict[str, pd.Series] = {}
    if primary_metric_model:
        for _, row in results_df.iterrows():
            if pd.isna(row.get(primary_metric_model)):
                continue
            model_label = row["model"]
            score = row[primary_metric_model]
            if model_label not in best_rows or score > best_rows[model_label][primary_metric_model]:
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
    clearml_mgr.report_table("cv_results_best", results_df_best, series="cv")
    # bar chart for metrics comparison
    try:
        import plotly.express as px  # type: ignore

        metrics_list = ["r2", "mse", "rmse"]
        metric_cols = [m for m in metrics_list if m in results_df_best.columns]
        if metric_cols:
            df_bar = results_df_best[["model"] + metric_cols].melt(id_vars=["model"], var_name="metric", value_name="value")
            fig = px.bar(df_bar, x="model", y="value", color="metric", barmode="group", title="Model metric comparison")
            if clearml_mgr.logger:
                clearml_mgr.logger.report_plotly(title="metric_bar", series="metrics", iteration=0, figure=fig)
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
    # Choose the overall best model across models using the filtered results
    primary_metric_global = primary_metric_model
    if not primary_metric_global or primary_metric_global not in results_df_best.columns:
        cols_available = [c for c in results_df_best.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_global = cols_available[0] if cols_available else None
    if primary_metric_global:
        best_row = results_df_best.loc[results_df_best[primary_metric_global].idxmax()]
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
    # Create only if enabled and regression
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.predicted_vs_actual:
        visual_pred_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_scatter:
        visual_resid_scatter_dir.mkdir(exist_ok=True)
    if problem_type == "regression" and cfg.output.generate_plots and cfg.visualizations.residual_hist:
        visual_resid_hist_dir.mkdir(exist_ok=True)
    visual_metric_hist_dir.mkdir(exist_ok=True)
    visual_interp_dir.mkdir(exist_ok=True)
    if cfg.visualizations.feature_importance:
        visual_fi_dir.mkdir(exist_ok=True)
    if cfg.visualizations.shap_summary:
        visual_shap_dir.mkdir(exist_ok=True)
    model_task_records: List[Dict[str, Any]] = []
    # Iterate over best models (per model training tasks)
    for model_label, row in best_rows.items():
        model_task_mgr = None
        try:
            preproc_name = row["preprocessor"]
            params = row["params"] if isinstance(row.get("params"), dict) else {}
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
                    estimator_obj = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
            else:
                # Base model
                try:
                    cls = _get_model_class(model_label, problem_type)
                except Exception:
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
                except Exception:
                    continue
            if problem_type == "regression":
                estimator_obj = _maybe_wrap_with_target_scaler(estimator_obj, cfg, problem_type)
            # Build and fit pipeline
            from sklearn.pipeline import Pipeline as SKPipeline
            pipeline_best = SKPipeline([
                ("preprocessor", transformer),
                ("model", estimator_obj),
            ])
            try:
                pipeline_best.fit(X_train, y_train)
            except Exception:
                continue
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
            y_pred_for_plots: Optional[np.ndarray] = None
            residuals_for_plots: Optional[np.ndarray] = None
            r2_for_plots: Optional[float] = None
            # Compute train predictions/metrics
            try:
                y_pred_train = np.asarray(pipeline_best.predict(X_train))
                y_pred_for_plots = y_pred_train
                residuals_for_plots = y_train_array - y_pred_train
                r2_for_plots = float(r2_score(y_train_array, y_pred_train))
            except Exception:
                y_pred_for_plots = None
                residuals_for_plots = None
                r2_for_plots = None
            # Optionally compute test predictions/metrics
            test_metrics = {}
            y_pred_test = None
            if X_test is not None and len(X_test) > 0:
                try:
                    y_pred_test = np.asarray(pipeline_best.predict(X_test))
                    test_metrics = {
                        "R2": float(r2_score(y_test, y_pred_test)),
                        "MSE": float(mean_squared_error(y_test, y_pred_test)),
                        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                        "MAE": float(mean_absolute_error(y_test, y_pred_test)),
                    }
                except Exception:
                    test_metrics = {}
            # Safe name for file outputs
            safe_name = f"{model_label.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')}_{preproc_name.replace('|', '_')}"
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
            # Summary scatter/hist in training-summary task (ClearML scatter2d/hist)
            if clearml_mgr.logger and y_pred_for_plots is not None:
                try:
                    if Scatter2D:
                        scatter = Scatter2D(mode=Scatter2D.Mode.markers, x=y_train_array.tolist(), y=y_pred_for_plots.tolist())
                        clearml_mgr.logger.report_scatter2d(
                            title=f"{model_label} y_true vs y_pred",
                            series=model_label,
                            iteration=0,
                            scatter=scatter,
                        )
                        # 45 degree and regression line
                        min_val = float(min(np.min(y_train_array), np.min(y_pred_for_plots)))
                        max_val = float(max(np.max(y_train_array), np.max(y_pred_for_plots)))
                        reg_line_x = [min_val, max_val]
                        try:
                            coef = np.polyfit(y_train_array, y_pred_for_plots, 1)
                            reg_line_y = [coef[0] * min_val + coef[1], coef[0] * max_val + coef[1]]
                        except Exception:
                            reg_line_y = reg_line_x
                        clearml_mgr.logger.report_scatter2d(
                            title=f"{model_label} y_true vs y_pred",
                            series=f"{model_label}_ideal",
                            iteration=0,
                            scatter=Scatter2D(mode=Scatter2D.Mode.lines, x=reg_line_x, y=reg_line_x),
                        )
                        clearml_mgr.logger.report_scatter2d(
                            title=f"{model_label} y_true vs y_pred",
                            series=f"{model_label}_regline",
                            iteration=0,
                            scatter=Scatter2D(mode=Scatter2D.Mode.lines, x=reg_line_x, y=reg_line_y),
                        )
                except Exception:
                    pass
            # Interpolation / feature space plot (PCA on transformed features)
            interp_path = visual_interp_dir / f"{safe_name}.png"
            proj_csv_path = visual_interp_dir / "feature_space_projection.csv"
            try:
                Xt_full = pipeline_best.named_steps["preprocessor"].transform(X_train)
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
            # ClearML per-model training task
            dataset_label = (dataset_id_for_load or Path(cfg.data.csv_path).stem or "dataset").replace(" ", "_")
            model_task_name = f"train_{model_label}_ds={dataset_label}"
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
                        tags_to_add = ["regression", model_label]
                        if preproc_name:
                            tags_to_add.append(preproc_name)
                        try:
                            task_obj.add_tags(tags_to_add)
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
                parent=clearml_mgr.task.id if clearml_mgr.task else None,
                existing_task=task_obj,
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
                model_task_mgr.connect_params(params_connect)
            except Exception:
                pass
            try:
                if r2_for_plots is not None:
                    model_task_mgr.report_scalar("train", "R2", float(r2_for_plots), iteration=0)
                if mse_val is not None:
                    model_task_mgr.report_scalar("train", "MSE", float(mse_val), iteration=0)
                if rmse_val is not None:
                    model_task_mgr.report_scalar("train", "RMSE", float(rmse_val), iteration=0)
                if mae_val is not None:
                    model_task_mgr.report_scalar("train", "MAE", float(mae_val), iteration=0)
                if test_metrics:
                    for m_name, m_val in test_metrics.items():
                        if m_val is not None:
                            model_task_mgr.report_scalar("test", m_name, float(m_val), iteration=0)
            except Exception:
                pass
            model_task_mgr.register_output_model(model_path, name=model_label)
            try:
                related_plots = list((output_dir / "scatter_plots").glob(f"{model_label.replace(' ', '_')}*.png"))
                related_plots += list((output_dir / "feature_importances_models").glob(f"{model_label.replace(' ', '_')}*.png"))
                related_csvs = list((output_dir / "scatter_plots").glob(f"{model_label.replace(' ', '_')}*.csv"))
                related_csvs += list((output_dir / "residual_scatter").glob(f"{model_label.replace(' ', '_')}*.csv"))
                related_csvs += list((output_dir / "residual_hist").glob(f"{model_label.replace(' ', '_')}*.csv"))
                related_plots += list((output_dir / "interpolation_space").glob(f"{model_label.replace(' ', '_')}*.png"))
                model_task_mgr.upload_artifacts([p for p in (related_plots + related_csvs) if p.exists()])
            except Exception:
                pass
            # Metrics table in ClearML plots
            try:
                metrics_rows = [
                    {"metric": "R2", "split": "train", "value": r2_for_plots},
                    {"metric": "MSE", "split": "train", "value": mse_val},
                    {"metric": "RMSE", "split": "train", "value": rmse_val},
                    {"metric": "MAE", "split": "train", "value": mae_val},
                ]
                if test_metrics:
                    for k, v in test_metrics.items():
                        metrics_rows.append({"metric": k, "split": "test", "value": v})
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
                    (visual_fi_dir / f"{safe_name}.png", "feature_importance"),
                    (visual_shap_dir / f"{safe_name}.png", "shap_summary"),
                ]:
                    if img_path.exists():
                        try:
                            model_task_mgr.logger.report_image(
                                title=img_path.stem,
                                series=series_name,
                                local_path=str(img_path),
                            )
                            # Also push as plotly if applicable
                            if series_name == "pred_vs_actual" and y_pred_for_plots is not None:
                                fig = build_plotly_pred_vs_actual(
                                    y_train_array,
                                    y_pred_for_plots,
                                    title=img_path.stem,
                                    add_regression_line=True,
                                )
                                if fig is not None:
                                    model_task_mgr.logger.report_plotly(
                                        title=img_path.stem,
                                        series=series_name,
                                        iteration=0,
                                        figure=fig,
                                    )
                            if series_name == "residual_scatter" and y_pred_for_plots is not None and residuals_for_plots is not None:
                                fig = build_plotly_residual_scatter(
                                    y_pred_for_plots,
                                    residuals_for_plots,
                                    title=img_path.stem,
                                )
                                if fig is not None:
                                    model_task_mgr.logger.report_plotly(
                                        title=img_path.stem,
                                        series=series_name,
                                        iteration=0,
                                        figure=fig,
                                    )
                            if series_name == "residual_hist" and residuals_for_plots is not None:
                                fig = build_plotly_histogram(
                                    residuals_for_plots.tolist(),
                                    metric_name="residual",
                                    title=img_path.stem,
                                )
                                if fig is not None:
                                    model_task_mgr.logger.report_plotly(
                                        title=img_path.stem,
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
                                        title=img_path.stem,
                                    )
                                    if fig is not None:
                                        model_task_mgr.logger.report_plotly(
                                            title=img_path.stem,
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
                    visual_fi_dir / f"{safe_name}.png",
                    visual_shap_dir / f"{safe_name}.png",
                ]
                for pf in plot_files:
                    if pf.exists():
                        plot_rows.append({"file": pf.name, "path": str(pf)})
                if plot_rows:
                    debug_rows.extend(plot_rows)
                if debug_rows:
                    model_task_mgr.report_table("DEbugsamples", pd.DataFrame(debug_rows), series="DEbugsamples")
            except Exception:
                pass
            # Record task link for summary table
            try:
                task_url = None
                if model_task_mgr.task and hasattr(model_task_mgr.task, "get_output_log_web_page"):
                    task_url = model_task_mgr.task.get_output_log_web_page()
                model_task_records.append(
                    {
                        "model": model_label,
                        "preprocessor": preproc_name,
                        "task_id": model_task_mgr.task.id if model_task_mgr.task else "",
                        "url": task_url or "",
                        "link_html": f'<a href="{task_url}">{model_label} ({preproc_name})</a>' if task_url else "",
                        "r2": r2_for_plots,
                        "mse": mse_val,
                        "rmse": rmse_val,
                        "mae": mae_val,
                    }
                )
            except Exception:
                pass
            if model_task_mgr.task:
                training_task_ids.append(model_task_mgr.task.id)
        finally:
            if model_task_mgr:
                model_task_mgr.close()
        # Visualizations (also mirrored to training-summary task for summary)
        if problem_type == "regression" and cfg.output.generate_plots:
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
        # Feature importance per model
        if cfg.visualizations.feature_importance:
            try:
                # Get feature names
                try:
                    fitted_preprocessor = pipeline_best.named_steps["preprocessor"]
                    fnames = fitted_preprocessor.get_feature_names_out()
                except Exception:
                    fitted_preprocessor = pipeline_best.named_steps["preprocessor"]
                    xt = fitted_preprocessor.transform(X_train)
                    fnames = [f"f{i}" for i in range(xt.shape[1])]
                imp_df = extract_feature_importance(pipeline_best.named_steps["model"], list(fnames))
                if imp_df is not None:
                    fi_plot_path = visual_fi_dir / f"{safe_name}.png"
                    plot_feature_importance(imp_df, fi_plot_path, title=f"Feature Importance: {model_label} ({preproc_name})")
                    if clearml_mgr.logger and fi_plot_path.exists():
                        try:
                            clearml_mgr.logger.report_image(
                                title=f"fi_{model_label}",
                                series="feature_importance",
                                local_path=str(fi_plot_path),
                            )
                            try:
                                import plotly.express as px  # type: ignore

                                fig = px.bar(imp_df, x="feature", y="importance", title=f"Feature Importance: {model_label}")
                                clearml_mgr.logger.report_plotly(
                                    title=f"fi_{model_label}",
                                    series="feature_importance",
                                    iteration=0,
                                    figure=fig,
                                )
                            except Exception:
                                pass
                        except Exception:
                            pass
            except Exception:
                pass
        # SHAP summary per model
        if cfg.visualizations.shap_summary:
            try:
                shap_plot_path = visual_shap_dir / f"{safe_name}.png"
                plot_shap_summary(pipeline_best, X_train, shap_plot_path)
                if clearml_mgr.logger and shap_plot_path.exists():
                    try:
                        clearml_mgr.logger.report_image(
                            title=f"shap_{model_label}",
                            series="shap_summary",
                            local_path=str(shap_plot_path),
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        # Save predictions and metrics CSV artifacts
        try:
            preds_rows = []
            if y_pred_for_plots is not None:
                for idx, (yt, yp) in enumerate(zip(y_train_array, y_pred_for_plots)):
                    preds_rows.append({"sample": idx, "split": "train", "y_true": yt, "y_pred": yp, "residual": yt - yp})
            if y_pred_test is not None:
                for idx, (yt, yp) in enumerate(zip(y_test, y_pred_test)):
                    preds_rows.append({"sample": idx, "split": "test", "y_true": yt, "y_pred": yp, "residual": yt - yp})
            if preds_rows:
                preds_df = pd.DataFrame(preds_rows)
                preds_path = output_dir / f"predictions_{safe_name}.csv"
                preds_df.to_csv(preds_path, index=False)
            metrics_records = []
            metrics_records.append({"metric": "R2", "split": "train", "value": r2_for_plots})
            metrics_records.append({"metric": "MSE", "split": "train", "value": mse_val})
            metrics_records.append({"metric": "RMSE", "split": "train", "value": rmse_val})
            metrics_records.append({"metric": "MAE", "split": "train", "value": mae_val})
            for k, v in test_metrics.items():
                metrics_records.append({"metric": k, "split": "test", "value": v})
            metrics_df = pd.DataFrame(metrics_records)
            metrics_path = output_dir / f"metrics_{safe_name}.csv"
            metrics_df.to_csv(metrics_path, index=False)
        except Exception:
            pass
    # Record links to per-model tasks for debugging / navigation
    try:
        if clearml_mgr.logger and model_task_records:
            df_links = pd.DataFrame(model_task_records)
            clearml_mgr.report_table("DEbugsamples", df_links, series="DEbugsamples")
            clearml_mgr.report_table("model_metrics", df_links, series="metrics")
            # Also emit HTML list soリンクを直接クリック可能
            try:
                links_html = "<br/>".join([row["link_html"] for row in model_task_records if row.get("link_html")])
                if links_html:
                    clearml_mgr.logger.report_text(f"Model task links:<br/>{links_html}", title="model_task_links")
            except Exception:
                pass
    except Exception:
        pass
    # Debugsamples: plot artifact listing for summary task
    try:
        plot_rows = []
        for pf in [
            *output_dir.glob("*.png"),
            *(output_dir / "scatter_plots").glob("*.png"),
            *(output_dir / "residual_scatter").glob("*.png"),
            *(output_dir / "residual_hist").glob("*.png"),
            *(output_dir / "interpolation_space").glob("*.png"),
        ]:
            plot_rows.append({"file": pf.name, "path": str(pf)})
        if plot_rows:
            clearml_mgr.report_table("plot_artifacts", pd.DataFrame(plot_rows), series="DEbugsamples")
    except Exception:
        pass
    # Upload key artifacts to ClearML
    try:
        artifacts_to_upload = [results_path, results_path.with_suffix(".json")]
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
        # パイプライン実行中は step wrapper が戻り値(artifact/param)を書き戻すため、ここでは閉じない
        if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
            clearml_mgr.close()
    return {
        "dataset_id": dataset_id_for_load,
        "summary_task_id": clearml_mgr.task.id if clearml_mgr.task else None,
        "training_task_ids": training_task_ids,
        "metrics": model_task_records,  # 各モデルのメトリクス/リンクを比較フェーズに渡す
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
