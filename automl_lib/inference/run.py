"""Inference script for AutoML models.

This script loads trained model pipelines (preprocessor + estimator) saved by
the AutoML training pipeline and performs predictions on new data. It
supports two input modes:

1. CSV mode: load a CSV file of feature values and predict the target for
   each row.
2. Parameter search mode: define ranges or lists of values for each input
   variable and either enumerate all combinations (grid search) or use
   Optuna to sample candidate combinations via random, TPE or CMA-ES
   samplers. The objective is to maximize or minimize the predicted
   target.

For each model, predictions are saved to a per-model CSV file. An
aggregated CSV collecting predictions from all selected models is also
generated. Additionally, plots are produced to visualize predictions and
model comparisons. Correlation and agreement metrics between models are
computed and visualized to help assess consistency.

Example usage:

    python inference.py \
        --model-dir outputs/train/models \
        --models Ridge,RandomForest \
        --input-csv new_data.csv \
        --output-dir outputs/inference

    python inference.py \
        --model-dir outputs/train/models \
        --input-params params.json \
        --search-method tpe \
        --n-trials 50 \
        --goal max \
        --output-dir outputs/inference

    python inference.py --config inference_config.yaml

The JSON file for parameter search should have the following structure:

    {
        "variables": [
            {"name": "age", "type": "int", "method": "range", "min": 20, "max": 60, "step": 10},
            {"name": "gender", "type": "categorical", "method": "values", "values": ["M", "F"]},
            {"name": "income", "type": "float", "method": "values", "values": [50000.0, 75000.0, 100000.0]}
        ]
    }

Notes
-----
* The CMA-ES sampler in Optuna only supports continuous (float) and integer
  distributions. If categorical variables are present, CMA-ES search will
  raise an error. Use random or TPE search instead.
* For classification models, predictions are taken from the probability of
  the positive class if available. If the classifier does not support
  probabilities, predicted class labels are used and correlation analysis is
  skipped.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from automl_lib.inference import (
    compute_consistency_scores,
    enumerate_parameter_combinations,
    load_models,
    parse_param_spec,
    parse_variables_from_config,
    plot_agreement_heatmap,
    plot_correlation_heatmap,
    plot_predictions,
    run_grid_search,
    run_optimization,
    save_matrices_and_scores,
    save_results,
)
from automl_lib.clearml import ensure_local_dataset_copy, find_first_csv, init_task, load_input_model
from automl_lib.config.loaders import load_yaml
from automl_lib.training.clearml_integration import ClearMLManager, build_clearml_config_from_dict
try:
    from clearml import Task, TaskTypes, InputModel  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Task = None
    TaskTypes = None
    InputModel = None

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _resolve_config_path(config_arg: Optional[str]) -> Optional[Path]:
    """Return a path to the YAML config if supplied or if a default exists."""

    if config_arg:
        return Path(config_arg)
    default_config = Path("inference_config.yaml")
    return default_config if default_config.is_file() else None


def _load_config(config_arg: Optional[str]) -> Optional[Dict[str, Any]]:
    path = _resolve_config_path(config_arg)
    if path is None:
        return None
    return load_yaml(path)


def _parse_model_names(models_section: Any) -> Optional[List[str]]:
    if not models_section:
        return None
    if isinstance(models_section, str):
        selected = [m.strip() for m in models_section.split(",") if m.strip()]
        if not selected:
            raise ValueError("At least one model name must be specified in 'models'")
        return selected
    if isinstance(models_section, list):
        parsed: List[str] = []
        for item in models_section:
            if isinstance(item, str):
                name = item.strip()
                if name:
                    parsed.append(name)
                continue
            if isinstance(item, dict):
                name = item.get("name")
                if not name:
                    raise ValueError("Each model entry must include a 'name'")
                enabled_value = item.get("enable", item.get("enabled", True))
                if isinstance(enabled_value, str):
                    enabled = enabled_value.strip().lower() not in {"false", "0", "no", "off"}
                else:
                    enabled = bool(enabled_value)
                if enabled:
                    parsed.append(str(name).strip())
                continue
            raise ValueError("'models' list entries must be strings or dictionaries with 'name'/'enable'")
        if not parsed:
            raise ValueError("At least one model must be enabled in the configuration")
        return parsed
    raise ValueError("'models' in configuration must be a list or comma-separated string")


def _finalize_run(
    models: Dict[str, Any],
    aggregated: pd.DataFrame,
    per_model_results: Dict[str, pd.DataFrame],
    output_dir: str,
    prefix: str,
    goal_for_plot: Optional[str],
    clearml_mgr: Optional[ClearMLManager] = None,
) -> None:
    save_results(aggregated, per_model_results, output_dir, prefix)
    print(f"Saved aggregated and per-model results to directory '{output_dir}'.")
    plot_predictions(
        aggregated,
        models.keys(),
        output_dir,
        per_model_results=per_model_results,
        goal=goal_for_plot,
    )
    corr_matrix, agreement_matrix, mean_corr, mean_agreement = compute_consistency_scores(
        aggregated, models.keys()
    )
    save_matrices_and_scores(corr_matrix, agreement_matrix, mean_corr, mean_agreement, output_dir, prefix)
    try:
        plot_correlation_heatmap(aggregated, models.keys(), output_dir)
    except Exception as exc:
        warnings.warn(f"Correlation heatmap could not be generated: {exc}")
    try:
        plot_agreement_heatmap(aggregated, models.keys(), output_dir)
    except Exception as exc:
        warnings.warn(f"Agreement heatmap could not be generated: {exc}")
    if clearml_mgr:
        try:
            clearml_mgr.report_table("inference_aggregated", aggregated, series="predictions")
            if goal_for_plot:
                direction = str(goal_for_plot).lower()
                for model_label in models.keys():
                    if model_label in aggregated:
                        values = aggregated[model_label]
                        best_val = values.min() if direction == "min" else values.max()
                        if pd.notna(best_val):
                            clearml_mgr.report_scalar("inference_best", model_label, float(best_val))
            artifact_paths = [Path(output_dir) / f"{prefix}_aggregated.csv"]
            for model_label in per_model_results.keys():
                model_safe = model_label.replace(" ", "_")
                artifact_paths.append(Path(output_dir) / f"{prefix}_{model_safe}.csv")
            artifact_paths.extend([
                Path(output_dir) / f"{prefix}_correlation_matrix.csv",
                Path(output_dir) / f"{prefix}_agreement_matrix.csv",
                Path(output_dir) / f"{prefix}_mean_correlation.csv",
                Path(output_dir) / f"{prefix}_mean_agreement.csv",
            ])
            clearml_mgr.upload_artifacts([p for p in artifact_paths if p.exists()])
        except Exception:
            pass
    with pd.option_context("display.max_columns", None):
        print("Aggregated results (first 5 rows):")
        print(aggregated.head())
        if not corr_matrix.empty:
            print("\nCorrelation matrix:")
            print(corr_matrix)
        if not agreement_matrix.empty:
            print("\nAgreement matrix:")
            print(agreement_matrix)


def _load_models_maybe_clearml(model_dir: Optional[str], models_cfg: Any, clearml_cfg) -> Dict[str, Any]:
    """Load models either from local joblib directory or ClearML InputModel when model_id is provided."""

    selected_models = _parse_model_names(models_cfg)
    # If clearml model ids are provided, prefer InputModel
    clearml_models_cfg = []
    if isinstance(models_cfg, list):
        for item in models_cfg:
            if isinstance(item, dict) and item.get("model_id"):
                clearml_models_cfg.append(item)
    if clearml_models_cfg and InputModel is not None and clearml_cfg and clearml_cfg.enabled:
        models: Dict[str, Any] = {}
        for item in clearml_models_cfg:
            name = str(item.get("name"))
            if not name:
                continue
            if selected_models and name not in selected_models:
                continue
            model_id = str(item.get("model_id"))
            try:
                local_path = load_input_model(model_id)
                if local_path:
                    import joblib  # type: ignore

                    models[name] = joblib.load(local_path)
                    print(f"Loaded model '{name}' from ClearML InputModel id={model_id}")
            except Exception as exc:
                print(f"Warning: failed to load InputModel {model_id} ({name}): {exc}")
        if models:
            return models
    # fallback to local joblib
    return load_models(model_dir, selected_names=selected_models)


def _create_child_task(clearml_cfg, parent_task_id: Optional[str], name: str) -> Optional[ClearMLManager]:
    """Create a child inference task if ClearML is enabled."""

    if not (clearml_cfg and clearml_cfg.enabled and Task and TaskTypes):
        return None
    try:
        task_obj = Task.create(
            project_name=clearml_cfg.project_name or "AutoML",
            task_name=name,
            task_type=getattr(TaskTypes, "inference", None),
        )
        if parent_task_id:
            try:
                task_obj.add_parent(parent_task_id)
            except Exception:
                pass
        return ClearMLManager(
            clearml_cfg,
            task_name=name,
            task_type="inference",
            default_project=clearml_cfg.project_name or "AutoML",
            parent=None,
            existing_task=task_obj,
        )
    except Exception as exc:
        print(f"Warning: failed to create child inference task {name}: {exc}")
        return None


def _run_from_config(config_data: Dict[str, Any]) -> None:
    clearml_cfg_raw = config_data.get("clearml") or {}
    clearml_cfg = build_clearml_config_from_dict(clearml_cfg_raw if isinstance(clearml_cfg_raw, dict) else {})

    model_dir = config_data.get("model_dir")
    if not model_dir:
        raise ValueError("'model_dir' must be specified in the configuration")
    models = _load_models_maybe_clearml(model_dir, config_data.get("models"), clearml_cfg)
    if not models:
        print(
            f"No models were loaded from directory '{model_dir}'. Please check that the directory exists and contains "
            "joblib files, and that the model names in the configuration match the saved models."
        )
        return
    print(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")

    output_dir = config_data.get("output_dir", "outputs/inference")
    os.makedirs(output_dir, exist_ok=True)
    task_name = clearml_cfg.task_name if clearml_cfg else None
    summary_task_name = task_name or f"inference-summary-{config_data.get('input', {}).get('mode', 'unknown')}"

    parent_task = None
    clearml_mgr = None
    if clearml_cfg and clearml_cfg.enabled:
        parent_id = os.environ.get("AUTO_ML_PARENT_TASK_ID")
        parent_task = init_task(
            project=clearml_cfg.project_name or "AutoML",
            name=summary_task_name,
            task_type="inference",
            queue=clearml_cfg.queue,
            parent=parent_id,
            tags=clearml_cfg.tags,
            reuse=False,
        )
        clearml_mgr = ClearMLManager(
            clearml_cfg,
            task_name=summary_task_name,
            task_type="inference",
            default_project=clearml_cfg.project_name or "AutoML",
            parent=None,
            existing_task=parent_task,
        )
        clearml_mgr.connect_configuration(config_data)
        parent_task_id = clearml_mgr.task.id if clearml_mgr.task else None
    else:
        parent_task_id = None

    input_conf = config_data.get("input")
    if not input_conf or not isinstance(input_conf, dict):
        raise ValueError("'input' section missing or invalid in configuration")
    mode = input_conf.get("mode")
    if mode is None:
        raise ValueError("'mode' must be specified in 'input' section (csv or params)")
    mode_lower = str(mode).lower()

    aggregated: pd.DataFrame
    per_model_results: Dict[str, pd.DataFrame]
    goal_for_plot: Optional[str] = None
    prefix = "results"
    child_mgr: Optional[ClearMLManager] = None

    if mode_lower == "csv":
        csv_path = input_conf.get("csv_path")
        if not csv_path and input_conf.get("dataset_id"):
            dataset_id = str(input_conf["dataset_id"])
            local_copy = ensure_local_dataset_copy(dataset_id, Path(output_dir) / "clearml_dataset")
            candidate_csv = find_first_csv(local_copy) if local_copy else None
            if candidate_csv:
                csv_path = str(candidate_csv)
        if not csv_path:
            raise ValueError("'csv_path' must be provided for CSV input mode")
        df_input = pd.read_csv(csv_path)
        if df_input.empty:
            raise ValueError("Input CSV contains no data")
        if clearml_mgr:
            clearml_mgr.log_dataset_overview(df_input, "inference_input", source=csv_path)
        print(f"Running predictions on CSV input '{csv_path}' for {len(models)} model(s)...")
        child_mgr = _create_child_task(clearml_cfg, parent_task_id, "inference-single")
        aggregated, per_model_results = run_grid_search(models, df_input.to_dict(orient="records"))
        prefix = "csv"
    elif mode_lower == "params":
        vars_spec = None
        if input_conf.get("variables"):
            vars_spec = parse_variables_from_config(input_conf["variables"])
        elif input_conf.get("params_path"):
            vars_spec = parse_param_spec(input_conf["params_path"])
        else:
            raise ValueError("For 'params' mode, specify either 'variables' list or 'params_path'")
        search_conf = config_data.get("search", {})
        method = str(search_conf.get("method", "grid")).lower()
        n_trials = int(search_conf.get("n_trials", 20))
        goal = str(search_conf.get("goal", "max")).lower()
        goal_for_plot = goal
        if method == "grid":
            combos = enumerate_parameter_combinations(vars_spec)
            print(f"Performing grid search over {len(combos)} parameter combinations for {len(models)} model(s)...")
            child_mgr = _create_child_task(clearml_cfg, parent_task_id, "inference-grid")
            aggregated, per_model_results = run_grid_search(models, combos)
            prefix = "grid"
        else:
            print(
                f"Performing Optuna '{method}' optimization with {n_trials} trial(s) for {len(models)} model(s)..."
            )
            child_mgr = _create_child_task(clearml_cfg, parent_task_id, f"inference-{method}")
            aggregated, per_model_results = run_optimization(
                models=models,
                vars_list=vars_spec,
                method=method,
                n_trials=n_trials,
                goal=goal,
            )
            prefix = method
    else:
        raise ValueError("Unknown input mode: choose 'csv' or 'params'")

    _finalize_run(models, aggregated, per_model_results, output_dir, prefix, goal_for_plot, clearml_mgr=clearml_mgr)
    if child_mgr:
        _finalize_run(models, aggregated, per_model_results, output_dir, prefix, goal_for_plot, clearml_mgr=child_mgr)
        child_mgr.close()
    if clearml_mgr:
        clearml_mgr.close()

    artifacts: List[str] = []
    output_path = Path(output_dir)
    candidates: List[Path] = [
        output_path / f"{prefix}_aggregated.csv",
        output_path / f"{prefix}_correlation_matrix.csv",
        output_path / f"{prefix}_agreement_matrix.csv",
        output_path / f"{prefix}_mean_correlation.csv",
        output_path / f"{prefix}_mean_agreement.csv",
        output_path / "predictions_plot.png",
        output_path / "correlation_heatmap.png",
        output_path / "agreement_heatmap.png",
    ]
    for model_label in per_model_results.keys():
        model_safe = str(model_label).replace(" ", "_")
        candidates.append(output_path / f"{prefix}_{model_safe}.csv")
    for cand in candidates:
        try:
            if cand.exists():
                artifacts.append(str(cand))
        except Exception:
            continue

    child_ids: List[str] = []
    try:
        if child_mgr and getattr(child_mgr, "task", None) is not None:
            child_ids.append(str(child_mgr.task.id))
    except Exception:
        pass

    task_id = None
    try:
        if clearml_mgr and getattr(clearml_mgr, "task", None) is not None:
            task_id = str(clearml_mgr.task.id)
    except Exception:
        task_id = None

    return {
        "task_id": task_id,
        "child_task_ids": child_ids,
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "mode": str(mode_lower),
    }


def _run_from_cli(args: argparse.Namespace) -> None:
    selected_models = None
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not args.model_dir:
        raise SystemExit("--model-dir is required if no configuration file is provided")
    models = load_models(args.model_dir, selected_names=selected_models)
    if not models:
        print(
            f"No models were loaded from directory '{args.model_dir}'. Please check that the directory exists and "
            "contains joblib files matching the selected models."
        )
        return
    print(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.input_csv and args.input_params:
        raise ValueError("Specify only one of --input-csv or --input-params")
    if not args.input_csv and not args.input_params:
        raise ValueError("One of --input-csv or --input-params must be provided")

    aggregated: pd.DataFrame
    per_model_results: Dict[str, pd.DataFrame]
    prefix = "results"
    goal_for_plot: Optional[str] = str(args.goal).lower() if args.goal else None

    if args.input_csv:
        df_input = pd.read_csv(args.input_csv)
        if df_input.empty:
            raise ValueError("Input CSV contains no data")
        print(f"Running predictions on CSV input '{args.input_csv}' for {len(models)} model(s)...")
        aggregated, per_model_results = run_grid_search(models, df_input.to_dict(orient="records"))
        prefix = "csv"
    else:
        vars_list = parse_param_spec(args.input_params)
        if args.search_method == "grid":
            combos = enumerate_parameter_combinations(vars_list)
            print(
                f"Performing grid search over {len(combos)} parameter combinations for {len(models)} model(s)..."
            )
            aggregated, per_model_results = run_grid_search(models, combos)
            prefix = "grid"
        else:
            print(
                f"Performing Optuna '{args.search_method}' optimization with {args.n_trials} trial(s) for "
                f"{len(models)} model(s)..."
            )
            aggregated, per_model_results = run_optimization(
                models=models,
                vars_list=vars_list,
                method=args.search_method,
                n_trials=args.n_trials,
                goal=args.goal,
            )
            prefix = args.search_method

    _finalize_run(models, aggregated, per_model_results, args.output_dir, prefix, goal_for_plot, clearml_mgr=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference script for AutoML models")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML configuration file specifying model directory, models, input, search and output settings. "
            "If omitted, the script looks for 'inference_config.yaml' in the project root. Other command-line options "
            "are ignored when a configuration file is used."
        ),
    )
    parser.add_argument("--model-dir", default=None, help="Directory containing trained joblib models")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to use (e.g. 'Ridge,RandomForest'). If omitted, all models are used.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to CSV file containing input feature data for inference.",
    )
    parser.add_argument(
        "--input-params",
        type=str,
        default=None,
        help="Path to JSON file defining parameter ranges/values for search. See documentation for structure.",
    )
    parser.add_argument(
        "--search-method",
        type=str,
        default="grid",
        choices=["grid", "random", "tpe", "cmaes"],
        help="Search method for parameter search mode. 'grid' enumerates all combinations; other methods use Optuna.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials for random/TPE/CMA-ES search. Ignored for grid search.",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="max",
        choices=["max", "min"],
        help="Objective direction for optimization: 'max' to maximize prediction, 'min' to minimize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="Directory to save output files.",
    )
    args = parser.parse_args()

    config_data = _load_config(args.config)
    if config_data is not None:
        _run_from_config(config_data)
    else:
        _run_from_cli(args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Programmatic entry point for pipeline/phase execution
# ---------------------------------------------------------------------------
def run_inference_from_config(config_path: Path) -> None:
    """Run inference using a YAML config (used by pipeline phases)."""

    config_data = _load_config(str(config_path))
    if config_data is None:
        raise FileNotFoundError(f"Inference config not found: {config_path}")
    _run_from_config(config_data)
