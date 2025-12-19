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
import json
import os
import time
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
from automl_lib.integrations.clearml import ensure_local_dataset_copy, find_first_csv, load_input_model
from automl_lib.integrations.clearml.context import (
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    run_scoped_output_dir,
    set_run_id_env,
)
from automl_lib.config.loaders import load_yaml
from automl_lib.inference.model_utils import _get_required_feature_names, _predict_with_model

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
    with pd.option_context("display.max_columns", None):
        print("Aggregated results (first 5 rows):")
        print(aggregated.head())
        if not corr_matrix.empty:
            print("\nCorrelation matrix:")
            print(corr_matrix)
        if not agreement_matrix.empty:
            print("\nAgreement matrix:")
            print(agreement_matrix)


def _run_from_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    run_cfg = config_data.get("run") if isinstance(config_data.get("run"), dict) else {}
    run_id = resolve_run_id(from_config=(run_cfg or {}).get("id"), from_env=get_run_id_env())
    set_run_id_env(run_id)

    input_conf = config_data.get("input")
    if not isinstance(input_conf, dict):
        raise ValueError("'input' section missing or invalid in configuration")
    mode = str(input_conf.get("mode") or "single").strip().lower()
    if mode == "csv":
        mode = "batch"
    if mode == "params":
        mode = "optimize"
    if mode not in {"single", "batch", "optimize"}:
        raise ValueError("input.mode must be one of: single, batch, optimize")

    output_base = Path(str(config_data.get("output_dir") or "outputs/inference"))
    output_dir = run_scoped_output_dir(output_base, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_key = resolve_dataset_key(
        explicit=(run_cfg or {}).get("dataset_key"),
        dataset_id=str(input_conf.get("dataset_id")) if input_conf.get("dataset_id") else None,
        csv_path=(str(input_conf.get("csv_path")) if input_conf.get("csv_path") else None)
        or (str(input_conf.get("single_json")) if input_conf.get("single_json") else None),
    )

    # ------------------------------------------------------------------
    # Resolve a single model (recommended usage: config.model_id)
    # ------------------------------------------------------------------
    model_id = config_data.get("model_id")
    model_name = config_data.get("model_name")
    model_path = config_data.get("model_path")
    model_dir = config_data.get("model_dir")
    legacy_models = config_data.get("models")

    def _load_joblib(path: Path):
        import joblib  # type: ignore

        return joblib.load(path)

    model_local_path: Optional[Path] = None
    if model_id:
        model_local_path = load_input_model(str(model_id))
        if not model_local_path:
            raise ValueError(f"Failed to load ClearML model_id={model_id}")
    elif model_path:
        model_local_path = Path(str(model_path))
        if not model_local_path.exists():
            raise ValueError(f"model_path does not exist: {model_local_path}")
    else:
        enabled_specs: List[Dict[str, Any]] = []
        if isinstance(legacy_models, list):
            for item in legacy_models:
                if not isinstance(item, dict):
                    continue
                enabled = item.get("enable", item.get("enabled", True))
                enabled_bool = enabled.strip().lower() not in {"false", "0", "no", "off"} if isinstance(enabled, str) else bool(enabled)
                if enabled_bool:
                    enabled_specs.append(item)
        if len(enabled_specs) != 1:
            raise ValueError("Provide inference.model_id (recommended) or enable exactly one entry in models[]")
        spec = enabled_specs[0]
        spec_id = spec.get("model_id")
        if spec_id:
            model_id = str(spec_id)
            model_local_path = load_input_model(model_id)
            if not model_local_path:
                raise ValueError(f"Failed to load ClearML model_id={model_id}")
        else:
            if not model_dir:
                raise ValueError("model_dir is required when loading an enabled models[] entry without model_id")
            selected_name = str(spec.get("name") or "").strip()
            if not selected_name:
                raise ValueError("models[].name is required when loading from model_dir")
            loaded = load_models(str(model_dir), selected_names=[selected_name])
            if not loaded:
                raise ValueError(f"Failed to load model '{selected_name}' from model_dir={model_dir}")
            # Take the first match (filtered)
            first_key = next(iter(loaded.keys()))
            pipeline = loaded[first_key]
            model_label = model_name or selected_name
            model_meta = {
                "model_source": "local_dir",
                "model_dir": str(model_dir),
                "model_name": model_label,
            }
            return _run_single_inference(
                mode=mode,
                pipeline=pipeline,
                model_label=model_label,
                model_id=None,
                model_meta=model_meta,
                run_id=run_id,
                dataset_key=dataset_key,
                input_conf=input_conf,
                search_conf=config_data.get("search") if isinstance(config_data.get("search"), dict) else {},
                output_dir=output_dir,
                full_config=config_data,
            )

    pipeline = _load_joblib(Path(model_local_path))
    model_label = str(model_name or "model").strip() or "model"
    model_meta: Dict[str, Any] = {
        "model_source": ("clearml_model" if model_id else "local_path"),
        "model_id": str(model_id or ""),
        "model_path": str(model_local_path),
        "model_name": model_label,
        "pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__name__}",
        "required_feature_names": _get_required_feature_names(pipeline) or [],
    }

    return _run_single_inference(
        mode=mode,
        pipeline=pipeline,
        model_label=model_label,
        model_id=str(model_id) if model_id else None,
        model_meta=model_meta,
        run_id=run_id,
        dataset_key=dataset_key,
        input_conf=input_conf,
        search_conf=config_data.get("search") if isinstance(config_data.get("search"), dict) else {},
        output_dir=output_dir,
        full_config=config_data,
    )


def run_inference_core(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference without any ClearML Task side-effects.

    Intended to be called from `automl_lib.workflow.inference.*` which owns ClearML logging.
    Returns a dict that may contain non-JSON-serialisable objects (e.g., Path, DataFrame).
    """

    return _run_from_config(config_data)


def _run_single_inference(
    *,
    mode: str,
    pipeline: Any,
    model_label: str,
    model_id: Optional[str],
    model_meta: Dict[str, Any],
    run_id: str,
    dataset_key: str,
    input_conf: Dict[str, Any],
    search_conf: Dict[str, Any],
    output_dir: Path,
    full_config: Dict[str, Any],
) -> Dict[str, Any]:
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _resolve_single_input() -> tuple[Dict[str, Any], Optional[str]]:
        if input_conf.get("single") is not None:
            if not isinstance(input_conf["single"], dict):
                raise ValueError("input.single must be a JSON object")
            return dict(input_conf["single"]), "config:inline"
        json_path = input_conf.get("single_json")
        if json_path:
            p = Path(str(json_path))
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                if not raw or not isinstance(raw[0], dict):
                    raise ValueError("input.single_json must contain an object or a non-empty list of objects")
                return dict(raw[0]), str(p)
            if isinstance(raw, dict):
                return dict(raw), str(p)
            raise ValueError("input.single_json must contain an object or a list of objects")
        raise ValueError("single mode requires input.single or input.single_json")

    def _find_inference_input_csv(local_copy: Path) -> Optional[Path]:
        processed_names = {"data_processed.csv", "preprocessed_features.csv"}
        try:
            candidates = [p for p in Path(local_copy).rglob("*.csv") if p.name not in processed_names]
            if candidates:
                candidates.sort(key=lambda p: (len(p.parts), str(p)))
                return candidates[0]
        except Exception:
            pass
        return find_first_csv(Path(local_copy))

    def _resolve_batch_input() -> tuple[pd.DataFrame, str]:
        csv_path = input_conf.get("csv_path")
        if csv_path:
            p = Path(str(csv_path))
            df = pd.read_csv(p)
            return df, str(p)
        dataset_id = input_conf.get("dataset_id")
        if dataset_id:
            dsid = str(dataset_id)
            local_copy = ensure_local_dataset_copy(dsid, output_dir / "clearml_dataset")
            if not local_copy:
                raise ValueError(f"Failed to download ClearML dataset_id={dsid}")
            csv = _find_inference_input_csv(local_copy)
            # If only the processed table is present, try the parent dataset (raw) from manifest.json.
            try:
                if csv and csv.name in {"data_processed.csv", "preprocessed_features.csv"}:
                    manifest_path = Path(local_copy) / "manifest.json"
                    if manifest_path.exists():
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        parent_id = manifest.get("parent_dataset_id") if isinstance(manifest, dict) else None
                        if parent_id:
                            parent_copy = ensure_local_dataset_copy(
                                str(parent_id), output_dir / "clearml_dataset_parent"
                            )
                            if parent_copy:
                                csv_parent = _find_inference_input_csv(parent_copy)
                                if csv_parent:
                                    csv = csv_parent
            except Exception:
                pass
            if not csv:
                raise ValueError(f"No CSV found for dataset_id={dsid}")
            df = pd.read_csv(csv)
            return df, str(csv)
        raise ValueError("batch mode requires input.csv_path or input.dataset_id")

    artifacts: List[Path] = []
    # Always persist a reproducible config snapshot.
    cfg_path = output_dir / "inference_config.json"
    try:
        _write_json(cfg_path, full_config)
        artifacts.append(cfg_path)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Execute inference
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    user_props: Dict[str, Any] = {
        "run_id": str(run_id),
        "dataset_key": str(dataset_key),
        "mode": mode,
        "model_id": model_id or "",
        "model_name": model_label,
    }
    prediction_value: Optional[float] = None
    preview_df: Optional[pd.DataFrame] = None
    trials_preview_df: Optional[pd.DataFrame] = None

    if mode == "single":
        row, src = _resolve_single_input()
        df = pd.DataFrame([row])
        pred_arr = _predict_with_model(pipeline, df)
        pred = pred_arr[0] if len(pred_arr) else None
        payload_in = {"source": src or "", "row": row}
        prediction_value = float(pred) if pred is not None else None
        payload_out = {"prediction": prediction_value}
        in_path = output_dir / "input.json"
        out_path = output_dir / "output.json"
        _write_json(in_path, payload_in)
        _write_json(out_path, payload_out)
        artifacts.extend([in_path, out_path])
        user_props["prediction"] = payload_out["prediction"]
    elif mode == "batch":
        df_in, src = _resolve_batch_input()
        if df_in.empty:
            raise ValueError("Batch input contains no rows")
        preds = _predict_with_model(pipeline, df_in)
        df_out = df_in.copy()
        df_out["prediction"] = preds
        out_csv = output_dir / "predictions.csv"
        df_out.to_csv(out_csv, index=False)
        artifacts.append(out_csv)
        user_props["n_rows"] = int(len(df_out))
        try:
            s = pd.to_numeric(df_out["prediction"], errors="coerce")
            user_props["prediction_min"] = float(s.min()) if s.notna().any() else ""
            user_props["prediction_max"] = float(s.max()) if s.notna().any() else ""
        except Exception:
            pass
        try:
            preview_df = df_out.head(50)
        except Exception:
            preview_df = None
        try:
            in_meta = {"source": src, "n_rows": int(len(df_in)), "columns": list(df_in.columns)}
            in_meta_path = output_dir / "input_meta.json"
            _write_json(in_meta_path, in_meta)
            artifacts.append(in_meta_path)
        except Exception:
            pass
    elif mode == "optimize":
        vars_spec = None
        if input_conf.get("variables"):
            vars_spec = parse_variables_from_config(input_conf["variables"])
        elif input_conf.get("params_path"):
            vars_spec = parse_param_spec(input_conf["params_path"])
        else:
            raise ValueError("Optimize mode requires input.variables or input.params_path")
        method = str(search_conf.get("method", "grid")).strip().lower()
        n_trials = int(search_conf.get("n_trials", 20))
        goal = str(search_conf.get("goal", "max")).strip().lower()
        models = {model_label: pipeline}
        if method == "grid":
            combos = enumerate_parameter_combinations(vars_spec)
            aggregated, per_model = run_grid_search(models, combos)
        else:
            aggregated, per_model = run_optimization(models=models, vars_list=vars_spec, method=method, n_trials=n_trials, goal=goal)
        # Single-model output
        df_trials = None
        try:
            df_trials = per_model.get(model_label) if isinstance(per_model, dict) else None
            if df_trials is None and isinstance(per_model, dict) and per_model:
                df_trials = per_model[next(iter(per_model.keys()))]
        except Exception:
            df_trials = None
        if df_trials is None or not isinstance(df_trials, pd.DataFrame):
            raise RuntimeError("Failed to generate optimization trials")
        trials_path = output_dir / "trials.csv"
        df_trials.to_csv(trials_path, index=False)
        artifacts.append(trials_path)
        try:
            trials_preview_df = df_trials.head(50)
        except Exception:
            trials_preview_df = None
        best_payload: Dict[str, Any] = {
            "goal": goal,
            "method": method,
            "n_trials": n_trials,
            "best_prediction": None,
            "best_input": {},
        }
        try:
            if "prediction" in df_trials.columns and not df_trials.empty:
                s = pd.to_numeric(df_trials["prediction"], errors="coerce")
                if s.notna().any():
                    idx = int(s.idxmin()) if goal == "min" else int(s.idxmax())
                    row = df_trials.loc[idx].to_dict()
                    best_payload["best_prediction"] = float(pd.to_numeric(row.get("prediction"), errors="coerce"))
                    best_payload["best_input"] = {k: v for k, v in row.items() if k not in {"prediction", "trial_index"}}
                    user_props["best_prediction"] = best_payload["best_prediction"]
        except Exception:
            pass
        best_path = output_dir / "best_solution.json"
        _write_json(best_path, best_payload)
        artifacts.append(best_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    user_props["total_seconds"] = float(time.perf_counter() - t0)
    return {
        "mode": str(mode),
        "output_dir": output_dir,
        "artifacts": [p for p in artifacts if p.exists()],
        "user_props": user_props,
        "model_meta": model_meta,
        "prediction": prediction_value,
        "preview_df": preview_df,
        "trials_preview_df": trials_preview_df,
        "model_id": model_id,
        "model_name": model_label,
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

    _finalize_run(models, aggregated, per_model_results, args.output_dir, prefix, goal_for_plot)


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

    config_path = _resolve_config_path(args.config)
    if config_path is not None:
        # Prefer the workflow entrypoint so ClearML logging stays centralized under workflow/.
        from automl_lib.workflow.inference.processing import run_inference_processing

        run_inference_processing(Path(config_path))
    else:
        _run_from_cli(args)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Programmatic entry point for pipeline/phase execution
# ---------------------------------------------------------------------------
def run_inference_from_config(config_path: Path) -> None:
    """Run inference using a YAML config (used by pipeline phases)."""

    from automl_lib.workflow.inference.processing import run_inference_processing

    run_inference_processing(Path(config_path))
