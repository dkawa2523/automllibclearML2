"""Model evaluation helpers for inference (grid search and Optuna optimisation)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from .model_utils import (
    _align_input_columns,
    _log_if_almost_constant,
    _predict_with_model,
)

try:
    import optuna  # type: ignore
    from optuna.samplers import TPESampler, CmaEsSampler
except Exception:  # pragma: no cover - optional dependency
    optuna = None


def run_grid_search(models: Dict[str, Any], combos: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Evaluate all models on each parameter combination."""

    if not combos:
        raise ValueError("No combinations were generated for grid search")
    df_inputs = pd.DataFrame(combos)
    df_inputs = _align_input_columns(df_inputs, models)
    results_all = df_inputs.copy()
    results_by_model: Dict[str, pd.DataFrame] = {}
    for model_label, pipeline in models.items():
        try:
            preds = _predict_with_model(pipeline, df_inputs)
        except Exception as exc:
            warnings.warn(f"Prediction failed for model {model_label}: {exc}")
            preds = np.full(len(df_inputs), np.nan)
        else:
            _log_if_almost_constant(model_label, preds)
        results_all[model_label] = preds
        df_model = df_inputs.copy()
        df_model["prediction"] = preds
        results_by_model[model_label] = df_model
    return results_all, results_by_model


def run_optimization(
    models: Dict[str, Any],
    vars_list: List[Dict[str, Any]],
    method: str,
    n_trials: int,
    goal: str,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run optimisation using Optuna for each model."""

    if optuna is None:
        raise ImportError("Optuna is required for optimization; please install optuna")

    results_by_model: Dict[str, pd.DataFrame] = {}
    combo_records: List[Dict[str, Any]] = []
    combo_keys: set[Tuple[Tuple[str, Any], ...]] = set()
    var_order = [var["name"] for var in vars_list]

    for model_label, pipeline in models.items():
        def objective(trial: optuna.trial.Trial) -> float:
            sample: Dict[str, Any] = {}
            for var in vars_list:
                name = var["name"]
                vtype = var["type"]
                if var["method"] == "range":
                    low, high = var["min"], var["max"]
                    if vtype == "int":
                        sample[name] = trial.suggest_int(name, int(low), int(high))
                    elif vtype == "float":
                        step = var.get("step")
                        if step:
                            sample[name] = trial.suggest_float(name, float(low), float(high), step=float(step))
                        else:
                            sample[name] = trial.suggest_float(name, float(low), float(high))
                    else:
                        raise ValueError("Range method not supported for categorical variable")
                else:
                    values = var["values"]
                    sample[name] = trial.suggest_categorical(name, values)
            df_sample = pd.DataFrame([sample])
            try:
                pred_arr = _predict_with_model(pipeline, df_sample)
                val = float(pred_arr[0])
            except Exception:
                val = float("nan")
            trial.set_user_attr("sample", sample)
            trial.set_user_attr("prediction", val)
            return val

        sampler: Optional[optuna.samplers.BaseSampler] = None
        method_lower = method.lower()
        if method_lower == "random":
            sampler = optuna.samplers.RandomSampler(seed=0)
        elif method_lower == "tpe":
            sampler = TPESampler(seed=0)
        elif method_lower == "cmaes":
            if any(var["type"] == "categorical" or var["method"] == "values" for var in vars_list):
                raise ValueError("CMA-ES sampler does not support categorical variables; use random or TPE")
            try:
                import cmaes  # type: ignore  # noqa: F401
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ImportError(
                    "CMA-ES sampler requires the optional 'cmaes' package. Install it with 'pip install cmaes' or "
                    "choose a different search method."
                ) from exc
            sampler = CmaEsSampler(seed=0)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        study = optuna.create_study(direction="maximize" if goal == "max" else "minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        rows = []
        for t in study.trials:
            sample = t.user_attrs.get("sample", {})
            pred = t.user_attrs.get("prediction", np.nan)
            row = sample.copy()
            row["prediction"] = pred
            rows.append(row)
            key = tuple((name, sample.get(name)) for name in var_order)
            if key not in combo_keys:
                combo_keys.add(key)
                combo_records.append({name: sample.get(name) for name in var_order})
        df_model = pd.DataFrame(rows)
        if not df_model.empty:
            df_model.insert(0, "trial_index", np.arange(len(df_model)))
        results_by_model[model_label] = df_model

    if combo_records:
        aggregated_inputs = pd.DataFrame(combo_records)
        aggregated_inputs = aggregated_inputs.reindex(columns=var_order)
    else:
        aggregated_inputs = pd.DataFrame(columns=var_order)
    aggregated_inputs = _align_input_columns(aggregated_inputs, models)
    aggregated = aggregated_inputs.copy()
    for model_label, pipeline in models.items():
        if aggregated_inputs.empty:
            aggregated[model_label] = pd.Series(dtype=float)
        else:
            preds = _predict_with_model(pipeline, aggregated_inputs)
            _log_if_almost_constant(model_label, preds)
            aggregated[model_label] = preds
    results_all = aggregated
    return results_all, results_by_model


__all__ = [
    "run_grid_search",
    "run_optimization",
]
