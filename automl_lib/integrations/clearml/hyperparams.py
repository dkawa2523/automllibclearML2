from __future__ import annotations

import os
from typing import Any, Dict, Optional


def get_current_task_hyperparams(*, cast: bool = True) -> Optional[Dict[str, Any]]:
    """Return current ClearML task HyperParameters as a dict (best-effort).

    Notes:
    - When running remotely/cloned/enqueued, ClearML sets CLEARML_TASK_ID and Task.current_task() is available.
    - When running locally without ClearML, this returns None.
    """

    task_id = str(os.environ.get("CLEARML_TASK_ID") or "").strip()
    if not task_id:
        return None

    try:
        from clearml import Task  # type: ignore
    except Exception:
        return None

    task = None
    try:
        task = Task.current_task()
    except Exception:
        task = None

    if task is None:
        try:
            task = Task.get_task(task_id=task_id)
        except Exception:
            task = None

    if task is None:
        return None

    try:
        params = task.get_parameters_as_dict(cast=bool(cast))
    except Exception:
        return None
    return params if isinstance(params, dict) else None


def _get_section(params: Dict[str, Any], names: list[str]) -> Optional[Dict[str, Any]]:
    for name in names:
        try:
            value = params.get(name)
        except Exception:
            value = None
        if isinstance(value, dict):
            return value
    return None


def apply_preprocessing_hyperparams(cfg_dict: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply phase-relevant HyperParameters to a PreprocessingConfig dict."""

    out = dict(cfg_dict or {})

    input_sec = _get_section(params, ["Input", "input"])
    if input_sec:
        data = dict(out.get("data") or {})
        for key in ("dataset_id", "csv_path", "target_column"):
            if key in input_sec:
                val = input_sec.get(key)
                if isinstance(val, str) and not val.strip():
                    val = None
                data[key] = val
        out["data"] = data

    preproc_sec = _get_section(params, ["Preprocessing", "preprocessing"])
    if preproc_sec:
        out["preprocessing"] = dict(preproc_sec)

    return out


def apply_training_hyperparams(cfg_dict: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply phase-relevant HyperParameters to a TrainingConfig dict."""

    out = dict(cfg_dict or {})

    # Map lightweight "Training" section to the actual TrainingConfig schema.
    training_sec = _get_section(params, ["Training", "training"])
    if training_sec:
        data = dict(out.get("data") or {})
        if "dataset_id" in training_sec:
            dsid = training_sec.get("dataset_id")
            if isinstance(dsid, str) and not dsid.strip():
                dsid = None
            data["dataset_id"] = dsid
        if "target_column" in training_sec:
            tc = training_sec.get("target_column")
            if isinstance(tc, str) and not tc.strip():
                tc = None
            data["target_column"] = tc
        if "test_size" in training_sec:
            data["test_size"] = training_sec.get("test_size")
        if "random_seed" in training_sec:
            data["random_seed"] = training_sec.get("random_seed")
        out["data"] = data

        cv = dict(out.get("cross_validation") or {})
        if "cv_shuffle" in training_sec:
            cv["shuffle"] = bool(training_sec.get("cv_shuffle"))
        if "cv_folds" in training_sec:
            folds = training_sec.get("cv_folds")
            if folds in {"auto", "", None}:
                cv["n_folds"] = None
            else:
                cv["n_folds"] = folds
        out["cross_validation"] = cv

    models_sec = _get_section(params, ["Models", "models"])
    if models_sec is not None:
        models_val = models_sec.get("models") if isinstance(models_sec, dict) else None
        if isinstance(models_val, list):
            out["models"] = models_val
        elif isinstance(models_sec, list):
            out["models"] = models_sec

    ensembles_sec = _get_section(params, ["Ensembles", "ensembles"])
    if ensembles_sec is not None:
        out["ensembles"] = dict(ensembles_sec)

    cross_sec = _get_section(params, ["CrossValidation", "cross_validation", "crossvalidation"])
    if cross_sec is not None:
        out["cross_validation"] = dict(cross_sec)

    eval_sec = _get_section(params, ["Evaluation", "evaluation"])
    if eval_sec is not None:
        out["evaluation"] = dict(eval_sec)

    opt_sec = _get_section(params, ["Optimization", "optimization"])
    if opt_sec is not None:
        out["optimization"] = dict(opt_sec)

    return out


def apply_data_registration_hyperparams(cfg_dict: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply phase-relevant HyperParameters to a DataRegistrationConfig dict."""

    out = dict(cfg_dict or {})

    input_sec = _get_section(params, ["Input", "input"])
    if input_sec:
        data = dict(out.get("data") or {})
        for key in ("csv_path", "dataset_id"):
            if key in input_sec:
                val = input_sec.get(key)
                if isinstance(val, str) and not val.strip():
                    val = None
                data[key] = val
        out["data"] = data

    return out


def apply_data_editing_hyperparams(cfg_dict: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply phase-relevant HyperParameters to a DataEditingConfig dict."""

    out = dict(cfg_dict or {})

    input_sec = _get_section(params, ["Input", "input"])
    if input_sec:
        data = dict(out.get("data") or {})
        for key in ("dataset_id", "csv_path"):
            if key in input_sec:
                val = input_sec.get(key)
                if isinstance(val, str) and not val.strip():
                    val = None
                data[key] = val
        out["data"] = data

    edit_sec = _get_section(params, ["Editing", "editing"])
    if edit_sec is not None:
        out["editing"] = dict(edit_sec)

    return out


def apply_inference_hyperparams(cfg_dict: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply phase-relevant HyperParameters to an InferenceConfig dict."""

    out = dict(cfg_dict or {})

    model_sec = _get_section(params, ["Model", "model"])
    if model_sec:
        for key in ("model_id", "model_name", "model_path", "model_dir"):
            if key in model_sec:
                val = model_sec.get(key)
                if isinstance(val, str) and not val.strip():
                    val = None
                out[key] = val

    input_sec = _get_section(params, ["Input", "input"])
    if input_sec:
        inp = dict(out.get("input") or {})
        for key in ("mode", "csv_path", "dataset_id", "single_json", "params_path"):
            if key in input_sec:
                val = input_sec.get(key)
                if isinstance(val, str) and not val.strip():
                    val = None
                inp[key] = val
        out["input"] = inp

    # Editable per-variable values for single mode.
    single_sec = _get_section(params, ["SingleInput", "single_input", "singleinput"])
    if single_sec is not None:
        inp = dict(out.get("input") or {})
        if isinstance(single_sec, dict):
            inp["single"] = dict(single_sec)
        out["input"] = inp

    # Editable variable specs for optimize mode.
    # We represent variables as a dict keyed by variable name for better UI readability/editing:
    # Variables:
    #   x1: {type: float, method: range, min: 0, max: 1, step: 0.1}
    vars_sec = _get_section(params, ["Variables", "variables", "SearchSpace", "search_space"])
    if vars_sec is not None:
        inp = dict(out.get("input") or {})
        if isinstance(vars_sec, dict):
            vars_list = []
            for name, spec in vars_sec.items():
                var_name = str(name).strip()
                if not var_name:
                    continue
                if isinstance(spec, dict):
                    row = dict(spec)
                else:
                    # Best-effort: allow a plain value to mean "fixed".
                    row = {"type": "float", "method": "fixed", "value": spec}
                row["name"] = var_name
                vars_list.append(row)
            inp["variables"] = vars_list
        elif isinstance(vars_sec, list):
            inp["variables"] = list(vars_sec)
        out["input"] = inp

    search_sec = _get_section(params, ["Search", "search"])
    if search_sec is not None:
        out["search"] = dict(search_sec)

    output_sec = _get_section(params, ["Output", "output"])
    if output_sec and "output_dir" in output_sec:
        val = output_sec.get("output_dir")
        if isinstance(val, str) and not val.strip():
            val = None
        out["output_dir"] = val

    return out
