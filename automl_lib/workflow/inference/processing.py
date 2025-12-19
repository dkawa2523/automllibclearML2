"""
Core processing for inference phase.
single/batch/optimize の推論ロジックを実行する。
"""

from typing import Any, Dict, Optional
from pathlib import Path

from automl_lib.config.loaders import load_inference_config
from automl_lib.types import InferenceInfo
from automl_lib.integrations.clearml.context import get_run_id_env, resolve_run_id, set_run_id_env
from automl_lib.workflow.inference.runner import run_inference_workflow


def run_inference_processing(
    config_path: Path,
    input_info: Optional[Dict[str, Any]] = None,
    *,
    run_id: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run inference from config (ClearML optional).
    """
    # Pipeline hand-off (training -> inference): allow upstream to provide the chosen model_id.
    upstream_model_id = (input_info or {}).get("recommended_model_id") or (input_info or {}).get("model_id")
    if isinstance(config_data, dict):
        raw: Dict[str, Any] = dict(config_data)
        if upstream_model_id:
            raw["model_id"] = str(upstream_model_id)
        from automl_lib.config.schemas import InferenceConfig

        cfg = InferenceConfig.model_validate(raw)
    else:
        cfg = load_inference_config(config_path)
    try:
        from automl_lib.integrations.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    # Allow ClearML HyperParameters edits (clone -> edit -> run) to override YAML config.
    try:
        if cfg.clearml and bool(getattr(cfg.clearml, "enabled", False)):
            from automl_lib.integrations.clearml.hyperparams import (
                apply_inference_hyperparams,
                get_current_task_hyperparams,
            )

            params = get_current_task_hyperparams(cast=True)
            if isinstance(params, dict):
                cfg_dump = cfg.model_dump()
                applied = apply_inference_hyperparams(cfg_dump, params)
                if applied != cfg_dump:
                    cfg = type(cfg).model_validate(applied)
    except Exception as exc:
        raise ValueError(f"Invalid ClearML HyperParameters override for inference: {exc}") from exc

    # Pipeline hand-off: upstream model_id always wins (keep pipeline deterministic).
    try:
        if upstream_model_id:
            cfg = type(cfg).model_validate({**cfg.model_dump(), "model_id": str(upstream_model_id)})
    except Exception:
        pass
    run_id = resolve_run_id(
        explicit=run_id,
        from_input=(input_info or {}).get("run_id"),
        from_config=getattr(cfg.run, "id", None),
        from_env=get_run_id_env(),
    )
    set_run_id_env(run_id)
    payload = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()  # type: ignore[attr-defined]
    try:
        payload.setdefault("run", {})
        payload["run"]["id"] = run_id
    except Exception:
        pass
    result = run_inference_workflow(payload, input_info=input_info) or {}
    if not isinstance(result, dict):
        result = {}
    result.setdefault("run_id", run_id)
    info = InferenceInfo(**result)
    return info.model_dump()
