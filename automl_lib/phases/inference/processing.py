"""
Core processing for inference phase.
単一推論/範囲探索/最適化のロジックを担当する予定のスタブ。
"""

from typing import Any, Dict, Optional
from pathlib import Path

from automl_lib.inference.run import _run_from_config as run_inference_config
from automl_lib.config.loaders import load_inference_config
from automl_lib.types import InferenceInfo
from automl_lib.clearml.context import get_run_id_env, resolve_run_id, set_run_id_env


def run_inference_processing(
    config_path: Path,
    input_info: Optional[Dict[str, Any]] = None,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bridge to legacy inference implementation.
    """
    cfg = load_inference_config(config_path)
    try:
        from automl_lib.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
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
    result = run_inference_config(payload) or {}
    if not isinstance(result, dict):
        result = {}
    result.setdefault("run_id", run_id)
    info = InferenceInfo(**result)
    return info.model_dump()
