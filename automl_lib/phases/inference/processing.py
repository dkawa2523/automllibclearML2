"""
Core processing for inference phase.
単一推論/範囲探索/最適化のロジックを担当する予定のスタブ。
"""

from typing import Any, Dict, Optional
from pathlib import Path

from automl_lib.inference.run import _run_from_config as run_inference_config
from automl_lib.config.loaders import load_inference_config
from automl_lib.types import InferenceInfo


def run_inference_processing(config_path: Path, input_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Bridge to legacy inference implementation.
    """
    cfg = load_inference_config(config_path)
    payload = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()  # type: ignore[attr-defined]
    result = run_inference_config(payload) or {}
    if not isinstance(result, dict):
        result = {}
    info = InferenceInfo(**result)
    return info.model_dump()
