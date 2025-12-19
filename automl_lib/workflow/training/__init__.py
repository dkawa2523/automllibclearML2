"""
training workflow package
-------------------------
処理の入口（processing）を提供する。

NOTE:
import 時に重い依存関係（sklearn/clearml等）を読み込まない & 循環importを避けるため、
ここでは lazy import の薄い関数だけを公開する。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = ["run_training_processing"]


def run_training_processing(
    config_path,
    input_info: Optional[Dict[str, Any]] = None,
    parent_task_id: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    from .processing import run_training_processing as _run

    return _run(
        config_path,
        input_info=input_info,
        parent_task_id=parent_task_id,
        run_id=run_id,
    )
