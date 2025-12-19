"""
training package
----------------
Backward-compatible entrypoints for training execution.

NOTE:
Avoid eager imports to prevent circular imports between `automl_lib.training`
and `automl_lib.workflow.training`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["run_automl"]


def run_automl(
    config_path: Path,
    *,
    dataset_id: Optional[str] = None,
    parent_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    from .run import run_automl as _run

    return _run(config_path, dataset_id=dataset_id, parent_task_id=parent_task_id)
