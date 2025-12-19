"""Compatibility wrapper for the AutoML training runner.

The orchestration entrypoint lives under `automl_lib.workflow.training.runner`.
The `automl_lib.training.*` package is reserved for core ML logic (models,
evaluation, search, plots) that the workflow code composes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from automl_lib.workflow.training.runner import run_automl as _run_automl


def run_automl(
    config_path: Path,
    *,
    dataset_id: str | None = None,
    parent_task_id: str | None = None,
) -> Dict[str, Any]:
    return _run_automl(config_path, dataset_id=dataset_id, parent_task_id=parent_task_id)


__all__ = ["run_automl"]
