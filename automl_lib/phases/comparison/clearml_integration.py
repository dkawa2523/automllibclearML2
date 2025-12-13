"""
ClearML integration for comparison phase.
タスク生成とArtifacts登録を担うシンプルなスタブ。
"""

import os
from typing import Dict, Any, Optional

from automl_lib.clearml import init_task, report_hyperparams, upload_artifacts


def create_comparison_task(
    config: Dict[str, Any],
    parent_task_id: Optional[str] = None,
    project_name: Optional[str] = None,
):
    clearml_cfg = config.get("clearml") or {}
    project = project_name or clearml_cfg.get("project_name") or "AutoML"
    queue = clearml_cfg.get("queue") or "services"

    if not clearml_cfg.get("enabled", False):
        return None

    # If running inside ClearML PipelineController step, reuse the step task.
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            current = Task.current_task()
            if current:
                try:
                    report_hyperparams(current, config)
                except Exception:
                    pass
                return current
        except Exception:
            pass

    task = init_task(
        project=project,
        name=clearml_cfg.get("comparison_task_name") or "comparison",
        task_type="analysis",
        queue=queue,
        parent=parent_task_id,
        tags=clearml_cfg.get("tags"),
        reuse=False,
    )
    try:
        report_hyperparams(task, config)
    except Exception:
        pass
    return task


def finalize_comparison_task(task, artifact_paths=None):
    if not (task and artifact_paths):
        return
    try:
        from pathlib import Path

        paths = [Path(p) for p in artifact_paths]
        upload_artifacts(task, paths)
    except Exception:
        pass
