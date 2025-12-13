"""
ClearML integration for preprocessing phase.
タスク生成・Artifacts/Plots登録・親子リンクを扱う。
"""

import os
from typing import Dict, Any, Optional

from automl_lib.clearml import init_task, report_hyperparams, upload_artifacts


def create_preprocessing_task(config: Dict[str, Any], parent_task_id: Optional[str] = None) -> Dict[str, Any]:
    clearml_cfg = config.get("clearml") or {}
    project = clearml_cfg.get("project_name") or "AutoML"
    queue = clearml_cfg.get("queue")
    if not clearml_cfg.get("enabled", False):
        return {"task": None, "logger": None, "project": project, "queue": queue}

    # If running inside ClearML PipelineController local/agent step, reuse the step task.
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            current = Task.current_task()
            if current:
                try:
                    report_hyperparams(current, config)
                except Exception:
                    pass
                return {"task": current, "logger": current.get_logger(), "project": project, "queue": queue}
        except Exception:
            pass

    task = init_task(
        project=project,
        name="preprocessing",
        task_type="data_processing",
        queue=queue,
        parent=parent_task_id,
        tags=clearml_cfg.get("tags"),
        reuse=False,
    )
    logger = task.get_logger() if task else None
    if task:
        try:
            report_hyperparams(task, config)
        except Exception:
            pass
    return {"task": task, "logger": logger, "project": project, "queue": queue}


def finalize_preprocessing_task(task, artifact_paths=None):
    if artifact_paths:
        upload_artifacts(task, artifact_paths)
