import os
from typing import Any, Dict, Optional

from automl_lib.clearml import init_task, report_hyperparams, upload_artifacts


def create_data_editing_task(config: Dict[str, Any], parent_task_id: Optional[str] = None) -> Dict[str, Any]:
    clearml_cfg = config.get("clearml") or {}
    if not clearml_cfg.get("enabled", False):
        return {"task": None, "logger": None}
    project = clearml_cfg.get("project_name") or "AutoML"
    queue = clearml_cfg.get("queue")

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
                return {"task": current, "logger": current.get_logger()}
        except Exception:
            pass

    task = init_task(
        project=project,
        name="data_editing",
        task_type="data_processing",
        queue=queue,
        parent=parent_task_id,
        tags=clearml_cfg.get("tags"),
        reuse=False,
    )
    if task:
        try:
            report_hyperparams(task, config)
        except Exception:
            pass
    return {"task": task, "logger": task.get_logger() if task else None}


def finalize_data_editing_task(task, artifact_paths=None) -> None:
    if task and artifact_paths:
        upload_artifacts(task, artifact_paths)
