"""
ClearML integration for inference phase.
親タスク(summary)＋子タスク(single/grid/optuna)の生成やArtifacts登録を担当。
"""

from typing import Dict, Any, Optional

from automl_lib.clearml import init_task, create_child_task, report_hyperparams, upload_artifacts


def create_inference_summary_task(config: Dict[str, Any], parent_task_id: Optional[str] = None) -> Dict[str, Any]:
    clearml_cfg = config.get("clearml") or {}
    project = clearml_cfg.get("project_name") or "AutoML"
    queue = clearml_cfg.get("queue")
    task = init_task(
        project=project,
        name=clearml_cfg.get("task_name") or "inference-summary",
        task_type="inference",
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


def create_inference_child_task(project: str, mode: str, parent_task, queue: Optional[str] = None, tags=None):
    """
    mode: single/grid/optuna など
    """
    name = f"inference-{mode}"
    child = create_child_task(
        parent_task=parent_task,
        project=project,
        name=name,
        task_type="inference",
        queue=queue,
        tags=tags,
    )
    return child


def finalize_inference_task(task, artifact_paths=None):
    if artifact_paths:
        upload_artifacts(task, artifact_paths)
