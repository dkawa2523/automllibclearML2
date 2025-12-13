"""
ClearML integration for training phase.
タスク生成・親子リンク・queue設定・Artifacts/Plots登録を担当する。
現状は簡易なラッパで、後続の処理移植に合わせて拡張する。
"""

from typing import Dict, Any, Optional

from automl_lib.clearml import (
    init_task,
    create_child_task,
    report_hyperparams,
    upload_artifacts,
)


def create_training_tasks(
    config: Dict[str, Any],
    parent_task_id: Optional[str] = None,
    project_name: Optional[str] = None,
    tags=None,
) -> Dict[str, Any]:
    """
    Create training summary task (parent) and return handles for later logging.
    """
    clearml_cfg = config.get("clearml") or {}
    project = project_name or clearml_cfg.get("project_name") or "AutoML"
    queue = clearml_cfg.get("queue")

    summary_task = init_task(
        project=project,
        name=clearml_cfg.get("task_name") or "training-summary",
        task_type="training",
        queue=queue,
        parent=parent_task_id,
        tags=tags or clearml_cfg.get("tags"),
        reuse=False,
    )
    logger = summary_task.get_logger() if summary_task else None

    # Hyperparametersを親タスクに登録
    if summary_task:
        try:
            report_hyperparams(summary_task, config)
        except Exception:
            pass

    return {
        "task": summary_task,
        "logger": logger,
        "project": project,
        "queue": queue,
    }


def create_model_child_task(
    project: str,
    model_name: str,
    preproc_name: str,
    parent_task,
    queue: Optional[str] = None,
    tags=None,
):
    """
    Create per-model training task under <project>/train_models.
    """
    child_proj = f"{project}/train_models"
    child = create_child_task(
        parent_task=parent_task,
        project=child_proj,
        name=f"train_{model_name}",
        task_type="training",
        queue=queue,
        tags=tags,
    )
    return child


def finalize_training_task(task, artifact_paths=None):
    """
    Upload artifacts for the training summary task.
    """
    if artifact_paths:
        upload_artifacts(task, artifact_paths)
