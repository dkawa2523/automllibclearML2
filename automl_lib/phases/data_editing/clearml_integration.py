import os
from typing import Any, Dict, Optional

from automl_lib.clearml import init_task, report_hyperparams_sections, upload_artifacts
from automl_lib.clearml.context import build_run_context, get_run_id_env, resolve_dataset_key, resolve_run_id
from automl_lib.clearml.naming import build_project_path, build_tags, task_name


def create_data_editing_task(config: Dict[str, Any], parent_task_id: Optional[str] = None) -> Dict[str, Any]:
    clearml_cfg = config.get("clearml") or {}
    if not clearml_cfg.get("enabled", False):
        return {"task": None, "logger": None}
    run_cfg = config.get("run") or {}
    data_cfg = config.get("data") or {}
    run_id = resolve_run_id(from_config=run_cfg.get("id"), from_env=get_run_id_env())
    dataset_key = resolve_dataset_key(
        explicit=run_cfg.get("dataset_key"),
        dataset_id=data_cfg.get("dataset_id"),
        csv_path=data_cfg.get("csv_path"),
    )
    ctx = build_run_context(
        run_id=run_id,
        dataset_key=dataset_key,
        project_root=clearml_cfg.get("project_name") or "AutoML",
        dataset_project=clearml_cfg.get("dataset_project"),
        user=run_cfg.get("user"),
    )
    naming_cfg = clearml_cfg.get("naming") or {}
    project = build_project_path(ctx, project_mode=naming_cfg.get("project_mode", "root"))
    queue = clearml_cfg.get("queue")
    tags = build_tags(ctx, phase="data_editing", extra=clearml_cfg.get("tags"))

    # If running inside ClearML PipelineController step, reuse the step task.
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            current = Task.current_task()
            if current:
                try:
                    current.add_tags(tags)
                except Exception:
                    pass
                try:
                    report_hyperparams_sections(
                        current,
                        {
                            "Run": config.get("run") or {},
                            "Data": config.get("data") or {},
                            "Editing": config.get("editing") or {},
                            "ClearML": clearml_cfg,
                        },
                    )
                except Exception:
                    pass
                return {"task": current, "logger": current.get_logger()}
        except Exception:
            pass

    task = init_task(
        project=project,
        name=task_name("data_editing", ctx),
        task_type="data_processing",
        queue=queue,
        parent=parent_task_id,
        tags=tags,
        reuse=False,
    )
    if task:
        try:
            report_hyperparams_sections(
                task,
                {
                    "Run": config.get("run") or {},
                    "Data": config.get("data") or {},
                    "Editing": config.get("editing") or {},
                    "ClearML": clearml_cfg,
                },
            )
        except Exception:
            pass
    return {"task": task, "logger": task.get_logger() if task else None}


def finalize_data_editing_task(task, artifact_paths=None) -> None:
    if task and artifact_paths:
        upload_artifacts(task, artifact_paths)
