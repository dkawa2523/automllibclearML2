"""
ClearML integration for comparison phase.
タスク生成とArtifacts登録を担うシンプルなスタブ。
"""

import os
from typing import Dict, Any, Optional

from automl_lib.clearml import init_task, report_hyperparams_sections, upload_artifacts
from automl_lib.clearml.context import build_run_context, get_run_id_env, resolve_dataset_key, resolve_run_id
from automl_lib.clearml.naming import build_project_path, build_tags, task_name


def create_comparison_task(
    config: Dict[str, Any],
    parent_task_id: Optional[str] = None,
    project_name: Optional[str] = None,
):
    clearml_cfg = config.get("clearml") or {}
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
        project_root=(project_name or clearml_cfg.get("project_name") or "AutoML"),
        dataset_project=clearml_cfg.get("dataset_project"),
        user=run_cfg.get("user"),
    )
    naming_cfg = clearml_cfg.get("naming") or {}
    project = build_project_path(ctx, project_mode=naming_cfg.get("project_mode", "root"))
    queue = clearml_cfg.get("queue") or "services"
    tags = build_tags(ctx, phase="comparison", extra=clearml_cfg.get("tags"))

    if not clearml_cfg.get("enabled", False):
        return None

    # If running inside ClearML PipelineController step, reuse the step task.
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            current = Task.current_task()
            if current:
                try:
                    current.set_name(task_name("comparison", ctx))
                except Exception:
                    pass
                if parent_task_id:
                    try:
                        current.add_parent(str(parent_task_id))
                    except Exception:
                        try:
                            current.set_parent(str(parent_task_id))
                        except Exception:
                            pass
                try:
                    current.add_tags(tags)
                except Exception:
                    pass
                try:
                    report_hyperparams_sections(
                        current,
                        {
                            "Run": config.get("run") or {},
                            "Ranking": config.get("ranking") or {},
                            "Output": config.get("output") or {},
                            "ClearML": clearml_cfg,
                        },
                    )
                except Exception:
                    pass
                return current
        except Exception:
            pass

    task = init_task(
        project=project,
        name=(clearml_cfg.get("comparison_task_name") or task_name("comparison", ctx)),
        task_type="analysis",
        queue=queue,
        parent=parent_task_id,
        tags=tags,
        reuse=False,
    )
    try:
        report_hyperparams_sections(
            task,
            {
                "Run": config.get("run") or {},
                "Ranking": config.get("ranking") or {},
                "Output": config.get("output") or {},
                "ClearML": clearml_cfg,
            },
        )
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
