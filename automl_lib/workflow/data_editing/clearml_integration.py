"""
ClearML integration for data_editing phase.
タスク生成を扱う。
"""

from typing import Any, Dict, Optional

from automl_lib.integrations.clearml.logging import upload_artifacts
from automl_lib.integrations.clearml.context import build_run_context, get_run_id_env, resolve_dataset_key, resolve_run_id
from automl_lib.integrations.clearml.naming import build_project_path, build_tags, task_name
from automl_lib.integrations.clearml.manager import ClearMLManager, build_clearml_config_from_dict


def create_data_editing_task(config: Dict[str, Any], parent_task_id: Optional[str] = None) -> Dict[str, Any]:
    clearml_cfg = config.get("clearml") or {}
    clearml_settings = build_clearml_config_from_dict(clearml_cfg if isinstance(clearml_cfg, dict) else {})
    if not (clearml_settings and clearml_settings.enabled):
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
    tags = build_tags(ctx, phase="data_editing", extra=clearml_cfg.get("tags"))

    mgr = ClearMLManager(
        clearml_settings,
        task_name=task_name("data_editing", ctx),
        task_type="data_processing",
        default_project=project,
        project=project,
        parent=parent_task_id,
        extra_tags=tags,
    )
    try:
        mgr.connect_params_sections(
            {
                "Input": {
                    "dataset_id": (config.get("data") or {}).get("dataset_id") or "",
                    "csv_path": (config.get("data") or {}).get("csv_path") or "",
                },
                "Editing": config.get("editing") or {},
            }
        )
    except Exception:
        pass
    try:
        mgr.connect_configuration(config, name="OmegaConf")
    except Exception:
        pass
    return {"task": mgr.task, "logger": mgr.logger}


def finalize_data_editing_task(task, artifact_paths=None) -> None:
    if task and artifact_paths:
        upload_artifacts(task, artifact_paths)
