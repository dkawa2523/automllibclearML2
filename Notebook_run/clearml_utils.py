from __future__ import annotations

from typing import Any, Dict, Optional


def task_url(task_id: Optional[str]) -> str:
    if not task_id:
        return ""
    try:
        from clearml import Task  # type: ignore

        t = Task.get_task(task_id=str(task_id))
        return str(t.get_output_log_web_page())
    except Exception:
        return ""


def dataset_url(dataset_id: Optional[str]) -> str:
    if not dataset_id:
        return ""
    try:
        from clearml import Dataset  # type: ignore

        ds = Dataset.get(dataset_id=str(dataset_id))
        task = getattr(ds, "_task", None)
        if task is None:
            return ""
        app = task._get_app_server()
        project = task.project if getattr(task, "project", None) is not None else "*"
        return f"{app}/datasets/simple/{project}/experiments/{task.id}"
    except Exception:
        return ""


def find_latest_task_by_prefix(*, run_id: str, prefix: str):
    try:
        from clearml import Task  # type: ignore

        tasks = Task.get_tasks(tags=[f"run:{run_id}"])
        cands = [t for t in (tasks or []) if str(getattr(t, "name", "") or "").startswith(prefix)]
        if not cands:
            return None
        cands = sorted(cands, key=lambda t: float(getattr(getattr(t, "data", None), "last_update", 0) or 0))
        return cands[-1]
    except Exception:
        return None


def task_user_properties(task) -> Dict[str, str]:
    if task is None:
        return {}
    try:
        props = task.get_user_properties(value_only=True)
        if isinstance(props, dict):
            return {str(k): str(v) for k, v in props.items()}
    except Exception:
        pass
    return {}

