from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .bootstrap import ensure_clearml_config_file

ensure_clearml_config_file()

try:
    from clearml import Task, TaskTypes, InputModel, Dataset, OutputModel  # type: ignore
except Exception:  # pragma: no cover
    Task = None
    TaskTypes = None
    InputModel = None
    Dataset = None
    OutputModel = None


def disable_resource_monitoring(task) -> None:
    """Best-effort disable ClearML ResourceMonitor for the given task.

    PipelineController generated step scripts start `auto_resource_monitoring=True` by default,
    which can produce noisy logs (GPU monitoring warnings) on non-GPU environments.
    """

    if not task:
        return
    try:
        monitor = getattr(task, "_resource_monitor", None)
        if monitor:
            try:
                monitor.stop()
            except Exception:
                pass
            try:
                setattr(task, "_resource_monitor", None)
            except Exception:
                pass
    except Exception:
        return


def init_task(
    project: str,
    name: str,
    task_type: str = "training",
    queue: Optional[str] = None,
    parent: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    reuse: bool = False,
):
    if Task is None:
        return None
    if not reuse:
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""
    try:
        tt = getattr(TaskTypes, task_type, getattr(TaskTypes, "custom", None))
        task = Task.init(
            project_name=project,
            task_name=name,
            task_type=tt,
            reuse_last_task_id=reuse,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            auto_connect_arg_parser=False,
        )
        if parent:
            try:
                if hasattr(task, "add_parent"):
                    task.add_parent(parent)
                else:
                    task.set_parent(parent)
            except Exception:
                try:
                    task.set_parent(parent)
                except Exception:
                    pass
        if tags:
            try:
                task.add_tags(list(tags))
            except Exception:
                pass
        if queue:
            try:
                task.set_parameter("requested_queue", queue)
            except Exception:
                pass
        return task
    except Exception:
        return None


def create_child_task(
    parent_task,
    project: str,
    name: str,
    task_type: str = "training",
    queue: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
):
    parent_id = None
    if parent_task and hasattr(parent_task, "id"):
        parent_id = parent_task.id
    return init_task(project, name, task_type, queue=queue, parent=parent_id, tags=tags, reuse=False)


def load_input_model(model_id: str) -> Optional[Path]:
    if InputModel is None:
        return None
    try:
        im = InputModel(model_id=model_id)
        lp = im.get_local_copy()
        return Path(lp) if lp else None
    except Exception:
        return None


def register_dataset(path: Path, name: str, project: Optional[str] = None, parent_ids=None) -> Optional[str]:
    if Dataset is None or not path.exists():
        return None
    try:
        ds = Dataset.create(dataset_name=name, dataset_project=project or "datasets", parent_datasets=parent_ids)
        ds.add_files(str(path))
        ds.upload()
        ds.finalize()
        return ds.id
    except Exception:
        return None


def register_output_model(task, model_path: Path, name: str) -> Optional[str]:
    if OutputModel is None or not model_path.exists() or task is None:
        return None
    try:
        om = OutputModel(task=task, name=name, framework="joblib")
        om.update_weights(str(model_path))
        om.update_design()
        return om.id
    except Exception:
        return None
