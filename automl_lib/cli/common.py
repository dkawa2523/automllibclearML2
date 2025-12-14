from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore


def clearml_avoid_task_reuse() -> None:
    """Ensure ClearML creates a fresh task (do not reuse previous CLEARML_TASK_ID)."""

    # When running inside an existing ClearML task (e.g., cloned/enqueued execution),
    # CLEARML_TASK_ID is provided by the agent and must be preserved.
    current = os.environ.get("CLEARML_TASK_ID")
    if current and str(current).strip():
        return
    os.environ.pop("CLEARML_TASK_ID", None)
    os.environ["CLEARML_TASK_ID"] = ""


def maybe_clone_from_config(cfg: Any, *, phase: str, output_info: Optional[Path] = None) -> bool:
    """Clone/enqueue a template task when config requests execution_mode='clone'.

    When this returns True, callers should exit without running the normal phase.
    """

    clearml_cfg = getattr(cfg, "clearml", None)
    if clearml_cfg is None:
        return False

    try:
        mode = str(getattr(clearml_cfg, "execution_mode", "new") or "new").strip().lower()
    except Exception:
        mode = "new"
    if mode != "clone":
        return False

    enabled = bool(getattr(clearml_cfg, "enabled", False))
    template_task_id = getattr(clearml_cfg, "template_task_id", None)
    if not enabled:
        raise ValueError("clearml.execution_mode='clone' requires clearml.enabled=true")
    if not template_task_id:
        raise ValueError("clearml.execution_mode='clone' requires clearml.template_task_id")

    queue = getattr(clearml_cfg, "queue", None)
    run_tasks_locally = bool(getattr(clearml_cfg, "run_tasks_locally", True))
    queue_to_use = (str(queue) if (queue and not run_tasks_locally) else None)

    project = getattr(clearml_cfg, "project_name", None)
    project = str(project).strip() if project else None

    extra_tags = []
    try:
        extra_tags.extend([str(t) for t in (getattr(clearml_cfg, "tags", []) or []) if str(t).strip()])
    except Exception:
        pass
    extra_tags.append(f"phase:{phase}")

    from automl_lib.clearml.clone import clone_task

    _, info = clone_task(
        str(template_task_id),
        queue=queue_to_use,
        project=project,
        extra_tags=extra_tags,
        overrides=None,
    )
    print_and_write_json(info, output_info)
    return True


def load_json_or_yaml(path: Optional[Path]) -> Any:
    if not path:
        return None
    payload = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(payload)
    if suffix == ".json":
        return json.loads(payload)
    try:
        return json.loads(payload)
    except Exception:
        return yaml.safe_load(payload)


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def print_and_write_json(obj: Any, output_path: Optional[Path]) -> None:
    payload = dump_json(obj)
    print(payload)
    if output_path:
        output_path.write_text(payload, encoding="utf-8")
