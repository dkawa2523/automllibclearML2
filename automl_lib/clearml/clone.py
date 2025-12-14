from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .bootstrap import ensure_clearml_config_file
from .context import generate_run_id


_RUN_ID_BRACKET_RE = re.compile(r"\s*\[[0-9]{8}-[0-9]{6}-[0-9a-f]{6}\]\s*$", re.IGNORECASE)


def _dedupe_tags(tags: Iterable[str]) -> List[str]:
    seen = set()
    uniq: List[str] = []
    for t in tags:
        s = str(t).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _with_run_id_suffix(name: str, run_id: str) -> str:
    base = str(name or "").strip()
    base = _RUN_ID_BRACKET_RE.sub("", base).strip()
    if not base:
        base = "cloned-task"
    return f"{base} [{run_id}]"


def clone_task(
    template_task_id: str,
    *,
    run_id: Optional[str] = None,
    name: Optional[str] = None,
    project: Optional[str] = None,
    queue: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Iterable[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Clone a ClearML task and optionally enqueue it.

    Returns:
      (cloned_task_id, info_dict)
    """

    ensure_clearml_config_file()
    try:
        from clearml import Task  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("clearml is required for clone mode") from exc

    new_run_id = run_id or generate_run_id()
    cloned = Task.clone(
        source_task=str(template_task_id),
        name=(str(name).strip() if name else None),
        project=(str(project).strip() if project else None),
        comment=f"cloned_from={template_task_id} run_id={new_run_id}",
    )
    # Best-effort: if no name was provided, append run_id for discoverability.
    if not name:
        try:
            cloned.set_name(_with_run_id_suffix(getattr(cloned, "name", "") or "", new_run_id))
        except Exception:
            pass

    tags: List[str] = []
    if extra_tags:
        tags.extend(list(extra_tags))
    tags.extend([f"run:{new_run_id}", f"cloned_from:{template_task_id}"])
    try:
        cloned.add_tags(_dedupe_tags(tags))
    except Exception:
        pass

    # Always inject run.id so execution resolves the same run_id as the clone tag/name.
    merged_overrides: Dict[str, Any] = {}
    if overrides and isinstance(overrides, dict):
        merged_overrides.update(dict(overrides))
    merged_overrides["run.id"] = new_run_id

    if merged_overrides:
        # Store overrides as a configuration object for future consumption by the execution code.
        try:
            cloned.set_configuration_object(name="overrides", config_dict=merged_overrides)
        except Exception:
            pass
        # Also store as parameters (best-effort) so they are visible in UI even without config consumption.
        try:
            for k, v in merged_overrides.items():
                cloned.set_parameter(str(k), str(v))
        except Exception:
            pass

    queued = False
    if queue:
        try:
            Task.enqueue(cloned, queue_name=str(queue))
            queued = True
        except Exception:
            queued = False

    info: Dict[str, Any] = {
        "template_task_id": str(template_task_id),
        "task_id": str(getattr(cloned, "id", "") or ""),
        "run_id": new_run_id,
        "queued": queued,
        "queue": (str(queue) if queue else None),
    }
    return info["task_id"], info
