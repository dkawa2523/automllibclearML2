from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore


def clearml_avoid_task_reuse() -> None:
    """Ensure ClearML creates a fresh task (do not reuse previous CLEARML_TASK_ID)."""

    os.environ.pop("CLEARML_TASK_ID", None)
    os.environ["CLEARML_TASK_ID"] = ""


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

