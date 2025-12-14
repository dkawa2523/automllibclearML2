from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from automl_lib.clearml import clone_task
from automl_lib.cli.common import load_json_or_yaml, print_and_write_json


def _extract_overrides(payload: Any) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    overrides = payload.get("overrides", payload)
    if overrides is None:
        return None
    if not isinstance(overrides, dict):
        return None
    return dict(overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone a ClearML task and optionally enqueue it.")
    parser.add_argument("--task-id", required=True, help="Template ClearML task id to clone.")
    parser.add_argument("--queue", default=None, help="Optional queue name to enqueue the cloned task.")
    parser.add_argument("--project", default=None, help="Optional project name for the cloned task.")
    parser.add_argument("--name", default=None, help="Optional name for the cloned task (default: template name + run_id).")
    parser.add_argument("--run-id", default=None, help="Optional run_id to tag the cloned task (default: auto-generated).")
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="Optional JSON/YAML file containing overrides (either {overrides:{...}} or {...}).",
    )
    parser.add_argument("--output-info", type=Path, default=None, help="Optional path to write output info as JSON.")
    args = parser.parse_args()

    overrides_payload = load_json_or_yaml(args.overrides)
    overrides = _extract_overrides(overrides_payload)
    _, info = clone_task(
        args.task_id,
        queue=args.queue,
        project=args.project,
        name=args.name,
        run_id=args.run_id,
        overrides=overrides,
    )
    print_and_write_json(info, args.output_info)


if __name__ == "__main__":
    main()

