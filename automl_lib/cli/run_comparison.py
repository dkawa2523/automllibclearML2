from pathlib import Path
import argparse

from automl_lib.phases import run_comparison
from automl_lib.config.loaders import load_comparison_config
from automl_lib.cli.common import clearml_avoid_task_reuse, load_json_or_yaml, print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib comparison phase.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config_comparison.yaml if exists, else config.yaml)",
    )
    parser.add_argument(
        "--parent-task-id",
        type=str,
        action="append",
        default=None,
        help="Optional ClearML parent task id (repeatable) to auto-discover training tasks.",
    )
    parser.add_argument(
        "--training-info",
        type=Path,
        action="append",
        default=None,
        help="Optional JSON/YAML file containing training_info (repeatable).",
    )
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write resulting comparison info as JSON.",
    )
    args = parser.parse_args()
    config_path = args.config
    if config_path is None:
        config_path = Path("config_comparison.yaml")
        if not config_path.exists():
            config_path = Path("config.yaml")
    load_comparison_config(config_path)
    clearml_avoid_task_reuse()
    training_infos = []
    if args.training_info:
        for p in args.training_info:
            payload = load_json_or_yaml(p)
            if payload is None:
                continue
            if isinstance(payload, list):
                for idx, item in enumerate(payload):
                    if isinstance(item, dict):
                        d = dict(item)
                        d.setdefault("run_label", f"{Path(p).stem}:{idx+1}")
                        training_infos.append(d)
                continue
            if isinstance(payload, dict):
                d = dict(payload)
                d.setdefault("run_label", Path(p).stem)
                training_infos.append(d)
    training_info_arg = training_infos if training_infos else None
    parent_task_ids = args.parent_task_id if args.parent_task_id else None
    result = run_comparison(config_path, training_info=training_info_arg, parent_task_id=parent_task_ids)
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
