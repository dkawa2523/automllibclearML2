from pathlib import Path
import argparse

from automl_lib.workflow import run_data_registration
from automl_lib.config.loaders import load_data_registration_config
from automl_lib.cli.common import clearml_avoid_task_reuse, maybe_clone_from_config, print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib data registration phase.")
    parser.add_argument("--config", type=Path, default=Path("config_dataregit.yaml"), help="Path to config YAML")
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write resulting dataset/task info as JSON.",
    )
    args = parser.parse_args()
    cfg = load_data_registration_config(args.config)
    if maybe_clone_from_config(cfg, phase="data_registration", output_info=args.output_info):
        return
    clearml_avoid_task_reuse()
    result = run_data_registration(args.config)
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
