from pathlib import Path
import argparse

from automl_lib.phases import run_data_editing
from automl_lib.config.loaders import load_data_editing_config
from automl_lib.cli.common import clearml_avoid_task_reuse, load_json_or_yaml, print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib data editing phase.")
    parser.add_argument("--config", type=Path, default=Path("config_editing.yaml"), help="Path to config YAML")
    parser.add_argument("--input-info", type=Path, default=None, help="Optional JSON/YAML with data registration info")
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write resulting dataset/task info as JSON.",
    )
    args = parser.parse_args()
    load_data_editing_config(args.config)
    clearml_avoid_task_reuse()
    input_info = load_json_or_yaml(args.input_info)
    result = run_data_editing(args.config, input_info=input_info)
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
