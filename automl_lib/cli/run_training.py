from pathlib import Path
import argparse

from automl_lib.phases import run_training
from automl_lib.config.loaders import load_training_config
from automl_lib.cli.common import clearml_avoid_task_reuse, load_json_or_yaml, print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib training phase.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config_training.yaml if exists, else config.yaml)",
    )
    parser.add_argument("--input-info", type=Path, default=None, help="Optional JSON/YAML with preprocessing output info")
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write resulting task info as JSON.",
    )
    args = parser.parse_args()
    config_path = args.config
    if config_path is None:
        config_path = Path("config_training.yaml")
        if not config_path.exists():
            config_path = Path("config.yaml")
    load_training_config(config_path)
    clearml_avoid_task_reuse()
    input_info = load_json_or_yaml(args.input_info)
    result = run_training(config_path, input_info=input_info)
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
