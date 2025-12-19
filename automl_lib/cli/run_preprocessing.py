from pathlib import Path
import argparse

from automl_lib.workflow import run_preprocessing
from automl_lib.config.loaders import load_preprocessing_config
from automl_lib.cli.common import clearml_avoid_task_reuse, load_json_or_yaml, maybe_clone_from_config, print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib preprocessing phase.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config_preprocessing.yaml if exists, else config.yaml)",
    )
    parser.add_argument("--input-info", type=Path, default=None, help="Optional JSON/YAML with dataset info")
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write resulting dataset/task info as JSON.",
    )
    args = parser.parse_args()
    config_path = args.config
    if config_path is None:
        config_path = Path("config_preprocessing.yaml")
        if not config_path.exists():
            config_path = Path("config.yaml")
    cfg = load_preprocessing_config(config_path)
    if maybe_clone_from_config(cfg, phase="preprocessing", output_info=args.output_info):
        return
    clearml_avoid_task_reuse()
    input_info = load_json_or_yaml(args.input_info)
    result = run_preprocessing(config_path, input_info=input_info)
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
