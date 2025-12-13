from pathlib import Path
import argparse

from automl_lib.phases import run_inference
from automl_lib.config.loaders import load_inference_config
from automl_lib.cli.common import clearml_avoid_task_reuse
from automl_lib.cli.common import print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib inference phase (config-based).")
    parser.add_argument("--config", type=Path, default=Path("inference_config.yaml"), help="Path to inference config YAML")
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write inference info as JSON.",
    )
    args = parser.parse_args()
    load_inference_config(args.config)
    clearml_avoid_task_reuse()
    result = run_inference(args.config)
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
