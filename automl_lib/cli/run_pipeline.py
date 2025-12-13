from pathlib import Path
import argparse

from automl_lib.pipeline.controller import run_pipeline
from automl_lib.config.loaders import load_training_config
from automl_lib.cli.common import clearml_avoid_task_reuse, print_and_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoML pipeline (automl_lib wrapper).")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to training config YAML")
    parser.add_argument(
        "--datareg-config",
        type=Path,
        default=None,
        help="Optional path to data_registration config YAML (default: config_dataregit.yaml if exists).",
    )
    parser.add_argument(
        "--editing-config",
        type=Path,
        default=None,
        help="Optional path to data_editing config YAML (default: config_editing.yaml if exists).",
    )
    parser.add_argument(
        "--preproc-config",
        type=Path,
        default=None,
        help="Optional path to preprocessing config YAML (default: config_preprocessing.yaml if exists).",
    )
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=None,
        help="Optional path to comparison config YAML (default: config_comparison.yaml if exists).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "clearml", "in_process"],
        help="Execution mode: auto (default), clearml (PipelineController), in_process (call phases directly).",
    )
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write pipeline info as JSON.",
    )
    args = parser.parse_args()
    # Validate config
    load_training_config(args.config)
    # Avoid task reuse
    clearml_avoid_task_reuse()
    result = run_pipeline(
        args.config,
        mode=args.mode,
        data_registration_config=args.datareg_config,
        data_editing_config=args.editing_config,
        preprocessing_config=args.preproc_config,
        comparison_config=args.comparison_config,
    )
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
