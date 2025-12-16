from pathlib import Path
import argparse

from automl_lib.phases import run_reporting
from automl_lib.config.loaders import load_training_config
from automl_lib.cli.common import clearml_avoid_task_reuse, load_json_or_yaml, maybe_clone_from_config, print_and_write_json
from automl_lib.clearml.context import get_run_id_env


def main() -> None:
    parser = argparse.ArgumentParser(description="automl_lib reporting phase (report task + markdown).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to training config YAML (default: config.yaml).",
    )
    parser.add_argument(
        "--preproc-config",
        type=Path,
        default=None,
        help="Optional path to preprocessing config YAML (default: config_preprocessing.yaml if exists, else --config).",
    )
    parser.add_argument("--preprocessing-info", type=Path, default=None, help="Optional JSON/YAML with preprocessing info")
    parser.add_argument("--training-info", type=Path, default=None, help="Optional JSON/YAML with training info")
    parser.add_argument(
        "--preprocessing-task-id",
        type=str,
        default=None,
        help="Optional ClearML task id for preprocessing (used for artifact download / links).",
    )
    parser.add_argument(
        "--training-task-id",
        type=str,
        default=None,
        help="Optional ClearML task id for training-summary (used for artifact download / links).",
    )
    parser.add_argument(
        "--pipeline-task-id",
        type=str,
        default=None,
        help="Optional pipeline controller task id (for linkage tags/markdown).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id override for report generation.")
    parser.add_argument(
        "--output-info",
        type=Path,
        default=None,
        help="Optional path to write resulting reporting info as JSON.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_training_config(config_path)
    if maybe_clone_from_config(cfg, phase="reporting", output_info=args.output_info):
        return

    preproc_config_path = args.preproc_config
    if preproc_config_path is None:
        cand = Path("config_preprocessing.yaml")
        preproc_config_path = cand if cand.exists() else config_path

    clearml_avoid_task_reuse()
    preprocessing_info = load_json_or_yaml(args.preprocessing_info)
    if not isinstance(preprocessing_info, dict):
        preprocessing_info = None
    training_info = load_json_or_yaml(args.training_info)
    if not isinstance(training_info, dict):
        training_info = None
    if args.preprocessing_task_id:
        preprocessing_info = dict(preprocessing_info or {})
        preprocessing_info["task_id"] = str(args.preprocessing_task_id)
    if args.training_task_id:
        training_info = dict(training_info or {})
        training_info["task_id"] = str(args.training_task_id)
    inferred_run_id = (
        args.run_id
        or (preprocessing_info or {}).get("run_id")
        or (training_info or {}).get("run_id")
        or getattr(getattr(cfg, "run", None), "id", None)
        or get_run_id_env()
    )
    if not inferred_run_id:
        parser.error(
            "run_reporting requires --run-id or --preprocessing-info/--training-info "
            "(or set run.id / AUTO_ML_RUN_ID)"
        )
    result = run_reporting(
        config_path,
        preprocessing_config_path=preproc_config_path,
        preprocessing_info=preprocessing_info,
        training_info=training_info,
        pipeline_task_id=args.pipeline_task_id,
        run_id=args.run_id,
    )
    try:
        print_and_write_json(result, args.output_info)
    except Exception:
        pass


if __name__ == "__main__":
    main()
