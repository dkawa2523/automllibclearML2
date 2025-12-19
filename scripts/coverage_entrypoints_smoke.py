#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def _ensure_repo_root_on_syspath() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _set_offline_env() -> None:
    # Keep this script offline-safe by default (no ClearML server required).
    os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
    os.environ.setdefault("CLEARML_SKIP_PIP_FREEZE", "1")
    os.environ.setdefault("AUTO_ML_SKIP_CLEARML_PING", "1")
    os.environ.setdefault("AUTO_ML_SKIP_CLEARML_QUEUE_CHECK", "1")


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    import yaml  # type: ignore

    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_cli(module_name: str, argv: list[str]) -> None:
    import importlib

    mod = importlib.import_module(module_name)
    old_argv = sys.argv[:]
    sys.argv = argv[:]
    try:
        try:
            mod.main()
        except SystemExit as exc:
            # argparse / CLI may call sys.exit(0)
            if exc.code not in (None, 0):
                raise
    finally:
        sys.argv = old_argv


def _write_tiny_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "x1,cat,y",
                "0.1,pos,1.1",
                "-0.2,neg,-0.9",
                "0.3,pos,1.4",
                "-0.4,neg,-1.6",
                "0.5,pos,2.0",
                "-0.6,neg,-2.2",
                "0.7,pos,2.8",
                "-0.8,neg,-2.9",
                "0.9,pos,3.7",
                "-1.0,neg,-3.9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _pipeline_controller_smoke(config_path: Path) -> None:
    # Exercise PipelineController orchestration code without contacting a ClearML server.
    # We monkeypatch PipelineController to a lightweight stub.
    import clearml.automation.controller as ctrl  # type: ignore

    class _FakeTask:
        id = "fake_pipeline_task_id"

    class _FakePipelineController:
        def __init__(self, name: str, project: str, version: str) -> None:
            self.name = name
            self.project = project
            self.version = version
            self.task = _FakeTask()

        def set_default_execution_queue(self, queue: str) -> None:
            self.default_execution_queue = queue

        def add_function_step(self, **kwargs: Any) -> None:
            # Keep a minimal record (for debugging if needed).
            steps = getattr(self, "_steps", [])
            steps.append(dict(kwargs))
            setattr(self, "_steps", steps)

        def start_locally(self, run_pipeline_steps_locally: bool = True) -> None:  # noqa: ARG002
            return

        def start(self, queue: str) -> None:  # noqa: ARG002
            return

        def wait(self) -> None:
            return

    old_pc = getattr(ctrl, "PipelineController", None)
    setattr(ctrl, "PipelineController", _FakePipelineController)
    try:
        from automl_lib.config.loaders import load_training_config
        from automl_lib.pipeline.controller import _run_clearml_pipeline_controller

        cfg = load_training_config(config_path)
        info = _run_clearml_pipeline_controller(
            config_path=config_path,
            cfg=cfg,
            run_id="audit",
            data_registration_config=None,
            data_editing_config=None,
            preprocessing_config=None,
            inference_config=None,
            data_registration_config_data=None,
            data_editing_config_data=None,
            preprocessing_config_data=None,
            inference_config_data=None,
            training_config_data=None,
        )
        if not isinstance(info, dict) or info.get("mode") != "clearml_pipeline":
            raise RuntimeError(f"pipeline controller smoke failed: {info}")
    finally:
        if old_pc is None:
            try:
                delattr(ctrl, "PipelineController")
            except Exception:
                pass
        else:
            setattr(ctrl, "PipelineController", old_pc)


def main() -> None:
    _ensure_repo_root_on_syspath()
    _set_offline_env()

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        csv_path = base / "data.csv"
        _write_tiny_csv(csv_path)

        # ----------------------------
        # CLI smoke: training -> inference (local model_path)
        # ----------------------------
        run_id = "audit_run"
        train_out = base / "train_out"
        infer_out = base / "infer_out"
        train_cfg_path = base / "config_training.yaml"
        infer_cfg_path = base / "config_inference.yaml"

        _write_yaml(
            train_cfg_path,
            {
                "run": {"id": run_id},
                "data": {"csv_path": str(csv_path), "target_column": "y"},
                "models": [{"name": "ridge", "params": {"alpha": [1.0]}}],
                "cross_validation": {"n_folds": 2, "shuffle": True, "random_seed": 0},
                "output": {"output_dir": str(train_out), "save_models": True, "generate_plots": False},
                "evaluation": {"regression_metrics": ["rmse"], "classification_metrics": [], "primary_metric": "rmse"},
                "clearml": {"enabled": False},
            },
        )

        _run_cli(
            "automl_lib.cli.run_training",
            ["run_training", "--config", str(train_cfg_path)],
        )

        model_path = train_out / run_id / "models" / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Expected model file not found: {model_path}")

        _write_yaml(
            infer_cfg_path,
            {
                "run": {"id": run_id},
                "model_path": str(model_path),
                "model_name": "best_model",
                "clearml": {"enabled": False},
                "input": {"mode": "single", "single": {"x1": 0.25, "cat": "pos"}},
                "output_dir": str(infer_out),
            },
        )
        _run_cli(
            "automl_lib.cli.run_inference",
            ["run_inference", "--config", str(infer_cfg_path)],
        )

        # ----------------------------
        # PipelineController smoke (no execution, just orchestration code)
        # ----------------------------
        pipe_cfg_path = base / "config_pipeline.yaml"
        _write_yaml(
            pipe_cfg_path,
            {
                "run": {"id": "audit"},
                "data": {"dataset_id": "dummy", "csv_path": str(csv_path), "target_column": "y"},
                "preprocessing": {"numeric_imputation": ["mean"]},
                "models": [{"name": "ridge", "params": {"alpha": [1.0]}}],
                "clearml": {
                    "enabled": True,
                    "enable_pipeline": True,
                    "run_pipeline_locally": True,
                    "register_raw_dataset": False,
                    "enable_data_editing": False,
                },
            },
        )
        _pipeline_controller_smoke(pipe_cfg_path)


if __name__ == "__main__":
    main()
