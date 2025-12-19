import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml


class TestCLISmoke(unittest.TestCase):
    def _write_yaml(self, path: Path, payload: dict) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def test_run_training_cli_writes_output_info(self) -> None:
        from automl_lib.cli import run_training as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_path = tmp / "config_training.yaml"
            self._write_yaml(
                cfg_path,
                {
                    "data": {"dataset_id": "dummy"},
                    "models": [{"name": "ridge"}],
                },
            )
            input_info_path = tmp / "preproc.json"
            input_info = {"dataset_id": "ds1", "task_id": "t0"}
            input_info_path.write_text(json.dumps(input_info), encoding="utf-8")
            out_path = tmp / "training_info.json"

            expected_result = {"task_id": "train_task", "training_task_ids": ["c1"]}
            with mock.patch.object(cli, "run_training", return_value=expected_result) as mocked:
                argv = ["prog", "--config", str(cfg_path), "--input-info", str(input_info_path), "--output-info", str(out_path)]
                with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    cli.main()

                mocked.assert_called_once()
                _, kwargs = mocked.call_args
                self.assertEqual(kwargs["input_info"], input_info)

            self.assertTrue(out_path.exists())
            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written, expected_result)

    def test_run_preprocessing_cli_prefers_config_preprocessing_yaml_chdir(self) -> None:
        from automl_lib.cli import run_preprocessing as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_pre = tmp / "config_preprocessing.yaml"
            cfg_train = tmp / "config.yaml"
            self._write_yaml(cfg_train, {"data": {"dataset_id": "dummy"}, "models": [{"name": "ridge"}]})
            self._write_yaml(cfg_pre, {"data": {"dataset_id": "dummy"}, "preprocessing": {}})
            out_path = tmp / "preproc_info.json"

            expected_result = {"dataset_id": "ds2", "task_id": "p1"}
            with mock.patch.object(cli, "run_preprocessing", return_value=expected_result) as mocked:
                argv = ["prog", "--output-info", str(out_path)]
                with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    old = Path.cwd()
                    try:
                        os.chdir(tmp)
                        cli.main()
                    finally:
                        os.chdir(old)

                mocked.assert_called_once()
                called_path = mocked.call_args.args[0]
                self.assertEqual(Path(called_path).name, "config_preprocessing.yaml")

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written, expected_result)

    def test_run_data_registration_cli_writes_output_info(self) -> None:
        from automl_lib.cli import run_data_registration as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_path = tmp / "config_dataregit.yaml"
            self._write_yaml(cfg_path, {"data": {"csv_path": "data/example.csv"}})
            out_path = tmp / "datareg_info.json"

            expected_result = {"dataset_id": "ds_raw", "task_id": "t_raw"}
            with mock.patch.object(cli, "run_data_registration", return_value=expected_result):
                argv = ["prog", "--config", str(cfg_path), "--output-info", str(out_path)]
                with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    cli.main()

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written, expected_result)

    def test_run_data_editing_cli_passes_input_info(self) -> None:
        from automl_lib.cli import run_data_editing as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_path = tmp / "config_editing.yaml"
            self._write_yaml(cfg_path, {"data": {"csv_path": "data/example.csv"}, "editing": {}})
            input_info_path = tmp / "datareg.json"
            input_info = {"dataset_id": "ds_raw", "task_id": "t_raw", "csv_path": "data/example.csv"}
            input_info_path.write_text(json.dumps(input_info), encoding="utf-8")
            out_path = tmp / "editing_info.json"

            expected_result = {"dataset_id": "ds_edited", "task_id": "t_edit"}
            with mock.patch.object(cli, "run_data_editing", return_value=expected_result) as mocked:
                argv = [
                    "prog",
                    "--config",
                    str(cfg_path),
                    "--input-info",
                    str(input_info_path),
                    "--output-info",
                    str(out_path),
                ]
                with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    cli.main()

                mocked.assert_called_once()
                _, kwargs = mocked.call_args
                self.assertEqual(kwargs["input_info"], input_info)

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written, expected_result)

    def test_run_inference_cli_writes_output_info(self) -> None:
        from automl_lib.cli import run_inference as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_path = tmp / "inference_config.yaml"
            self._write_yaml(
                cfg_path,
                {
                    "model_dir": "outputs/train/models",
                    "models": [{"name": "ridge"}],
                    "input": {"mode": "csv", "csv_path": "data/example.csv"},
                },
            )
            out_path = tmp / "inference_info.json"

            expected_result = {"task_id": "infer", "child_task_ids": [], "artifacts": []}
            with mock.patch.object(cli, "run_inference", return_value=expected_result):
                argv = ["prog", "--config", str(cfg_path), "--output-info", str(out_path)]
                with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    cli.main()

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written, expected_result)

    def test_run_pipeline_cli_passes_optional_configs(self) -> None:
        from automl_lib.cli import run_pipeline as cli

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_train = tmp / "config.yaml"
            self._write_yaml(cfg_train, {"data": {"dataset_id": "dummy"}, "models": [{"name": "ridge"}]})
            out_path = tmp / "pipeline_info.json"

            expected = {"mode": "clearml_pipeline"}
            with mock.patch.object(cli, "run_pipeline", return_value=expected) as mocked:
                argv = [
                    "prog",
                    "--config",
                    str(cfg_train),
                    "--mode",
                    "clearml",
                    "--datareg-config",
                    str(tmp / "config_dataregit.yaml"),
                    "--editing-config",
                    str(tmp / "config_editing.yaml"),
                    "--preproc-config",
                    str(tmp / "config_preprocessing.yaml"),
                    "--output-info",
                    str(out_path),
                ]
                with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                    cli.main()

                mocked.assert_called_once()
                _, kwargs = mocked.call_args
                self.assertEqual(kwargs["mode"], "clearml")
                self.assertTrue(str(kwargs["data_registration_config"]).endswith("config_dataregit.yaml"))
                self.assertTrue(str(kwargs["data_editing_config"]).endswith("config_editing.yaml"))
                self.assertTrue(str(kwargs["preprocessing_config"]).endswith("config_preprocessing.yaml"))

            written = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(written, expected)
