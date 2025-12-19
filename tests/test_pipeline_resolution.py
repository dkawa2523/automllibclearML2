import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


class TestPipelineResolution(unittest.TestCase):
    def _write_yaml(self, path: Path, payload: dict) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def test_run_pipeline_resolves_optional_config_paths(self) -> None:
        from automl_lib.pipeline import controller as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_yaml(
                tmp / "config.yaml",
                {
                    "data": {"dataset_id": "dummy"},
                    "models": [{"name": "ridge"}],
                    "clearml": {"enabled": True, "enable_pipeline": True},
                },
            )
            for name in ["config_dataregit.yaml", "config_editing.yaml", "config_preprocessing.yaml"]:
                (tmp / name).write_text("{}", encoding="utf-8")

            old = Path.cwd()
            try:
                os.chdir(tmp)
                with mock.patch.object(mod, "_run_clearml_pipeline_controller", return_value={"mode": "clearml"}) as mocked:
                    result = mod.run_pipeline(Path("config.yaml"), mode="clearml")
                    self.assertEqual(result["mode"], "clearml")

                    _, kwargs = mocked.call_args
                    self.assertEqual(Path(kwargs["data_registration_config"]).name, "config_dataregit.yaml")
                    self.assertEqual(Path(kwargs["data_editing_config"]).name, "config_editing.yaml")
                    self.assertEqual(Path(kwargs["preprocessing_config"]).name, "config_preprocessing.yaml")
            finally:
                os.chdir(old)

    def test_run_pipeline_rejects_non_clearml_mode(self) -> None:
        from automl_lib.pipeline import controller as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            self._write_yaml(
                tmp / "config.yaml",
                {
                    "data": {"dataset_id": "dummy"},
                    "models": [{"name": "ridge"}],
                    "clearml": {"enabled": True, "enable_pipeline": True},
                },
            )

            old = Path.cwd()
            try:
                os.chdir(tmp)
                with self.assertRaises(ValueError):
                    mod.run_pipeline(Path("config.yaml"), mode="auto")
                with self.assertRaises(ValueError):
                    mod.run_pipeline(Path("config.yaml"), mode="in_process")
            finally:
                os.chdir(old)
