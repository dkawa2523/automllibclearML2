import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


class TestPhaseProcessingSmoke(unittest.TestCase):
    def _write_yaml(self, path: Path, payload: dict) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def test_preprocessing_processing_passthrough_without_clearml(self) -> None:
        from automl_lib.phases.preprocessing import processing as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = tmp / "config_preprocessing.yaml"
            self._write_yaml(cfg, {"data": {"dataset_id": "dummy"}, "preprocessing": {}})

            with mock.patch.object(mod, "dataframe_from_dataset", side_effect=AssertionError("should not be called")):
                result = mod.run_preprocessing_processing(
                    cfg,
                    input_info={"dataset_id": "ds1", "task_id": "t0", "csv_path": "data/example.csv"},
                )
            self.assertEqual(result["dataset_id"], "ds1")
            self.assertEqual(result["task_id"], "t0")
            self.assertEqual(result["csv_path"], "data/example.csv")

    def test_data_registration_processing_passthrough_without_clearml(self) -> None:
        from automl_lib.phases.data_registration import processing as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = tmp / "config_dataregit.yaml"
            self._write_yaml(cfg, {"data": {"csv_path": "dummy.csv"}})
            result = mod.run_data_registration_processing(cfg)
            self.assertIsNone(result["dataset_id"])
            self.assertEqual(result["csv_path"], "dummy.csv")

    def test_data_editing_processing_writes_csv_without_clearml(self) -> None:
        from automl_lib.phases.data_editing import processing as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_in = tmp / "in.csv"
            csv_in.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
            csv_out = tmp / "out.csv"

            cfg = tmp / "config_editing.yaml"
            self._write_yaml(
                cfg,
                {
                    "data": {"csv_path": str(csv_in)},
                    "editing": {"enable": True, "output_path": str(csv_out)},
                    "clearml": {"enabled": False},
                },
            )

            with mock.patch.object(mod, "dataframe_from_dataset", side_effect=AssertionError("should not be called")):
                result = mod.run_data_editing_processing(cfg)

            self.assertTrue(csv_out.exists())
            self.assertIsNone(result["dataset_id"])
            self.assertEqual(result["csv_path"], str(csv_out))

    def test_training_processing_sets_env_for_run_automl(self) -> None:
        from automl_lib.phases.training import processing as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = tmp / "config_training.yaml"
            self._write_yaml(cfg, {"data": {"dataset_id": "dummy"}, "models": [{"name": "ridge"}]})

            os.environ["AUTO_ML_PARENT_TASK_ID"] = "old_parent"
            os.environ["AUTO_ML_DATASET_ID"] = "old_dataset"

            def _fake_run_automl(_path):
                self.assertEqual(os.environ.get("AUTO_ML_PARENT_TASK_ID"), "preproc_task")
                self.assertEqual(os.environ.get("AUTO_ML_DATASET_ID"), "ds_preproc")
                self.assertEqual(os.environ.get("CLEARML_TASK_ID"), "")
                return {
                    "summary_task_id": "sum",
                    "training_task_ids": ["c1"],
                    "dataset_id": "ds_preproc",
                    "metrics": [{"model": "ridge", "rmse": 1.0}],
                }

            with mock.patch.object(mod, "run_automl", side_effect=_fake_run_automl):
                result = mod.run_training_processing(cfg, input_info={"dataset_id": "ds_preproc", "task_id": "preproc_task"})

            self.assertEqual(result["dataset_id"], "ds_preproc")
            self.assertEqual(result["task_id"], "sum")
            self.assertEqual(result["training_task_ids"], ["c1"])
            self.assertTrue(result.get("metrics"))

            # restored
            self.assertEqual(os.environ.get("AUTO_ML_PARENT_TASK_ID"), "old_parent")
            self.assertEqual(os.environ.get("AUTO_ML_DATASET_ID"), "old_dataset")

    def test_comparison_processing_uses_metrics_without_clearml(self) -> None:
        from automl_lib.phases.comparison import processing as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = tmp / "config_comparison.yaml"
            out_dir = tmp / "comparison"
            self._write_yaml(cfg, {"output": {"output_dir": str(out_dir)}})

            training_info = {
                "task_id": "sum",
                "metrics": [
                    {"model": "A", "rmse": 1.0},
                    {"model": "B", "rmse": 0.8},
                ],
            }
            result = mod.run_comparison_processing(cfg, training_info=training_info)
            self.assertIsNone(result["task_id"])
            artifacts = result.get("artifacts") or []
            self.assertTrue(artifacts)
            for p in artifacts:
                self.assertTrue(Path(p).exists(), p)

