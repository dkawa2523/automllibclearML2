import unittest
from pathlib import Path
from unittest.mock import patch


class TestTrainingRunCompatWrapper(unittest.TestCase):
    def test_training_run_module_wrapper_calls_workflow_runner(self) -> None:
        # Keep automl_lib/training/run.py as a compatibility API entrypoint.
        # This test ensures the wrapper delegates to the workflow runner without executing heavy training.
        with patch("automl_lib.training.run._run_automl") as mocked:
            mocked.return_value = {"ok": True}

            from automl_lib.training.run import run_automl as run_automl_compat

            cfg = Path("config_training.yaml")
            result = run_automl_compat(cfg, dataset_id="ds1", parent_task_id="parent1")

            mocked.assert_called_once_with(cfg, dataset_id="ds1", parent_task_id="parent1")
            self.assertEqual(result, {"ok": True})

    def test_training_package_export_keeps_working(self) -> None:
        with patch("automl_lib.training.run._run_automl") as mocked:
            mocked.return_value = {"ok": True}

            from automl_lib.training import run_automl

            cfg = Path("config_training.yaml")
            result = run_automl(cfg, dataset_id="ds2", parent_task_id="parent2")

            mocked.assert_called_once_with(cfg, dataset_id="ds2", parent_task_id="parent2")
            self.assertEqual(result, {"ok": True})

