import unittest
from types import SimpleNamespace
from unittest import mock

from automl_lib.cli.common import maybe_clone_from_config


class TestCloneModeFromConfig(unittest.TestCase):
    def test_maybe_clone_from_config_calls_clone_task(self) -> None:
        cfg = SimpleNamespace(
            clearml=SimpleNamespace(
                enabled=True,
                execution_mode="clone",
                template_task_id="template-1",
                queue="default",
                run_tasks_locally=False,
                project_name="AutoML",
                tags=["automl"],
            )
        )
        with mock.patch(
            "automl_lib.integrations.clearml.clone.clone_task",
            return_value=("t2", {"task_id": "t2"}),
        ) as mocked, mock.patch(
            "automl_lib.cli.common.print_and_write_json"
        ) as pwj:
            ok = maybe_clone_from_config(cfg, phase="training", output_info=None)
        self.assertTrue(ok)
        mocked.assert_called_once()
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs.get("queue"), "default")
        self.assertEqual(kwargs.get("project"), "AutoML")
        tags = kwargs.get("extra_tags") or []
        self.assertIn("phase:training", tags)
        self.assertIn("automl", tags)
        pwj.assert_called_once()

    def test_maybe_clone_from_config_uses_no_queue_when_local(self) -> None:
        cfg = SimpleNamespace(
            clearml=SimpleNamespace(
                enabled=True,
                execution_mode="clone",
                template_task_id="template-1",
                queue="default",
                run_tasks_locally=True,
                project_name="AutoML",
                tags=[],
            )
        )
        with mock.patch(
            "automl_lib.integrations.clearml.clone.clone_task",
            return_value=("t2", {"task_id": "t2"}),
        ) as mocked, mock.patch(
            "automl_lib.cli.common.print_and_write_json"
        ):
            ok = maybe_clone_from_config(cfg, phase="training", output_info=None)
        self.assertTrue(ok)
        _, kwargs = mocked.call_args
        self.assertIsNone(kwargs.get("queue"))

    def test_maybe_clone_from_config_errors_when_disabled(self) -> None:
        cfg = SimpleNamespace(
            clearml=SimpleNamespace(
                enabled=False,
                execution_mode="clone",
                template_task_id="template-1",
                queue="default",
                run_tasks_locally=False,
                project_name="AutoML",
                tags=[],
            )
        )
        with self.assertRaises(ValueError):
            maybe_clone_from_config(cfg, phase="training", output_info=None)
