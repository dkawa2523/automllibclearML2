import os
import unittest


class _FakeLogger:
    def __init__(self) -> None:
        self.flushed = False


class _FakeTask:
    def __init__(self, task_id: str) -> None:
        self.id = task_id
        self.closed = False
        self.flushed = False

    def get_logger(self) -> _FakeLogger:
        return _FakeLogger()

    def get_status(self) -> str:
        return "in_progress"

    def flush(self, **_kwargs) -> None:
        self.flushed = True

    def close(self) -> None:
        self.closed = True


class TestClearMLManagerPipelineStepClose(unittest.TestCase):
    def test_close_skips_pipeline_step_task(self) -> None:
        from automl_lib.integrations.clearml.manager import ClearMLManager

        old_active = os.environ.get("AUTO_ML_PIPELINE_ACTIVE")
        old_task_id = os.environ.get("CLEARML_TASK_ID")
        try:
            os.environ["AUTO_ML_PIPELINE_ACTIVE"] = "1"
            os.environ["CLEARML_TASK_ID"] = "step_task_1"

            task = _FakeTask("step_task_1")
            mgr = ClearMLManager(
                None,
                task_name="dummy",
                task_type="training",
                existing_task=task,
            )
            mgr.close()

            self.assertTrue(task.flushed)
            self.assertFalse(task.closed)
        finally:
            if old_active is None:
                os.environ.pop("AUTO_ML_PIPELINE_ACTIVE", None)
            else:
                os.environ["AUTO_ML_PIPELINE_ACTIVE"] = old_active
            if old_task_id is None:
                os.environ.pop("CLEARML_TASK_ID", None)
            else:
                os.environ["CLEARML_TASK_ID"] = old_task_id

    def test_close_closes_non_step_task_in_pipeline(self) -> None:
        from automl_lib.integrations.clearml.manager import ClearMLManager

        old_active = os.environ.get("AUTO_ML_PIPELINE_ACTIVE")
        old_task_id = os.environ.get("CLEARML_TASK_ID")
        try:
            os.environ["AUTO_ML_PIPELINE_ACTIVE"] = "1"
            os.environ["CLEARML_TASK_ID"] = "step_task_1"

            task = _FakeTask("child_task_1")
            mgr = ClearMLManager(
                None,
                task_name="dummy",
                task_type="training",
                existing_task=task,
            )
            mgr.close()

            self.assertTrue(task.flushed)
            self.assertTrue(task.closed)
        finally:
            if old_active is None:
                os.environ.pop("AUTO_ML_PIPELINE_ACTIVE", None)
            else:
                os.environ["AUTO_ML_PIPELINE_ACTIVE"] = old_active
            if old_task_id is None:
                os.environ.pop("CLEARML_TASK_ID", None)
            else:
                os.environ["CLEARML_TASK_ID"] = old_task_id

