import os
import unittest
from unittest import mock

from automl_lib.cli.common import clearml_avoid_task_reuse
from automl_lib.clearml.clone import clone_task
from automl_lib.clearml.overrides import apply_overrides, get_task_overrides
from automl_lib.clearml.utils import init_task


class TestCloneModeHelpers(unittest.TestCase):
    def test_apply_overrides_supports_paths_and_indices(self) -> None:
        cfg = {
            "data": {"dataset_id": "old"},
            "models": [{"name": "A", "params": {"n_estimators": 10}}],
        }
        overrides = {
            "data.dataset_id": "new",
            "models[0].params.n_estimators": 500,
            "models[1].name": "B",
        }
        out = apply_overrides(cfg, overrides)
        self.assertEqual(out["data"]["dataset_id"], "new")
        self.assertEqual(out["models"][0]["params"]["n_estimators"], 500)
        self.assertEqual(out["models"][1]["name"], "B")

    def test_clearml_avoid_task_reuse_preserves_existing_task_id(self) -> None:
        old = os.environ.get("CLEARML_TASK_ID")
        try:
            os.environ["CLEARML_TASK_ID"] = "existing-task"
            clearml_avoid_task_reuse()
            self.assertEqual(os.environ.get("CLEARML_TASK_ID"), "existing-task")
        finally:
            if old is None:
                os.environ.pop("CLEARML_TASK_ID", None)
            else:
                os.environ["CLEARML_TASK_ID"] = old

    def test_init_task_reuses_current_task_when_task_id_set(self) -> None:
        class DummyTask:
            def __init__(self) -> None:
                self.parents = []
                self.tags = []
                self.params = {}

            def add_parent(self, parent: str) -> None:
                self.parents.append(parent)

            def add_tags(self, tags) -> None:
                self.tags.extend(tags)

            def set_parameter(self, name: str, value: str) -> None:
                self.params[name] = value

        dummy = DummyTask()
        old = os.environ.get("CLEARML_TASK_ID")
        try:
            os.environ["CLEARML_TASK_ID"] = "running-task"
            with mock.patch("automl_lib.clearml.utils.Task.current_task", return_value=dummy), mock.patch(
                "automl_lib.clearml.utils.Task.init", side_effect=AssertionError("Task.init should not be called")
            ):
                task = init_task(
                    project="proj",
                    name="name",
                    task_type="training",
                    queue="q1",
                    parent="p1",
                    tags=["a", "a", "b"],
                    reuse=False,
                )
            self.assertIs(task, dummy)
            self.assertEqual(dummy.parents, ["p1"])
            self.assertEqual(dummy.tags, ["a", "b"])
            self.assertEqual(dummy.params.get("requested_queue"), "q1")
        finally:
            if old is None:
                os.environ.pop("CLEARML_TASK_ID", None)
            else:
                os.environ["CLEARML_TASK_ID"] = old

    def test_clone_task_adds_tags_and_overrides(self) -> None:
        class DummyCloned:
            def __init__(self) -> None:
                self.id = "cloned-id"
                self.name = "template-name"
                self.tags = []
                self.configs = {}
                self.params = {}

            def set_name(self, name: str) -> None:
                self.name = name

            def add_tags(self, tags) -> None:
                self.tags.extend(tags)

            def set_configuration_object(self, name: str, config_dict=None, **kwargs) -> None:
                self.configs[name] = config_dict

            def set_parameter(self, name: str, value: str, **kwargs) -> None:
                self.params[name] = value

        cloned = DummyCloned()
        with mock.patch("clearml.Task.clone", return_value=cloned) as mocked_clone, mock.patch(
            "clearml.Task.enqueue", return_value=True
        ) as mocked_enqueue:
            task_id, info = clone_task(
                "template-1",
                run_id="20251213-153012-a1b2c3",
                queue="default",
                overrides={"data.dataset_id": "ds1"},
                extra_tags=["x", "x"],
            )
        mocked_clone.assert_called_once()
        mocked_enqueue.assert_called_once()
        self.assertEqual(task_id, "cloned-id")
        self.assertEqual(info["task_id"], "cloned-id")
        self.assertEqual(info["run_id"], "20251213-153012-a1b2c3")
        self.assertTrue(info["queued"])
        self.assertIn("run:20251213-153012-a1b2c3", cloned.tags)
        self.assertIn("cloned_from:template-1", cloned.tags)
        self.assertIn("x", cloned.tags)
        self.assertTrue(cloned.name.endswith("[20251213-153012-a1b2c3]"))
        self.assertEqual(cloned.configs.get("overrides"), {"data.dataset_id": "ds1", "run.id": "20251213-153012-a1b2c3"})
        self.assertEqual(cloned.params.get("data.dataset_id"), "ds1")
        self.assertEqual(cloned.params.get("run.id"), "20251213-153012-a1b2c3")

    def test_get_task_overrides_reads_clearml_config_object(self) -> None:
        class DummyTask:
            def get_configuration_object_as_dict(self, name: str):
                if name == "overrides":
                    return {"data.dataset_id": "ds2"}
                return None

        old = os.environ.get("CLEARML_TASK_ID")
        try:
            os.environ["CLEARML_TASK_ID"] = "running-task"
            with mock.patch("clearml.Task.current_task", return_value=DummyTask()):
                overrides = get_task_overrides()
            self.assertEqual(overrides, {"data.dataset_id": "ds2"})
        finally:
            if old is None:
                os.environ.pop("CLEARML_TASK_ID", None)
            else:
                os.environ["CLEARML_TASK_ID"] = old
