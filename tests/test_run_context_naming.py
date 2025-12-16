import unittest

from automl_lib.clearml.context import build_run_context, generate_run_id, resolve_dataset_key
from automl_lib.clearml.naming import build_tags, dataset_name, task_name


class TestRunContextAndNaming(unittest.TestCase):
    def test_generate_run_id_format(self) -> None:
        run_id = generate_run_id()
        self.assertRegex(run_id, r"^\d{8}-\d{6}-[0-9a-f]{6}$")

    def test_resolve_dataset_key_prefers_dataset_id_short(self) -> None:
        key = resolve_dataset_key(dataset_id="20f90f6817834c609dcb35be5cadf17c")
        self.assertEqual(key, "20f90f68")

    def test_resolve_dataset_key_falls_back_to_csv_stem(self) -> None:
        key = resolve_dataset_key(csv_path="data/example.csv")
        self.assertEqual(key, "example")

    def test_build_tags_contains_base_and_phase(self) -> None:
        ctx = build_run_context(
            run_id="20251213-153012-a1b2c3",
            dataset_key="example",
            project_root="AutoML",
            dataset_project="datasets",
            user="tester",
        )
        tags = build_tags(ctx, phase="preprocessing", extra=["automl", "automl"])
        self.assertIn("run:20251213-153012-a1b2c3", tags)
        self.assertIn("dataset:example", tags)
        self.assertIn("phase:preprocessing", tags)
        self.assertEqual(tags.count("automl"), 1)

    def test_task_and_dataset_name_include_context(self) -> None:
        ctx = build_run_context(
            run_id="20251213-153012-a1b2c3",
            dataset_key="example",
            project_root="AutoML",
            dataset_project="datasets",
            user="tester",
        )
        self.assertEqual(task_name("preprocessing", ctx), "preprocessing ds:example run:20251213-153012-a1b2c3")
        self.assertIn("ds:example", dataset_name("preprocessed", ctx, preproc="standard"))
