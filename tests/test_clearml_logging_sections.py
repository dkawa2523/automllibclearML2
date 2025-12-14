import unittest

from automl_lib.clearml.logging import report_hyperparams_sections


class _DummyTask:
    def __init__(self) -> None:
        self.calls = []

    def connect(self, payload, name=None):  # mimic ClearML Task.connect
        self.calls.append((name, payload))
        return payload


class TestClearMLLoggingSections(unittest.TestCase):
    def test_report_hyperparams_sections_calls_connect_per_section(self) -> None:
        task = _DummyTask()
        report_hyperparams_sections(
            task,
            {
                "Run": {"id": "r1"},
                "Empty": {},
                "Data": {"dataset_id": "ds1"},
            },
        )

        self.assertEqual([name for name, _ in task.calls], ["Run", "Data"])
        self.assertEqual(task.calls[0][1]["id"], "r1")
        self.assertEqual(task.calls[1][1]["dataset_id"], "ds1")

