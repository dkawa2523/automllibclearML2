import unittest

from automl_lib.types.phase_io import InferenceInfo, TrainingInfo


class TestPhaseIO(unittest.TestCase):
    def test_default_lists_are_independent(self) -> None:
        t1 = TrainingInfo()
        t2 = TrainingInfo()
        t1.training_task_ids.append("x")
        self.assertEqual(t1.training_task_ids, ["x"])
        self.assertEqual(t2.training_task_ids, [])

        i1 = InferenceInfo()
        i2 = InferenceInfo()
        i1.child_task_ids.append("c")
        self.assertEqual(i1.child_task_ids, ["c"])
        self.assertEqual(i2.child_task_ids, [])

