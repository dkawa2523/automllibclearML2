import tempfile
import unittest
from pathlib import Path


class TestDatasetCsvSelection(unittest.TestCase):
    def test_find_first_csv_prefers_processed(self) -> None:
        from automl_lib.integrations.clearml.datasets import find_first_csv as find_first_csv_clearml
        from automl_lib.training.clearml_integration import find_first_csv as find_first_csv_training

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "data_processed.csv").write_text("x,y\n3,4\n", encoding="utf-8")
            self.assertEqual(find_first_csv_clearml(root), root / "data_processed.csv")
            self.assertEqual(find_first_csv_training(root), root / "data_processed.csv")

    def test_find_first_csv_prefers_processed_in_subdir(self) -> None:
        from automl_lib.integrations.clearml.datasets import find_first_csv as find_first_csv_clearml
        from automl_lib.training.clearml_integration import find_first_csv as find_first_csv_training

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "raw.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "child").mkdir(parents=True, exist_ok=True)
            (root / "child" / "data_processed.csv").write_text("x,y\n3,4\n", encoding="utf-8")
            self.assertEqual(find_first_csv_clearml(root), root / "child" / "data_processed.csv")
            self.assertEqual(find_first_csv_training(root), root / "child" / "data_processed.csv")


if __name__ == "__main__":
    unittest.main()
