import tempfile
import unittest
from pathlib import Path

import numpy as np

from automl_lib.training.reporting.plots import save_confusion_matrices, save_roc_pr_curves


class TestTrainingReportingPlots(unittest.TestCase):
    def test_save_confusion_matrices_writes_counts_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "conf"
            paths = save_confusion_matrices(
                y_true=[0, 0, 1, 1],
                y_pred=[0, 1, 1, 1],
                labels=[0, 1],
                out_dir=out_dir,
                base_name="cm",
                title_prefix="",
            )
            self.assertTrue(paths.confusion_csv and paths.confusion_csv.exists())
            self.assertTrue(paths.confusion_png and paths.confusion_png.exists())
            self.assertTrue(paths.confusion_normalized_csv and paths.confusion_normalized_csv.exists())
            self.assertTrue(paths.confusion_normalized_png and paths.confusion_normalized_png.exists())

    def test_save_roc_pr_curves_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            roc_dir = base / "roc"
            pr_dir = base / "pr"
            y_true = np.array([0, 0, 1, 1])
            scores = np.array([0.1, 0.2, 0.8, 0.7])
            paths = save_roc_pr_curves(
                y_true=y_true,
                scores=scores,
                classes=[0, 1],
                out_roc_dir=roc_dir,
                out_pr_dir=pr_dir,
                base_name="bin",
                title_prefix="",
            )
            self.assertTrue(paths.roc_png and paths.roc_png.exists())
            self.assertTrue(paths.roc_auc_csv and paths.roc_auc_csv.exists())
            self.assertTrue(paths.pr_png and paths.pr_png.exists())
            self.assertTrue(paths.pr_ap_csv and paths.pr_ap_csv.exists())

    def test_save_roc_pr_curves_multiclass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            roc_dir = base / "roc"
            pr_dir = base / "pr"
            y_true = np.array([0, 1, 2, 0, 1, 2])
            scores = np.array(
                [
                    [0.9, 0.05, 0.05],
                    [0.05, 0.9, 0.05],
                    [0.05, 0.05, 0.9],
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                ]
            )
            paths = save_roc_pr_curves(
                y_true=y_true,
                scores=scores,
                classes=[0, 1, 2],
                out_roc_dir=roc_dir,
                out_pr_dir=pr_dir,
                base_name="multi",
                title_prefix="",
            )
            self.assertTrue(paths.roc_png and paths.roc_png.exists())
            self.assertTrue(paths.roc_auc_csv and paths.roc_auc_csv.exists())
            self.assertTrue(paths.pr_png and paths.pr_png.exists())
            self.assertTrue(paths.pr_ap_csv and paths.pr_ap_csv.exists())

