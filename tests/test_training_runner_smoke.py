import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class TestTrainingRunnerSmoke(unittest.TestCase):
    def test_run_automl_smoke_without_clearml(self) -> None:
        old_offline = os.environ.get("CLEARML_OFFLINE_MODE")
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
        try:
            from automl_lib.workflow.training.runner import run_automl

            with tempfile.TemporaryDirectory() as tmpdir:
                base = Path(tmpdir)
                csv_path = base / "data.csv"
                out_dir = base / "out"
                out_dir.mkdir(parents=True, exist_ok=True)

                rng = np.random.default_rng(0)
                n = 30
                x1 = rng.normal(size=n)
                cat = np.where(x1 > 0, "pos", "neg")
                y = 2.0 * x1 + rng.normal(scale=0.1, size=n)
                pd.DataFrame({"x1": x1, "cat": cat, "y": y}).to_csv(csv_path, index=False)

                cfg_path = base / "config_training.yaml"
                cfg = {
                    "run": {"id": "testrun"},
                    "data": {"csv_path": str(csv_path), "target_column": "y"},
                    "models": [{"name": "ridge"}],
                    "cross_validation": {"n_folds": 2, "shuffle": True, "random_seed": 0},
                    "output": {"output_dir": str(out_dir), "save_models": False, "generate_plots": False},
                    "clearml": {"enabled": False},
                }
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

                result = run_automl(cfg_path)
                self.assertIsInstance(result, dict)
                self.assertIn("summary_task_id", result)
                self.assertIn("training_task_ids", result)
                self.assertIn("metrics", result)

                run_out = out_dir / "testrun"
                self.assertTrue((run_out / "results_summary.csv").exists())
                self.assertTrue((run_out / "leaderboard.csv").exists())
                self.assertTrue((run_out / "recommended_model.csv").exists())
        finally:
            if old_offline is None:
                os.environ.pop("CLEARML_OFFLINE_MODE", None)
            else:
                os.environ["CLEARML_OFFLINE_MODE"] = old_offline
