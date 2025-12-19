import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automl_lib.config.schemas import TrainingConfig
from automl_lib.workflow.training.recommendation import build_recommendation_and_leaderboard


class TestTrainingRecommendation(unittest.TestCase):
    def test_build_recommendation_writes_leaderboard_and_rationale(self) -> None:
        cfg = TrainingConfig.model_validate(
            {
                "data": {"dataset_id": "dummy", "target_column": "y"},
                "models": [{"name": "ridge"}],
                "clearml": {"enabled": False},
                "output": {"output_dir": "outputs/train", "save_models": False, "generate_plots": False},
            }
        )

        df_links = pd.DataFrame(
            [
                {
                    "model": "A",
                    "preprocessor": "p1",
                    "rmse": 1.2,
                    "train_seconds": 2.0,
                    "predict_seconds": 0.1,
                    "model_size_bytes": 100,
                    "num_features": 10,
                    "task_id": "t1",
                    "model_id": "m1",
                    "status": "ok",
                    "error": "",
                    "metric_source": "test",
                },
                {
                    "model": "B",
                    "preprocessor": "p1",
                    "rmse": 0.8,
                    "train_seconds": 3.0,
                    "predict_seconds": 0.2,
                    "model_size_bytes": 200,
                    "num_features": 12,
                    "task_id": "t2",
                    "model_id": "m2",
                    "status": "ok",
                    "error": "",
                    "metric_source": "test",
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            meta = build_recommendation_and_leaderboard(
                cfg=cfg,
                output_dir=out_dir,
                df_links=df_links,
                primary_metric_model="rmse",
                problem_type="regression",
                run_id="run1",
                dataset_key="ds1",
                summary_best_model_id=None,
            )

            self.assertTrue((out_dir / "leaderboard.csv").exists())
            self.assertTrue((out_dir / "model_tasks_ranked.csv").exists())
            self.assertTrue((out_dir / "recommended_model.csv").exists())
            self.assertTrue((out_dir / "recommendation_rationale.json").exists())
            self.assertTrue((out_dir / "recommendation_rationale.md").exists())

            ranked = pd.read_csv(out_dir / "leaderboard.csv")
            self.assertIn("rank", ranked.columns)
            self.assertIn("is_recommended", ranked.columns)
            self.assertEqual(ranked.iloc[0]["model"], "B")
            self.assertTrue(bool(ranked.iloc[0]["is_recommended"]))

            recommended = pd.read_csv(out_dir / "recommended_model.csv")
            self.assertEqual(recommended.iloc[0]["model"], "B")

            rationale = json.loads((out_dir / "recommendation_rationale.json").read_text(encoding="utf-8"))
            self.assertEqual(rationale["recommend_metric"], "rmse")
            self.assertEqual(rationale["recommend_goal"], "min")

            self.assertIsNotNone(meta.get("recommended_df"))
