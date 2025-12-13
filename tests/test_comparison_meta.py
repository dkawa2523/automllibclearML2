import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automl_lib.phases.comparison.meta import build_comparison_metadata


class TestComparisonMeta(unittest.TestCase):
    def test_build_comparison_metadata_multi_run_outputs(self) -> None:
        rows = [
            {"run_id": "run1", "model": "A", "rmse": 1.0},
            {"run_id": "run1", "model": "B", "rmse": 2.0},
            {"run_id": "run2", "model": "A", "rmse": 1.5},
            {"run_id": "run2", "model": "B", "rmse": 0.8},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            meta = build_comparison_metadata(
                rows,
                output_dir=out_dir,
                metric_cols=["rmse"],
                primary_metric="rmse",
                group_col="run_id",
            )

            artifacts = set(meta.get("artifacts") or [])
            expected = {
                str(out_dir / "comparison_metrics.csv"),
                str(out_dir / "comparison_ranked.csv"),
                str(out_dir / "best_result.json"),
                str(out_dir / "best_by_model.csv"),
                str(out_dir / "best_by_model.json"),
                str(out_dir / "best_by_run.csv"),
                str(out_dir / "best_by_run.json"),
                str(out_dir / "best_by_run_model.csv"),
                str(out_dir / "model_summary.csv"),
                str(out_dir / "win_summary.csv"),
                str(out_dir / "recommended_model.json"),
            }
            self.assertTrue(expected.issubset(artifacts))
            for path in expected:
                self.assertTrue(Path(path).exists(), path)

            best = json.loads((out_dir / "best_result.json").read_text(encoding="utf-8"))
            self.assertEqual(best["primary_metric"], "rmse")
            self.assertEqual(best["goal"], "min")
            self.assertAlmostEqual(float(best["best_row"]["rmse"]), 0.8)

            best_by_run = pd.read_csv(out_dir / "best_by_run.csv")
            self.assertEqual(set(best_by_run["run_id"].tolist()), {"run1", "run2"})
            best_by_run = best_by_run.set_index("run_id")
            self.assertEqual(best_by_run.loc["run1", "model"], "A")
            self.assertAlmostEqual(float(best_by_run.loc["run1", "rmse"]), 1.0)
            self.assertEqual(best_by_run.loc["run2", "model"], "B")
            self.assertAlmostEqual(float(best_by_run.loc["run2", "rmse"]), 0.8)

            summary = pd.read_csv(out_dir / "model_summary.csv")
            self.assertIn("model", summary.columns)
            self.assertIn("n_runs", summary.columns)
            self.assertIn("rmse_mean", summary.columns)
            self.assertEqual(set(summary["model"].tolist()), {"A", "B"})
            self.assertEqual(summary.iloc[0]["model"], "A")

    def test_composite_score_with_weights(self) -> None:
        rows = [
            {"model": "A", "rmse": 1.0, "r2": 0.8},
            {"model": "B", "rmse": 2.0, "r2": 0.9},
        ]

        meta = build_comparison_metadata(
            rows,
            metric_cols=["rmse", "r2"],
            primary_metric="composite_score",
            composite_enabled=True,
            composite_weights={"rmse": 2.0, "r2": 1.0},
        )

        df = meta["df"]
        self.assertIn("composite_score", df.columns)
        ranked = meta["ranked_df"]
        self.assertEqual(ranked.iloc[0]["model"], "A")

    def test_composite_score_require_all_metrics(self) -> None:
        rows = [
            {"model": "A", "rmse": 1.0, "r2": None},
            {"model": "B", "rmse": 2.0, "r2": 0.9},
        ]

        meta = build_comparison_metadata(
            rows,
            metric_cols=["rmse", "r2"],
            primary_metric="composite_score",
            composite_enabled=True,
            composite_require_all=True,
        )

        df = meta["df"]
        self.assertIn("composite_score", df.columns)
        values = pd.to_numeric(df["composite_score"], errors="coerce")
        self.assertTrue(values.isna().iloc[0])
        self.assertFalse(values.isna().iloc[1])

    def test_ranked_topk_artifact(self) -> None:
        rows = [
            {"model": "A", "rmse": 1.0},
            {"model": "B", "rmse": 2.0},
            {"model": "C", "rmse": 0.5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            meta = build_comparison_metadata(
                rows,
                output_dir=out_dir,
                metric_cols=["rmse"],
                primary_metric="rmse",
                top_k=2,
            )
            path = out_dir / "comparison_ranked_topk.csv"
            self.assertIn(str(path), set(meta.get("artifacts") or []))
            df = pd.read_csv(path)
            self.assertLessEqual(len(df), 2)
