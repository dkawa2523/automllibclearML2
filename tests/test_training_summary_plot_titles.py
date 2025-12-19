import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class _FakeLogger:
    def __init__(self) -> None:
        self.titles: list[str] = []

    def report_table(self, *, title: str, **_kwargs) -> None:
        self.titles.append(str(title))

    def report_plotly(self, *, title: str, **_kwargs) -> None:
        self.titles.append(str(title))

    def report_image(self, *, title: str, **_kwargs) -> None:
        self.titles.append(str(title))


class _FakeTask:
    def __init__(self, task_id: str) -> None:
        self.id = task_id

    def flush(self, **_kwargs) -> None:
        return

    def close(self) -> None:
        return


class _FakeClearMLManager:
    created: list["_FakeClearMLManager"] = []

    def __init__(
        self,
        _cfg,
        *,
        task_name: str,
        task_type: str,
        default_project: str = "AutoML",
        project: str | None = None,
        parent: str | None = None,
        existing_task=None,
        extra_tags=None,
    ) -> None:
        self.cfg = _cfg
        self.enabled = True
        self.task_name = task_name
        self.task_type = task_type
        self.default_project = default_project
        self.project = project
        self.parent = parent
        self.existing_task = existing_task
        self.extra_tags = extra_tags
        self.task = _FakeTask(task_id=f"fake:{task_name}")
        self.logger = _FakeLogger()
        _FakeClearMLManager.created.append(self)

    def connect_configuration(self, _obj, *, name: str = "config") -> None:
        return

    def connect_params_sections(self, _sections) -> None:
        return

    def upload_artifacts(self, _paths) -> None:
        return

    def report_table(self, title: str, _df, *, series: str = "table", iteration: int = 0) -> None:
        _ = series, iteration
        self.logger.report_table(title=title)

    def close(self) -> None:
        return


class TestTrainingSummaryPlotTitles(unittest.TestCase):
    def test_training_summary_logs_only_expected_titles(self) -> None:
        old_offline = os.environ.get("CLEARML_OFFLINE_MODE")
        os.environ["CLEARML_OFFLINE_MODE"] = "1"
        try:
            from automl_lib.workflow.training import model_tasks as model_tasks_mod
            from automl_lib.workflow.training import runner as runner_mod

            orig_mgr = runner_mod.ClearMLManager
            orig_import_clearml = model_tasks_mod._import_clearml
            try:
                _FakeClearMLManager.created.clear()
                runner_mod.ClearMLManager = _FakeClearMLManager
                # Avoid creating real ClearML child tasks during this unit test.
                model_tasks_mod._import_clearml = lambda: (None, None, None, None)

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
                        "data": {"csv_path": str(csv_path), "target_column": "y", "test_size": 0.2, "random_seed": 0},
                        "models": [{"name": "ridge"}],
                        "cross_validation": {"n_folds": 2, "shuffle": True, "random_seed": 0},
                        "output": {"output_dir": str(out_dir), "save_models": False, "generate_plots": True},
                        "interpretation": {"compute_feature_importance": True, "compute_shap": False},
                        "visualizations": {"predicted_vs_actual": True},
                        "clearml": {"enabled": True, "summary_plots": "best"},
                    }
                    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

                    runner_mod.run_automl(cfg_path)
                    self.assertGreaterEqual(len(_FakeClearMLManager.created), 1)
                    mgr = _FakeClearMLManager.created[0]
                    titles = mgr.logger.titles

                    allowed_exact = {
                        "01_Recommended Model",
                        "02_Leaderboard",
                        "03_Leaderboard Table",
                        "04_Tradeoff",
                        "05_Scatter Plot of Recommended Model",
                        "06_Feature Importance from Recommended Model",
                        "08_SHAP values",
                    }

                    for t in titles:
                        if str(t).startswith("07_Interpolation space:"):
                            continue
                        self.assertIn(t, allowed_exact)
                    self.assertIn("05_Scatter Plot of Recommended Model", titles)
                    self.assertTrue(any(str(t).startswith("07_Interpolation space:") for t in titles))
            finally:
                runner_mod.ClearMLManager = orig_mgr
                model_tasks_mod._import_clearml = orig_import_clearml
        finally:
            if old_offline is None:
                os.environ.pop("CLEARML_OFFLINE_MODE", None)
            else:
                os.environ["CLEARML_OFFLINE_MODE"] = old_offline
