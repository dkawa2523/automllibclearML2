import tempfile
import unittest
from pathlib import Path

import yaml

from automl_lib.config.loaders import load_comparison_config, load_preprocessing_config


class TestConfigLoaders(unittest.TestCase):
    def test_preprocessing_loader_ignores_training_output_dir(self) -> None:
        raw = {
            "data": {"dataset_id": "dummy"},
            "models": [{"name": "ridge"}],
            "output": {"output_dir": "outputs/train_custom"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

            cfg = load_preprocessing_config(path)
            self.assertEqual(cfg.output.output_dir, "outputs/preprocessing")

    def test_preprocessing_loader_honors_phase_output_dir(self) -> None:
        raw = {
            "data": {"dataset_id": "dummy"},
            "preprocessing": {"scaling": ["standard"]},
            "output": {"output_dir": "outputs/my_preproc"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config_preprocessing.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

            cfg = load_preprocessing_config(path)
            self.assertEqual(cfg.output.output_dir, "outputs/my_preproc")

    def test_comparison_loader_ignores_training_output_dir(self) -> None:
        raw = {
            "data": {"dataset_id": "dummy"},
            "models": [{"name": "ridge"}],
            "output": {"output_dir": "outputs/train_custom"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

            cfg = load_comparison_config(path)
            self.assertEqual(cfg.output.output_dir, "outputs/comparison")

    def test_comparison_loader_honors_phase_output_dir(self) -> None:
        raw = {
            "output": {"output_dir": "outputs/my_comparison"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config_comparison.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

            cfg = load_comparison_config(path)
            self.assertEqual(cfg.output.output_dir, "outputs/my_comparison")

    def test_comparison_loader_extracts_primary_metric_from_training_config(self) -> None:
        raw = {
            "data": {"dataset_id": "dummy"},
            "models": [{"name": "ridge"}],
            "evaluation": {"primary_metric": "rmse"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

            cfg = load_comparison_config(path)
            self.assertEqual(cfg.ranking.primary_metric, "rmse")

