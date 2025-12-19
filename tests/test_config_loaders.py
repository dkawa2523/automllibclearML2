import tempfile
import unittest
from pathlib import Path

import yaml

from automl_lib.config.loaders import load_preprocessing_config


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
