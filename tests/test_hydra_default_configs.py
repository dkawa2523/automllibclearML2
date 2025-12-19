import unittest
from pathlib import Path

import yaml

from automl_lib.config.schemas import (
    DataEditingConfig,
    DataRegistrationConfig,
    InferenceConfig,
    PreprocessingConfig,
    TrainingConfig,
)


class TestHydraDefaultConfigs(unittest.TestCase):
    def _load_yaml(self, rel_path: str) -> dict:
        root = Path(__file__).resolve().parents[1]
        path = root / rel_path
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def test_training_default_valid(self) -> None:
        TrainingConfig.model_validate(self._load_yaml("conf/training/default.yaml"))

    def test_preprocessing_default_valid(self) -> None:
        PreprocessingConfig.model_validate(self._load_yaml("conf/preprocessing/default.yaml"))

    def test_inference_default_valid(self) -> None:
        payload = self._load_yaml("conf/inference/default.yaml")
        if not payload.get("model_id"):
            payload["model_id"] = "00000000000000000000000000000000"
        InferenceConfig.model_validate(payload)

    def test_data_registration_default_valid(self) -> None:
        DataRegistrationConfig.model_validate(self._load_yaml("conf/data_registration/default.yaml"))

    def test_data_editing_default_valid(self) -> None:
        DataEditingConfig.model_validate(self._load_yaml("conf/data_editing/default.yaml"))
