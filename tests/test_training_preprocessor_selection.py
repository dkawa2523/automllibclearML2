import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from automl_lib.config.schemas import TrainingConfig
from automl_lib.workflow.training.preprocessor_selection import select_preprocessors


class TestTrainingPreprocessorSelection(unittest.TestCase):
    def _base_cfg(self) -> TrainingConfig:
        return TrainingConfig.model_validate(
            {
                "data": {"dataset_id": "dummy", "target_column": "y"},
                "models": [{"name": "ridge"}],
                "clearml": {"enabled": False},
                "output": {"output_dir": "outputs/train", "save_models": False, "generate_plots": False},
            }
        )

    def test_contract_with_categorical_dtype_uses_configured_preprocessors(self) -> None:
        cfg = self._base_cfg()
        X_train = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0],
                "cat": pd.Series(["A", "B", "A", "C"], dtype="category"),
            }
        )
        feature_types = {"numeric": ["num"], "categorical": ["cat"]}

        preprocessors = select_preprocessors(
            cfg=cfg,
            feature_types=feature_types,
            X_train=X_train,
            has_preproc_contract=True,
            preproc_manifest_src=None,
        )

        self.assertTrue(preprocessors)
        name, transformer = preprocessors[0]
        self.assertIsInstance(transformer, ColumnTransformer)
        self.assertNotEqual(name, "preprocessed_dataset")

    def test_contract_with_numeric_only_uses_identity_transformer_with_manifest_label(self) -> None:
        cfg = self._base_cfg()
        X_train = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
        feature_types = {"numeric": ["num"], "categorical": []}

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{"selected_preprocessor": "v1"}', encoding="utf-8")

            preprocessors = select_preprocessors(
                cfg=cfg,
                feature_types=feature_types,
                X_train=X_train,
                has_preproc_contract=True,
                preproc_manifest_src=manifest_path,
            )

        self.assertTrue(preprocessors)
        name, transformer = preprocessors[0]
        self.assertTrue(name.startswith("preprocessed|v1"))
        self.assertIsInstance(transformer, FunctionTransformer)

