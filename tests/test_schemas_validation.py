import unittest

from pydantic import ValidationError

from automl_lib.config.schemas import DataEditingConfig, DataRegistrationConfig, DataSettings, ModelSpec, TrainingConfig


class TestSchemasValidation(unittest.TestCase):
    def test_training_config_requires_data_source(self) -> None:
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate(
                {
                    "data": {},
                    "models": [{"name": "ridge"}],
                }
            )

        cfg = TrainingConfig.model_validate(
            {
                "data": {"dataset_id": "dummy"},
                "models": [{"name": "ridge"}],
            }
        )
        self.assertEqual(cfg.data.dataset_id, "dummy")

    def test_training_config_requires_models(self) -> None:
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate(
                {
                    "data": {"dataset_id": "dummy"},
                    "models": [],
                }
            )

    def test_training_config_rejects_unknown_model(self) -> None:
        with self.assertRaises(ValidationError):
            TrainingConfig.model_validate(
                {
                    "data": {"dataset_id": "dummy"},
                    "models": [{"name": "this_model_does_not_exist"}],
                }
            )

    def test_training_config_accepts_model_alias(self) -> None:
        cfg = TrainingConfig.model_validate(
            {
                "data": {"dataset_id": "dummy"},
                "models": [{"name": "KNN"}],
            }
        )
        self.assertTrue(cfg.models)

    def test_data_registration_requires_csv_path(self) -> None:
        with self.assertRaises(ValidationError):
            DataRegistrationConfig.model_validate({"data": {"dataset_id": "dummy"}})

    def test_data_editing_requires_csv_or_dataset(self) -> None:
        with self.assertRaises(ValidationError):
            DataEditingConfig.model_validate({"data": {}, "editing": {"drop_columns": []}})

    def test_data_settings_problem_type_normalization(self) -> None:
        s1 = DataSettings.model_validate({"problem_type": "Regression"})
        self.assertEqual(s1.problem_type, "regression")
        s2 = DataSettings.model_validate({"problem_type": "none"})
        self.assertIsNone(s2.problem_type)
        with self.assertRaises(ValidationError):
            DataSettings.model_validate({"problem_type": "unsupported"})

    def test_hidden_layer_sizes_normalization(self) -> None:
        spec = ModelSpec.model_validate({"name": "mlp", "params": {"hidden_layer_sizes": "(50, 100)"}})
        self.assertEqual(spec.params["hidden_layer_sizes"], [(50, 100)])


