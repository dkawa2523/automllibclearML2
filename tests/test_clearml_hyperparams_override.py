import unittest


from automl_lib.integrations.clearml.hyperparams import apply_preprocessing_hyperparams, apply_training_hyperparams


class TestClearMLHyperparamsOverride(unittest.TestCase):
    def test_apply_preprocessing_hyperparams_overrides_sections(self) -> None:
        cfg = {
            "data": {"dataset_id": "ds_old", "csv_path": "a.csv", "target_column": "y"},
            "preprocessing": {"numeric_imputation": ["mean"], "target_standardize": False},
        }
        params = {
            "Input": {"dataset_id": "ds_new", "csv_path": "", "target_column": "target"},
            "Preprocessing": {
                "numeric_imputation": ["median"],
                "categorical_imputation": ["most_frequent"],
                "target_standardize": True,
            },
        }
        out = apply_preprocessing_hyperparams(cfg, params)
        self.assertEqual(out["data"]["dataset_id"], "ds_new")
        self.assertIsNone(out["data"]["csv_path"])
        self.assertEqual(out["data"]["target_column"], "target")
        self.assertEqual(out["preprocessing"]["numeric_imputation"], ["median"])
        self.assertTrue(out["preprocessing"]["target_standardize"])

    def test_apply_training_hyperparams_overrides_sections(self) -> None:
        cfg = {
            "data": {"dataset_id": None, "csv_path": "data.csv", "target_column": "y", "test_size": 0.2, "random_seed": 1},
            "models": [{"name": "A", "params": {}}],
            "ensembles": {"stacking": {"enable": False}, "voting": {"enable": False}},
            "cross_validation": {"n_folds": 5, "shuffle": True, "random_seed": 42},
            "evaluation": {"regression_metrics": ["mae"], "classification_metrics": ["accuracy"], "primary_metric": None},
            "optimization": {"method": "grid", "n_iter": 10},
        }
        params = {
            "Training": {"dataset_id": "ds1", "target_column": "", "test_size": 0.0, "random_seed": 123, "cv_folds": "auto", "cv_shuffle": False},
            "Models": {"models": [{"name": "B", "enable": True, "params": {"alpha": [0.1, 1.0]}}]},
            "CrossValidation": {"n_folds": 3, "shuffle": False, "random_seed": 999},
            "Evaluation": {"regression_metrics": ["rmse"], "classification_metrics": ["f1_macro"], "primary_metric": None, "plugins": []},
            "Optimization": {"method": "random", "n_iter": 20},
        }
        out = apply_training_hyperparams(cfg, params)
        self.assertEqual(out["data"]["dataset_id"], "ds1")
        self.assertIsNone(out["data"]["target_column"])
        self.assertEqual(out["data"]["test_size"], 0.0)
        self.assertEqual(out["data"]["random_seed"], 123)
        # Training/cv_folds='auto' maps to n_folds=None, but CrossValidation section should override after
        self.assertEqual(out["cross_validation"]["n_folds"], 3)
        self.assertFalse(out["cross_validation"]["shuffle"])
        self.assertEqual(out["models"][0]["name"], "B")
        self.assertEqual(out["optimization"]["method"], "random")
        self.assertEqual(out["optimization"]["n_iter"], 20)

