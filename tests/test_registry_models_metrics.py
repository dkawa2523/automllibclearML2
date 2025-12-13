import unittest

from automl_lib.registry.metrics import add_derived_metrics, build_sklearn_scoring, get_metric_spec
from automl_lib.registry.models import list_models, resolve_model_class


class TestRegistry(unittest.TestCase):
    def test_metrics_registry_defaults(self) -> None:
        spec = get_metric_spec("rmse", problem_type="regression")
        self.assertEqual(spec.derived_from, "mse")
        self.assertTrue(spec.is_loss)

        scoring = build_sklearn_scoring("regression", ["rmse", "r2"])
        self.assertIn("mse", scoring)
        self.assertIn("r2", scoring)

        result = {"mse": 4.0}
        add_derived_metrics(result, problem_type="regression", requested_metrics=["rmse"])
        self.assertIn("rmse", result)
        self.assertAlmostEqual(float(result["rmse"]), 2.0)

    def test_models_registry_defaults(self) -> None:
        cls = resolve_model_class("knn", "regression")
        self.assertEqual(cls.__name__, "KNeighborsRegressor")
        self.assertTrue(list_models())


