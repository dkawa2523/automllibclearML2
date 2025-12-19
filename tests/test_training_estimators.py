import unittest

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression

from automl_lib.config.schemas import TrainingConfig
from automl_lib.workflow.training.estimators import maybe_wrap_with_target_scaler


class TestTrainingEstimators(unittest.TestCase):
    def _cfg(self, *, target_standardize: bool) -> TrainingConfig:
        return TrainingConfig.model_validate(
            {
                "data": {"dataset_id": "dummy", "target_column": "y"},
                "models": [{"name": "ridge"}],
                "preprocessing": {"target_standardize": target_standardize},
                "clearml": {"enabled": False},
                "output": {"output_dir": "outputs/train", "save_models": False, "generate_plots": False},
            }
        )

    def test_maybe_wrap_with_target_scaler_regression(self) -> None:
        cfg = self._cfg(target_standardize=True)
        est = LinearRegression()
        wrapped = maybe_wrap_with_target_scaler(est, cfg, "regression")
        self.assertIsInstance(wrapped, TransformedTargetRegressor)

    def test_maybe_wrap_with_target_scaler_noop_when_disabled(self) -> None:
        cfg = self._cfg(target_standardize=False)
        est = LinearRegression()
        wrapped = maybe_wrap_with_target_scaler(est, cfg, "regression")
        self.assertIs(wrapped, est)

    def test_maybe_wrap_with_target_scaler_noop_for_classification(self) -> None:
        cfg = self._cfg(target_standardize=True)
        est = LinearRegression()
        wrapped = maybe_wrap_with_target_scaler(est, cfg, "classification")
        self.assertIs(wrapped, est)

