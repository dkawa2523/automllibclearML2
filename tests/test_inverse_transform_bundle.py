import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from automl_lib.inference.model_utils import _predict_with_model
from automl_lib.preprocessing import PreprocessingBundle


class TestInverseTransformBundle(unittest.TestCase):
    def test_predict_with_model_applies_bundle_inverse_transform(self) -> None:
        X = pd.DataFrame({"x": np.linspace(0.0, 10.0, 50)})
        y = 3.0 * X["x"].to_numpy() + 5.0

        scaler = StandardScaler().fit(y.reshape(-1, 1))
        y_scaled = scaler.transform(y.reshape(-1, 1)).reshape(-1)

        model = LinearRegression().fit(X[["x"]], y_scaled)
        bundle = PreprocessingBundle(feature_transformer=FunctionTransformer(validate=False), target_transformer=scaler)
        pipeline = Pipeline([("preprocessor", bundle), ("model", model)])

        preds = _predict_with_model(pipeline, X[["x"]])
        self.assertTrue(np.max(np.abs(preds - y)) < 1e-6)

    def test_predict_with_model_does_not_double_inverse_transform(self) -> None:
        X = pd.DataFrame({"x": np.linspace(0.0, 10.0, 50)})
        y = 3.0 * X["x"].to_numpy() + 5.0

        scaler = StandardScaler()
        model = TransformedTargetRegressor(regressor=LinearRegression(), transformer=scaler).fit(X[["x"]], y)

        # Bundle has a fitted target transformer, but model already inverse-transforms.
        fitted_scaler = StandardScaler().fit(y.reshape(-1, 1))
        bundle = PreprocessingBundle(
            feature_transformer=FunctionTransformer(validate=False),
            target_transformer=fitted_scaler,
        )
        pipeline = Pipeline([("preprocessor", bundle), ("model", model)])

        preds = _predict_with_model(pipeline, X[["x"]])
        self.assertTrue(np.max(np.abs(preds - y)) < 1e-6)

    def test_preprocessing_processing_writes_fitted_target_transformer_bundle(self) -> None:
        from automl_lib.workflow.preprocessing import processing as mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            out_dir = tmp / "out"
            cfg_path = tmp / "config_preprocessing.yaml"
            cfg_path.write_text(
                yaml.safe_dump(
                    {
                        "data": {"dataset_id": "ds_raw", "target_column": "y", "feature_columns": ["x"]},
                        "preprocessing": {"target_standardize": True},
                        "output": {"output_dir": str(out_dir)},
                        "clearml": {"enabled": True, "enable_preprocessing": True},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            df_raw = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 11.0, 12.0]})

            with (
                mock.patch.object(mod, "dataframe_from_dataset", return_value=df_raw),
                mock.patch.object(mod, "register_dataset_from_path", return_value="ds_preprocessed"),
                mock.patch.object(
                    mod,
                    "create_preprocessing_task",
                    return_value={"task": None, "logger": None, "project": "AutoML", "queue": None},
                ),
                mock.patch.object(mod, "upload_artifacts", return_value=None),
                mock.patch.object(mod, "report_scalar", return_value=None),
                mock.patch.object(mod, "set_user_properties", return_value=None),
            ):
                mod.run_preprocessing_processing(
                    cfg_path,
                    input_info={"dataset_id": "ds_raw", "task_id": "t0"},
                    run_id="20251217-000000-abcdef",
                )

            bundle_path = out_dir / "20251217-000000-abcdef" / "dataset" / "preprocessing" / "bundle.joblib"
            self.assertTrue(bundle_path.exists())
            bundle = joblib.load(bundle_path)
            self.assertIsInstance(bundle, PreprocessingBundle)
            self.assertIsNotNone(bundle.target_transformer)
            self.assertTrue(hasattr(bundle.target_transformer, "mean_"))

