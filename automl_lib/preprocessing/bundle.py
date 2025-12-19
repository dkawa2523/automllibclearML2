from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
except Exception:  # pragma: no cover
    BaseEstimator = object  # type: ignore
    TransformerMixin = object  # type: ignore


class PreprocessingBundle(BaseEstimator, TransformerMixin):
    """Dataset contract bundle (v1).

    This object is stored into ClearML Dataset artifacts (`preprocessing/bundle.joblib`).
    It behaves like a scikit-learn transformer for feature preprocessing and additionally
    holds an (optional) target transformer template for training/inference consistency.
    """

    def __init__(
        self,
        feature_transformer: Any,
        *,
        target_transformer: Optional[Any] = None,
        contract_version: str = "v1",
    ) -> None:
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer
        self.contract_version = str(contract_version or "v1")

    def fit(self, X, y=None):  # noqa: N803 (sklearn signature)
        if hasattr(self.feature_transformer, "fit"):
            try:
                self.feature_transformer.fit(X, y)
            except TypeError:
                self.feature_transformer.fit(X)
        if self.target_transformer is not None and y is not None and hasattr(self.target_transformer, "fit"):
            y_arr = np.asarray(y).reshape(-1, 1)
            self.target_transformer.fit(y_arr)
        return self

    def transform(self, X):  # noqa: N803 (sklearn signature)
        if hasattr(self.feature_transformer, "transform"):
            return self.feature_transformer.transform(X)
        # Fallback: callable transformer
        return self.feature_transformer(X)

    def fit_transform(self, X, y=None, **fit_params):  # noqa: N803 (sklearn signature)
        if hasattr(self.feature_transformer, "fit_transform"):
            try:
                return self.feature_transformer.fit_transform(X, y, **fit_params)
            except TypeError:
                return self.feature_transformer.fit_transform(X, **fit_params)
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001 (sklearn API)
        if hasattr(self.feature_transformer, "get_feature_names_out"):
            return self.feature_transformer.get_feature_names_out(input_features)
        raise AttributeError("feature_transformer does not expose get_feature_names_out")

    def transform_target(self, y):
        if self.target_transformer is None:
            return np.asarray(y)
        y_arr = np.asarray(y).reshape(-1, 1)
        out = self.target_transformer.transform(y_arr)
        return np.asarray(out).reshape(-1)

    def inverse_transform_target(self, y):
        if self.target_transformer is None:
            return np.asarray(y)
        y_arr = np.asarray(y).reshape(-1, 1)
        out = self.target_transformer.inverse_transform(y_arr)
        return np.asarray(out).reshape(-1)

    def __getattr__(self, name: str):
        # Delegate optional sklearn APIs (e.g., set_output) to the underlying transformer.
        try:
            return getattr(self.feature_transformer, name)
        except Exception as exc:
            raise AttributeError(name) from exc

