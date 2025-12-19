"""Utilities for working with TabPFN in offline environments.

NOTE:
This project supports TabPFN as an optional model. In some environments, importing
or initializing TabPFN/torch internals can lead to instability (including hard
crashes). To keep the AutoML pipeline robust, we provide a lightweight fallback
estimator that does not depend on TabPFN/torch.
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array


class OfflineTabPFNRegressor(BaseEstimator, RegressorMixin):
    """Fallback regressor used when pretrained TabPFN weights are unavailable."""

    def __init__(
        self,
        *,
        base_estimator: Optional[RegressorMixin] = None,
        random_state: int | None = 0,
        n_estimators: int = 200,
        **_: Any,
    ) -> None:
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._model: Optional[RegressorMixin] = None

    def fit(self, X, y):  # type: ignore[override]
        X_val, y_val = check_X_y(X, y, accept_sparse=False, ensure_2d=True)
        if self.base_estimator is not None:
            model = clone(self.base_estimator)
        else:
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        model.fit(X_val, y_val)
        self._model = model
        self.is_fitted_ = True
        self.n_features_in_ = X_val.shape[1]
        if hasattr(model, "n_outputs_"):
            self.n_outputs_ = model.n_outputs_
        return self

    def predict(self, X):  # type: ignore[override]
        if self._model is None:
            raise RuntimeError("OfflineTabPFNRegressor is not fitted")
        X_val = check_array(X, accept_sparse=False, ensure_2d=True)
        return self._model.predict(X_val)
