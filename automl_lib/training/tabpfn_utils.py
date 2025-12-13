"""Utilities for working with TabPFN in offline environments."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any, Optional

try:
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    warnings.warn(f"PyTorch not available, TabPFN fallback disabled: {exc}")
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array

try:
    from tabpfn.architectures.base import get_architecture
    from tabpfn.architectures.base.config import ModelConfig
    from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution
    from tabpfn.base import RegressorModelSpecs
except Exception as exc:  # pragma: no cover - tabpfn may be missing
    warnings.warn(f"TabPFN modules could not be imported: {exc}")
    RegressorModelSpecs = None  # type: ignore


@lru_cache(maxsize=1)
def build_fallback_regressor_spec() -> Optional[RegressorModelSpecs]:
    """Create a lightweight TabPFN regressor spec with random weights.

    This is used when pretrained checkpoints are unavailable. The resulting
    model is not pre-trained but allows the pipeline to execute end-to-end.
    """
    if RegressorModelSpecs is None or torch is None:
        return None
    try:
        num_bars = 256
        config = ModelConfig(max_num_classes=1, num_buckets=num_bars)
        num_bars = 256
        bar_limit = 1.0e6
        edges = torch.linspace(-bar_limit, bar_limit, steps=num_bars + 1, dtype=torch.float32)
        architecture = get_architecture(
            config,
            n_out=num_bars,
            cache_trainset_representation=False,
        )
        architecture.eval()
        bar_distribution = FullSupportBarDistribution(edges)
        return RegressorModelSpecs(
            model=architecture,
            config=config,
            norm_criterion=bar_distribution,
        )
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Failed to build fallback TabPFN spec: {exc}")
        return None


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
