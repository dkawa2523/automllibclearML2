"""
Model registry
--------------
学習で使用するモデルを「名前 → (regressor/classifierクラス)」で登録する。

目的:
- `automl_lib/training/model_factory.py` からモデル解決ロジックを分離し、
  新しいモデル追加を `automl_lib/registry/models.py` の登録だけで完結させる。
- 既存の `config.yaml: models[].name` と互換のある名前解決（大小無視/空白やハイフン無視）。

使い方（例）:
    from sklearn.linear_model import HuberRegressor
    from automl_lib.registry.models import register_model

    register_model(
        "Huber",
        regressor=HuberRegressor,
        aliases=["HuberRegressor"],
    )
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence


ModelClassRef = type | str | Callable[[], type]


@dataclass(frozen=True)
class ModelEntry:
    regressor: Optional[ModelClassRef] = None
    classifier: Optional[ModelClassRef] = None


_MODELS: Dict[str, ModelEntry] = {}
_ALIASES: Dict[str, str] = {}
_DEFAULTS_REGISTERED = False


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_\\s]+", "", str(name).strip().lower())


def register_alias(alias: str, canonical_name: str) -> None:
    canonical_key = _normalize_name(canonical_name)
    _ALIASES[_normalize_name(alias)] = canonical_key


def register_model(
    name: str,
    *,
    regressor: Optional[ModelClassRef] = None,
    classifier: Optional[ModelClassRef] = None,
    aliases: Optional[Sequence[str]] = None,
) -> None:
    key = _normalize_name(name)
    _MODELS[key] = ModelEntry(regressor=regressor, classifier=classifier)
    for alias in aliases or []:
        register_alias(alias, name)


def _import_class(path: str) -> type:
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    if not isinstance(obj, type):
        raise TypeError(f"Resolved object is not a class: {path}")
    return obj


def _resolve_class(ref: ModelClassRef) -> type:
    if isinstance(ref, str):
        return _import_class(ref)
    if isinstance(ref, type):
        return ref
    resolved = ref()
    if not isinstance(resolved, type):
        raise TypeError("Model resolver must return a class")
    return resolved


def ensure_default_models_registered() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True

    # ---- scikit-learn (always available) -----------------------------------
    register_model("linearregression", regressor="sklearn.linear_model.LinearRegression")
    register_model("ridge", regressor="sklearn.linear_model.Ridge", classifier="sklearn.linear_model.RidgeClassifier")
    register_model("lasso", regressor="sklearn.linear_model.Lasso")
    register_model("elasticnet", regressor="sklearn.linear_model.ElasticNet")
    register_model("svr", regressor="sklearn.svm.SVR")
    register_model("kneighbors", regressor="sklearn.neighbors.KNeighborsRegressor", classifier="sklearn.neighbors.KNeighborsClassifier")
    register_alias("knn", "kneighbors")

    register_model(
        "randomforest",
        regressor="sklearn.ensemble.RandomForestRegressor",
        classifier="sklearn.ensemble.RandomForestClassifier",
    )
    register_model(
        "extratrees",
        regressor="sklearn.ensemble.ExtraTreesRegressor",
        classifier="sklearn.ensemble.ExtraTreesClassifier",
    )
    register_model(
        "gradientboosting",
        regressor="sklearn.ensemble.GradientBoostingRegressor",
        classifier="sklearn.ensemble.GradientBoostingClassifier",
    )
    register_model(
        "gaussianprocess",
        regressor="sklearn.gaussian_process.GaussianProcessRegressor",
        classifier="sklearn.gaussian_process.GaussianProcessClassifier",
    )
    register_model("mlp", regressor="sklearn.neural_network.MLPRegressor", classifier="sklearn.neural_network.MLPClassifier")

    # classification-specific aliases
    register_model("logisticregression", classifier="sklearn.linear_model.LogisticRegression")
    register_model("svc", classifier="sklearn.svm.SVC")
    register_alias("svm", "svc")

    # ---- optional third-party models --------------------------------------
    try:
        import lightgbm  # type: ignore  # noqa: F401

        register_model("lightgbm", regressor="lightgbm.LGBMRegressor", classifier="lightgbm.LGBMClassifier")
    except Exception:
        pass

    try:
        import xgboost  # type: ignore  # noqa: F401

        register_model("xgboost", regressor="xgboost.XGBRegressor", classifier="xgboost.XGBClassifier")
    except Exception:
        pass

    try:
        import catboost  # type: ignore  # noqa: F401

        register_model("catboost", regressor="catboost.CatBoostRegressor", classifier="catboost.CatBoostClassifier")
    except Exception:
        pass

    # TabNet (patched for tiny datasets / 1d targets)
    try:
        import numpy as np
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier  # type: ignore

        class _TabNetMixin:
            @staticmethod
            def _n_samples(X_train: Any) -> int:
                if hasattr(X_train, "shape") and X_train.shape is not None and len(X_train.shape) > 0:
                    return int(X_train.shape[0])
                return len(X_train)

            @classmethod
            def _normalize_train_kwargs(cls, n_samples: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
                params = dict(kwargs)
                batch_size = params.get("batch_size")
                if batch_size is None:
                    suggested = max(8, n_samples // 2)
                    batch_size = min(256, suggested)
                batch_size = max(1, min(batch_size, n_samples))
                params["batch_size"] = batch_size

                virtual_bs = params.get("virtual_batch_size")
                if virtual_bs is None:
                    half = batch_size // 2 if batch_size > 1 else 1
                    virtual_bs = max(1, min(32, half))
                virtual_bs = max(1, min(virtual_bs, batch_size))
                params["virtual_batch_size"] = virtual_bs

                drop_last = params.get("drop_last", True)
                if n_samples <= batch_size:
                    drop_last = False
                params["drop_last"] = drop_last
                return params

        class _PatchedTabNetRegressor(_TabNetMixin, TabNetRegressor):  # type: ignore
            def fit(self, X_train, y_train, **kwargs):  # type: ignore[override]
                y_array = np.asarray(y_train)
                if y_array.ndim == 1:
                    y_array = y_array.reshape(-1, 1)
                params = self._normalize_train_kwargs(self._n_samples(X_train), kwargs)
                return super().fit(X_train, y_array, **params)

            def predict(self, X):  # type: ignore[override]
                preds = super().predict(X)
                if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 1:
                    return preds.ravel()
                return preds

        class _PatchedTabNetClassifier(_TabNetMixin, TabNetClassifier):  # type: ignore
            def fit(self, X_train, y_train, **kwargs):  # type: ignore[override]
                y_array = np.asarray(y_train)
                params = self._normalize_train_kwargs(self._n_samples(X_train), kwargs)
                return super().fit(X_train, y_array, **params)

        register_model("tabnet", regressor=_PatchedTabNetRegressor, classifier=_PatchedTabNetClassifier)
    except Exception:
        pass

    # TabPFN (weights check is handled separately in training.model_factory.prepare_tabpfn_params)
    try:
        home = os.environ.get("TABPFN_HOME")
        if not home:
            project_root = Path(__file__).resolve().parent.parent.parent
            default_home = project_root / ".tabpfn_home"
            default_home.mkdir(parents=True, exist_ok=True)
            os.environ["TABPFN_HOME"] = str(default_home)
            os.environ.setdefault("TABPFN_STATE_DIR", str(default_home))
            os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(default_home / "model_cache"))
        else:
            Path(home).mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TABPFN_STATE_DIR", home)
            os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(Path(home) / "model_cache"))

        from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore

        register_model("tabpfn", regressor=TabPFNRegressor, classifier=TabPFNClassifier)
    except Exception:
        pass


def resolve_model_class(name: str, problem_type: str) -> type:
    ensure_default_models_registered()

    key = _normalize_name(name)
    canonical = _ALIASES.get(key, key)
    entry = _MODELS.get(canonical)
    if entry is None:
        raise KeyError(f"Model '{name}' is not registered")

    ptype = str(problem_type).strip().lower()
    if ptype not in {"regression", "classification"}:
        raise ValueError(f"Unknown problem_type: {problem_type!r}")
    ref = entry.regressor if ptype == "regression" else entry.classifier
    if ref is None:
        raise KeyError(f"Model '{name}' is not registered for problem type '{ptype}'")
    return _resolve_class(ref)


def list_models() -> Dict[str, ModelEntry]:
    ensure_default_models_registered()
    return dict(_MODELS)
