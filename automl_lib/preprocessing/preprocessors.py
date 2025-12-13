"""Preprocessing pipeline generation.

This module is migrated from the legacy `auto_ml.preprocessing.preprocessors`.
It builds scikit-learn ColumnTransformer pipelines based on config.
"""

from __future__ import annotations

import hashlib
import inspect
import importlib
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

from ..config.schemas import PreprocessSettings
from ..registry.preprocessors import ensure_default_preprocessors_registered, get_preprocessor

_ONE_HOT_ENCODER_KWARGS: Dict[str, object] = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
    _ONE_HOT_ENCODER_KWARGS["sparse_output"] = False
else:
    _ONE_HOT_ENCODER_KWARGS["sparse"] = False


def _to_numpy_array(data):
    return np.asarray(data)


def _build_numeric_pipeline(
    numeric_cols: List[str],
    imputation: Optional[str],
    scaling: Optional[str],
    polynomial_degree: Optional[int],
    extra_steps: List[PreprocessSettings.StepSpec],
) -> Tuple[str, Pipeline]:
    steps: List[Tuple[str, object]] = []
    name_parts: List[str] = []
    if imputation:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))
        name_parts.append(f"impute_{imputation}")
    if scaling:
        scaler: object
        if scaling in {"standard", "minmax", "robust"}:
            if scaling == "standard":
                scaler = StandardScaler()
            elif scaling == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
        else:
            scaler = get_preprocessor(str(scaling))()
        steps.append(("scaler", scaler))
        name_parts.append(f"scale_{scaling}")
    if polynomial_degree and isinstance(polynomial_degree, int) and polynomial_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
        name_parts.append(f"poly_{polynomial_degree}")
    if extra_steps:
        for idx, step in enumerate(extra_steps):
            params = step.params or {}
            try:
                payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
            except Exception:
                payload = repr(params).encode("utf-8")
            digest = hashlib.sha1(payload).hexdigest()[:8] if params else ""
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in step.name.strip().lower())
            step_tag = f"{safe_name}_{digest}" if digest else safe_name
            transformer = get_preprocessor(step.name)(**params)
            steps.append((f"step_{idx}_{step_tag}", transformer))
            name_parts.append(step_tag)
    if not steps:
        return "passthrough", "passthrough"
    return "_".join(name_parts), Pipeline(steps)


def _build_categorical_pipeline(
    categorical_cols: List[str],
    imputation: Optional[str],
    encoding: Optional[str],
    extra_steps: List[PreprocessSettings.StepSpec],
) -> Tuple[str, Pipeline]:
    steps: List[Tuple[str, object]] = []
    name_parts: List[str] = []

    if imputation:
        steps.append(("imputer", SimpleImputer(strategy=imputation)))
        name_parts.append(f"impute_{imputation}")

    if encoding:
        encoder: object
        if encoding in {"onehot", "ordinal"}:
            if encoding == "onehot":
                encoder = OneHotEncoder(**_ONE_HOT_ENCODER_KWARGS)
            else:
                encoder = OrdinalEncoder()
        else:
            encoder = get_preprocessor(str(encoding))()
        steps.append(("encoder", encoder))
        name_parts.append(f"encode_{encoding}")

    if extra_steps:
        for idx, step in enumerate(extra_steps):
            params = step.params or {}
            try:
                payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
            except Exception:
                payload = repr(params).encode("utf-8")
            digest = hashlib.sha1(payload).hexdigest()[:8] if params else ""
            safe_name = "".join(ch if ch.isalnum() else "_" for ch in step.name.strip().lower())
            step_tag = f"{safe_name}_{digest}" if digest else safe_name
            transformer = get_preprocessor(step.name)(**params)
            steps.append((f"step_{idx}_{step_tag}", transformer))
            name_parts.append(step_tag)

    if not steps:
        return "passthrough", "passthrough"
    return "_".join(name_parts), Pipeline(steps)


def generate_preprocessors(
    config: PreprocessSettings,
    feature_types: Dict[str, List[str]],
) -> List[Tuple[str, ColumnTransformer]]:
    ensure_default_preprocessors_registered()
    for module_name in getattr(config, "plugins", []) or []:
        importlib.import_module(module_name)

    numeric_cols = feature_types.get("numeric", [])
    cat_cols = feature_types.get("categorical", [])
    preprocessors: List[Tuple[str, ColumnTransformer]] = []

    numeric_extra_steps = list(getattr(config, "numeric_pipeline_steps", []) or [])
    categorical_extra_steps = list(getattr(config, "categorical_pipeline_steps", []) or [])

    if not numeric_cols:
        numeric_imputations = [None]
        scalings = [None]
        degrees = [config.polynomial_degree if isinstance(config.polynomial_degree, int) else None]
    else:
        numeric_imputations = config.numeric_imputation or [None]
        scalings = config.scaling or [None]
        degrees: List[Optional[int]]
        if config.polynomial_degree and isinstance(config.polynomial_degree, int) and config.polynomial_degree > 1:
            degrees = [None, int(config.polynomial_degree)]
        else:
            degrees = [None]

    if not cat_cols:
        categorical_imputations = [None]
        encodings = [None]
    else:
        categorical_imputations = config.categorical_imputation or [None]
        encodings = config.categorical_encoding or [None]

    for num_imp in numeric_imputations:
        for scale in scalings:
            for degree in degrees:
                numeric_name, numeric_pipeline = _build_numeric_pipeline(
                    numeric_cols, num_imp, scale, degree, numeric_extra_steps
                )
                for cat_imp in categorical_imputations:
                    for enc in encodings:
                        cat_name, cat_pipeline = _build_categorical_pipeline(cat_cols, cat_imp, enc, categorical_extra_steps)
                        transformers = []
                        if numeric_cols:
                            transformers.append(("numeric", numeric_pipeline, numeric_cols))
                        if cat_cols:
                            transformers.append(("categorical", cat_pipeline, cat_cols))
                        if transformers:
                            ct = ColumnTransformer(
                                transformers=transformers,
                                remainder="drop",
                                verbose_feature_names_out=False,
                            )
                            name_parts = []
                            if numeric_name != "passthrough":
                                name_parts.append(numeric_name)
                            if cat_name != "passthrough":
                                name_parts.append(cat_name)
                            name = "|".join(name_parts) or "no_preproc"
                            preprocessors.append((name, ct))
                        else:
                            preprocessors.append(
                                (
                                    "no_preproc",
                                    FunctionTransformer(_to_numpy_array, validate=False),
                                )
                            )

    unique = {}
    for name, ct in preprocessors:
        if name not in unique:
            unique[name] = ct
    return [(n, ct) for n, ct in unique.items()]
