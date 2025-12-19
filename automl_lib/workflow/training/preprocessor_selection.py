from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from automl_lib.config.schemas import TrainingConfig
from automl_lib.preprocessing.preprocessors import generate_preprocessors


def _has_non_numeric_columns(X: Any) -> bool:
    """Return True when the features likely need preprocessing (encoding, datetime handling, etc)."""

    if not isinstance(X, pd.DataFrame):
        return False
    try:
        for dtype in X.dtypes:
            try:
                if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
                    continue
                return True
            except Exception:
                return True
    except Exception:
        return True
    return False


def _identity_preprocessor_label(preproc_manifest_src: Optional[Path]) -> str:
    label = "preprocessed_dataset"
    if not preproc_manifest_src:
        return label
    try:
        if preproc_manifest_src.exists():
            manifest = json.loads(preproc_manifest_src.read_text(encoding="utf-8"))
            if isinstance(manifest, dict) and manifest.get("selected_preprocessor"):
                label = f"preprocessed|{manifest.get('selected_preprocessor')}"
    except Exception:
        return "preprocessed_dataset"
    return label


def select_preprocessors(
    *,
    cfg: TrainingConfig,
    feature_types: Dict[str, List[str]],
    X_train: Any,
    has_preproc_contract: bool,
    preproc_manifest_src: Optional[Path],
) -> List[Tuple[str, object]]:
    """
    Select preprocessors for training.

    When the input dataset follows the preprocessing contract and its feature table appears
    already numeric, use an identity transformer to avoid double preprocessing.
    If non-numeric columns exist (object/category/datetime/etc), fall back to the configured
    preprocessing pipelines so models receive numeric inputs.
    """

    if not has_preproc_contract:
        return generate_preprocessors(cfg.preprocessing, feature_types)

    if _has_non_numeric_columns(X_train):
        return generate_preprocessors(cfg.preprocessing, feature_types)

    label = _identity_preprocessor_label(preproc_manifest_src)
    try:
        from sklearn.preprocessing import FunctionTransformer

        return [(label, FunctionTransformer(validate=False))]
    except Exception:
        return [(label, "passthrough")]

