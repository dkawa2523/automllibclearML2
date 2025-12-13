"""
Preprocessor registry
---------------------
前処理ステップを「名前→生成関数/Transformer」で登録するための薄いラッパ。

目的:
- `automl_lib/preprocessing/preprocessors.py` から前処理ステップ追加を分離し、
  新しいTransformerを登録だけで使えるようにする。
- configから `preprocessing.*_pipeline_steps` で差し込みやすくする。
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, Optional, Sequence

# name -> callable(**params) -> transformer
_PREPROCESSORS: Dict[str, Callable[..., Any]] = {}
_ALIASES: Dict[str, str] = {}
_DEFAULTS_REGISTERED = False


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_\s]+", "", str(name).strip().lower())


def register_alias(alias: str, canonical_name: str) -> None:
    _ALIASES[_normalize_name(alias)] = _normalize_name(canonical_name)


def register_preprocessor(name: str, builder: Callable[..., Any], *, aliases: Optional[Sequence[str]] = None) -> None:
    key = _normalize_name(name)
    _PREPROCESSORS[key] = builder
    for alias in aliases or []:
        register_alias(alias, name)


def get_preprocessor(name: str) -> Callable[..., Any]:
    ensure_default_preprocessors_registered()
    key = _normalize_name(name)
    key = _ALIASES.get(key, key)
    if key not in _PREPROCESSORS:
        raise KeyError(f"Preprocessor '{name}' is not registered")
    return _PREPROCESSORS[key]


def list_preprocessors() -> Dict[str, Callable[..., Any]]:
    ensure_default_preprocessors_registered()
    return dict(_PREPROCESSORS)


def ensure_default_preprocessors_registered() -> None:
    """Register built-in preprocessors once.

    Users can override by calling register_preprocessor() after import.
    """

    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True

    # Basic scalers
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    register_preprocessor("standard", lambda **kwargs: StandardScaler(**kwargs))
    register_preprocessor("minmax", lambda **kwargs: MinMaxScaler(**kwargs))
    register_preprocessor("robust", lambda **kwargs: RobustScaler(**kwargs))

    # Encoders
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    onehot_kwargs: Dict[str, object] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder.__init__).parameters:
        onehot_kwargs["sparse_output"] = False
    else:
        onehot_kwargs["sparse"] = False

    def _onehot(**kwargs):
        merged = dict(onehot_kwargs)
        merged.update(kwargs)
        return OneHotEncoder(**merged)

    register_preprocessor("onehot", _onehot)
    register_preprocessor("ordinal", lambda **kwargs: OrdinalEncoder(**kwargs))

    # Common optional transforms (sklearn-only)
    try:
        from sklearn.preprocessing import QuantileTransformer

        register_preprocessor("quantile", lambda **kwargs: QuantileTransformer(**kwargs))
    except Exception:
        pass

    try:
        from sklearn.decomposition import PCA

        register_preprocessor("pca", lambda **kwargs: PCA(**kwargs))
    except Exception:
        pass
