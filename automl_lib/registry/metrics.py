"""
Metrics registry
----------------
評価指標や最適化指標を登録・参照するための薄いラッパ。

目的:
- `training/evaluation.py` の scoring 定義を集約し、指標追加を registry だけで済むようにする。
- RMSE のような derived 指標（MSE から算出）も扱えるようにする。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

_DEFAULTS_REGISTERED = False
_ALIASES: Dict[str, str] = {}


def _normalize_name(name: str) -> str:
    lowered = str(name).strip().lower()
    lowered = re.sub(r"[\s\-]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered)
    return lowered


@dataclass(frozen=True)
class MetricSpec:
    """Metric definition used for CV scoring & post-processing."""

    # Base key used in cross_validate output (e.g. "mse").
    key: str
    # sklearn scoring string or scorer callable (e.g. "neg_mean_squared_error"). None for derived metrics.
    sklearn_scoring: Optional[Any] = None
    # "regression" | "classification" | "both"
    kind: str = "both"
    # True if metric is a loss/error (lower is better). For sklearn "neg_*" scorers, this is typically True.
    is_loss: bool = False
    # Derived metric: computed from another metric key (e.g. rmse <- mse)
    derived_from: Optional[str] = None
    derive: Optional[Callable[[float], float]] = None


_METRICS: Dict[str, MetricSpec] = {}


def register_alias(alias: str, canonical_name: str) -> None:
    _ALIASES[_normalize_name(alias)] = _normalize_name(canonical_name)


def register_metric(
    name: str,
    *,
    sklearn_scoring: Optional[Any] = None,
    kind: str = "both",
    is_loss: bool = False,
    derived_from: Optional[str] = None,
    derive: Optional[Callable[[float], float]] = None,
    aliases: Optional[Sequence[str]] = None,
) -> None:
    key = _normalize_name(name)
    _METRICS[key] = MetricSpec(
        key=key,
        sklearn_scoring=sklearn_scoring,
        kind=kind,
        is_loss=is_loss,
        derived_from=_normalize_name(derived_from) if derived_from else None,
        derive=derive,
    )
    for alias in aliases or []:
        register_alias(alias, name)


def ensure_default_metrics_registered() -> None:
    global _DEFAULTS_REGISTERED
    if _DEFAULTS_REGISTERED:
        return
    _DEFAULTS_REGISTERED = True

    # Regression
    register_metric("r2", sklearn_scoring="r2", kind="regression", is_loss=False)
    register_metric("mae", sklearn_scoring="neg_mean_absolute_error", kind="regression", is_loss=True)
    register_metric("mse", sklearn_scoring="neg_mean_squared_error", kind="regression", is_loss=True)
    # RMSE is derived from MSE (sqrt is monotonic so using MSE for optimization is OK)
    register_metric("rmse", kind="regression", is_loss=True, derived_from="mse", derive=lambda v: float(math.sqrt(v)))

    # Classification
    register_metric("accuracy", sklearn_scoring="accuracy", kind="classification", is_loss=False)
    register_metric("precision_macro", sklearn_scoring="precision_macro", kind="classification", is_loss=False)
    register_metric("recall_macro", sklearn_scoring="recall_macro", kind="classification", is_loss=False)
    register_metric("f1_macro", sklearn_scoring="f1_macro", kind="classification", is_loss=False)
    register_metric("roc_auc_ovr", sklearn_scoring="roc_auc_ovr", kind="classification", is_loss=False)


def get_metric_spec(name: str, *, problem_type: Optional[str] = None) -> MetricSpec:
    ensure_default_metrics_registered()
    key = _normalize_name(name)
    key = _ALIASES.get(key, key)
    spec = _METRICS.get(key)
    if spec is None:
        raise KeyError(f"Metric '{name}' is not registered")
    if problem_type is None:
        return spec
    ptype = str(problem_type).strip().lower()
    if ptype not in {"regression", "classification"}:
        raise ValueError(f"Unknown problem_type: {problem_type!r}")
    if spec.kind not in {"both", ptype}:
        raise KeyError(f"Metric '{name}' is not registered for problem type '{ptype}'")
    return spec


def base_metric_key(name: str, *, problem_type: str) -> str:
    spec = get_metric_spec(name, problem_type=problem_type)
    if spec.derived_from:
        return spec.derived_from
    return spec.key


def build_sklearn_scoring(problem_type: str, metrics: Iterable[str]) -> Dict[str, Any]:
    """Build a scoring dict for sklearn cross_validate/cross_val_score."""

    scoring: Dict[str, Any] = {}
    ptype = str(problem_type).strip().lower()
    for metric_name in metrics:
        base_key = base_metric_key(metric_name, problem_type=ptype)
        base_spec = get_metric_spec(base_key, problem_type=ptype)
        if base_spec.sklearn_scoring is None:
            raise ValueError(f"Metric '{metric_name}' has no sklearn_scoring")
        scoring[base_key] = base_spec.sklearn_scoring
    return scoring


def is_loss_metric(name: str, *, problem_type: str) -> bool:
    spec = get_metric_spec(name, problem_type=problem_type)
    return bool(spec.is_loss)


def add_derived_metrics(result: Dict[str, Any], *, problem_type: str, requested_metrics: Iterable[str]) -> None:
    """Mutate `result` dict by adding derived metrics (if their base exists)."""

    ptype = str(problem_type).strip().lower()
    for metric_name in requested_metrics:
        spec = get_metric_spec(metric_name, problem_type=ptype)
        if not spec.derived_from or not spec.derive:
            continue
        if metric_name in result:
            continue
        base_key = spec.derived_from
        base_val = result.get(base_key)
        if base_val is None:
            continue
        try:
            result[spec.key] = spec.derive(float(base_val))
        except Exception:
            continue


def list_metrics() -> Dict[str, MetricSpec]:
    ensure_default_metrics_registered()
    return dict(_METRICS)

# サンプル（後で実装）
# from sklearn.metrics import r2_score
# register_metric("r2", r2_score)
