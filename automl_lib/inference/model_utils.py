"""Utilities for loading models and running individual predictions during inference."""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline as SKPipeline

try:
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sp = None


def load_models(model_dir: str, selected_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load all joblib pipelines from a directory, filtered by optional names."""

    models: Dict[str, Any] = {}
    failed: List[Tuple[str, str]] = []
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory '{model_dir}' does not exist")

    selection_tokens: Optional[set[str]] = None
    if selected_names:
        if not all(isinstance(name, str) for name in selected_names):
            raise ValueError("All selected model names must be strings")
        tokens: set[str] = set()
        for name in selected_names:
            lower = name.strip().lower()
            if not lower:
                continue
            tokens.add(lower)
            for suffix in ("regressor", "classifier"):
                if lower.endswith(suffix):
                    trimmed = lower[: -len(suffix)]
                    if trimmed:
                        tokens.add(trimmed)
        selection_tokens = tokens if tokens else None

    for fname in os.listdir(model_dir):
        if not fname.endswith(".joblib"):
            continue
        if fname.startswith("best_model"):
            continue
        fpath = os.path.join(model_dir, fname)
        file_stem = fname.rsplit(".", 1)[0]
        if "_impute" in file_stem:
            base_label = file_stem.split("_impute", 1)[0]
        else:
            base_label = file_stem
        if selection_tokens is not None:
            pre_candidates = {base_label.lower(), file_stem.lower()}
            if not any(
                candidate.startswith(sel) or sel.startswith(candidate)
                for candidate in pre_candidates
                for sel in selection_tokens
            ):
                continue
        try:
            pipeline = joblib.load(fpath)
        except Exception as exc:  # pragma: no cover - diagnostic
            reason = str(exc).strip() or exc.__class__.__name__
            failed.append((base_label or file_stem, reason))
            continue
        try:
            est = pipeline.named_steps.get("model", None)
            est_label = est.__class__.__name__ if est is not None else None
        except Exception:
            est = None
            est_label = None
        candidate_names = {base_label.lower(), file_stem.lower()}
        if est_label:
            est_label_lower = est_label.lower()
            candidate_names.add(est_label_lower)
            for suffix in ("regressor", "classifier"):
                if est_label_lower.endswith(suffix):
                    shortened = est_label_lower[: -len(suffix)]
                    if shortened:
                        candidate_names.add(shortened)
        if selection_tokens is not None and not any(
            candidate.startswith(sel) or sel.startswith(candidate)
            for candidate in candidate_names
            for sel in selection_tokens
        ):
            continue
        label = base_label if base_label else (est_label or file_stem)
        final_label = label
        idx = 1
        while final_label in models:
            idx += 1
            final_label = f"{label}_{idx}"
        models[final_label] = pipeline

    if not models:
        warnings.warn(
            "No models were loaded from directory '{}' with filters {}. Ensure that model files exist and names match "
            "the requested models.".format(model_dir, selected_names)
        )
    if failed:
        details = ", ".join(f"{name} ({reason})" for name, reason in failed)
        print(f"Skipped {len(failed)} model(s) that could not be loaded: {details}")
    return models


def _prepare_estimator_input(pipeline: Any, inputs: pd.DataFrame) -> Tuple[Any, Any]:
    """Return transformed features and the estimator used for prediction."""

    if isinstance(pipeline, SKPipeline) and "model" in pipeline.named_steps and len(pipeline.steps) >= 2:
        preprocessor = pipeline[:-1]
        estimator = pipeline.named_steps["model"]
        try:
            transformed = preprocessor.transform(inputs)
        except Exception:
            return inputs, pipeline
        feature_names: Optional[Iterable[str]] = None
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None
        estimator_feature_names: Optional[List[str]] = None
        if hasattr(estimator, "feature_name_"):
            try:
                candidate = list(estimator.feature_name_)
                if candidate:
                    estimator_feature_names = candidate
            except Exception:
                estimator_feature_names = None
        if estimator_feature_names is not None:
            feature_names = estimator_feature_names
        needs_named_input = feature_names is not None and estimator_feature_names is not None
        if needs_named_input:
            try:
                index = inputs.index if isinstance(inputs, pd.DataFrame) else None
                if sp is not None and hasattr(sp, "issparse") and sp.issparse(transformed):
                    transformed = pd.DataFrame.sparse.from_spmatrix(transformed, columns=feature_names, index=index)
                else:
                    transformed = pd.DataFrame(transformed, columns=feature_names, index=index)
            except Exception:
                pass
        return transformed, estimator
    return inputs, pipeline


def _get_required_feature_names(pipeline: Any) -> Optional[List[str]]:
    names = getattr(pipeline, "feature_names_in_", None)
    if names is not None:
        return list(names)
    if isinstance(pipeline, SKPipeline):
        preprocessor = pipeline.named_steps.get("preprocessor")
        if preprocessor is not None:
            names = getattr(preprocessor, "feature_names_in_", None)
            if names is not None:
                return list(names)
            transformers = getattr(preprocessor, "transformers", None)
            if transformers:
                collected: List[str] = []
                for _, _, cols in transformers:
                    if cols is None:
                        continue
                    if isinstance(cols, (list, tuple, np.ndarray)):
                        collected.extend(list(cols))
                    else:
                        collected.append(cols)
                if collected:
                    return collected
    return None


def _align_input_columns(df: pd.DataFrame, models: Dict[str, Any]) -> pd.DataFrame:
    required: List[str] = []
    seen = set()
    for pipeline in models.values():
        names = _get_required_feature_names(pipeline)
        if not names:
            continue
        for name in names:
            if name not in seen:
                required.append(name)
                seen.add(name)
    if not required:
        return df
    aligned = df.copy()
    for col in required:
        if col not in aligned.columns:
            aligned[col] = np.nan
    ordered = [col for col in required if col in aligned.columns]
    ordered += [col for col in aligned.columns if col not in required]
    aligned = aligned[ordered]
    return aligned


def _predict_with_model(pipeline: Any, inputs: pd.DataFrame) -> np.ndarray:
    """Predict using a pipeline/estimator while preserving feature names."""

    required = _get_required_feature_names(pipeline)
    if required:
        aligned = inputs.copy()
        for col in required:
            if col not in aligned.columns:
                aligned[col] = np.nan
        ordered = [col for col in required if col in aligned.columns]
        ordered += [col for col in aligned.columns if col not in required]
        aligned = aligned[ordered]
    else:
        aligned = inputs

    model_inputs, estimator = _prepare_estimator_input(pipeline, aligned)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                category=UserWarning,
            )
            preds = estimator.predict(model_inputs)
    except Exception:
        if estimator is not pipeline:
            model_inputs = aligned
            estimator = pipeline
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                    category=UserWarning,
                )
                preds = estimator.predict(aligned)
        else:
            raise

    try:
        if hasattr(estimator, "predict_proba"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
                    category=UserWarning,
                )
                probs = estimator.predict_proba(model_inputs)
            if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[1] >= 2:
                preds = probs[:, 1]
    except Exception:
        pass
    return np.asarray(preds)


def _shorten_label(label: str, used: Set[str], max_len: int = 15) -> str:
    cleaned = label.replace("Regressor", "").replace("Classifier", "")
    if "___" in cleaned:
        parts = [p for p in cleaned.split("___") if p]
        abbr_parts: List[str] = []
        for part in parts:
            part_no_underscore = part.replace("_", "")
            uppercase = "".join(ch for ch in part_no_underscore if ch.isupper())
            if len(uppercase) >= 2:
                abbr_parts.append(uppercase)
            else:
                abbr_parts.append(part_no_underscore[:3])
        candidate = "+".join(abbr_parts)
    else:
        candidate = cleaned.replace("_", "")
    candidate = candidate.strip()
    if not candidate:
        candidate = label
    if len(candidate) > max_len:
        candidate = candidate[: max_len - 3] + "..."
    base = candidate
    idx = 2
    while candidate in used:
        candidate = f"{base}{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def _shorten_labels(model_names: Iterable[str]) -> Dict[str, str]:
    used: Set[str] = set()
    mapping: Dict[str, str] = {}
    for name in model_names:
        mapping[name] = _shorten_label(name, used)
    return mapping


def _running_best(values: np.ndarray, goal: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    best = np.empty_like(arr, dtype=float)
    if goal == "min":
        current = np.inf
        for i, val in enumerate(arr):
            if np.isnan(val):
                best[i] = current
                continue
            current = val if not np.isfinite(current) else min(current, val)
            best[i] = current
    else:
        current = -np.inf
        for i, val in enumerate(arr):
            if np.isnan(val):
                best[i] = current
                continue
            current = val if not np.isfinite(current) else max(current, val)
            best[i] = current
    invalid = ~np.isfinite(best)
    best[invalid] = np.nan
    return best


def _log_if_almost_constant(model_label: str, preds: np.ndarray, threshold: float = 1e-6) -> None:
    finite = preds[np.isfinite(preds)]
    if finite.size == 0:
        return
    if np.nanmax(finite) - np.nanmin(finite) <= threshold:
        print(
            f"Warning: model '{model_label}' produced nearly constant predictions (range <= {threshold:g}). "
            "Check feature ranges or retrain the model if this is unexpected."
        )


__all__ = [
    "load_models",
    "_align_input_columns",
    "_log_if_almost_constant",
    "_predict_with_model",
    "_running_best",
    "_shorten_label",
    "_shorten_labels",
]
