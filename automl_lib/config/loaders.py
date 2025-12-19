from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore

from .schemas import (
    DataEditingConfig,
    DataRegistrationConfig,
    InferenceConfig,
    PreprocessingConfig,
    TrainingConfig,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    # Prefer OmegaConf when available: supports interpolations, typed containers,
    # and is the foundation for Hydra-based config composition.
    try:  # pragma: no cover - optional dependency
        from omegaconf import OmegaConf  # type: ignore

        cfg = OmegaConf.load(str(path))
        data = OmegaConf.to_container(cfg, resolve=True)
        return data if isinstance(data, dict) else {}
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def load_training_config(path: Path) -> TrainingConfig:
    raw = load_yaml(path)
    return TrainingConfig.model_validate(raw)


def load_inference_config(path: Path) -> InferenceConfig:
    raw = load_yaml(path)
    return InferenceConfig.model_validate(raw)


def load_data_registration_config(path: Path) -> DataRegistrationConfig:
    raw = load_yaml(path)
    return DataRegistrationConfig.model_validate(raw)


def load_data_editing_config(path: Path) -> DataEditingConfig:
    raw = load_yaml(path)
    return DataEditingConfig.model_validate(raw)


def _extract_output_dir(raw: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    out_raw = raw.get("output")
    if isinstance(out_raw, dict):
        out_dir = out_raw.get("output_dir")
        if out_dir is not None:
            output["output_dir"] = out_dir
    if not output:
        root_dir = raw.get("output_dir")
        if root_dir is not None:
            output["output_dir"] = root_dir
    return output


def _looks_like_training_config(raw: Dict[str, Any]) -> bool:
    # Heuristic: training configs usually include these sections.
    return any(
        key in raw
        for key in (
            "models",
            "ensembles",
            "cross_validation",
            "evaluation",
            "optimization",
            "interpretation",
            "visualizations",
        )
    )


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    raw = load_yaml(path)
    output = _extract_output_dir(raw) if not _looks_like_training_config(raw) else {}
    slim: Dict[str, Any] = {
        "run": raw.get("run") or {},
        "data": raw.get("data") or {},
        "preprocessing": raw.get("preprocessing") or {},
        "output": output,
        "clearml": raw.get("clearml"),
    }
    return PreprocessingConfig.model_validate(slim)
