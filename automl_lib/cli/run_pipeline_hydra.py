from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import hydra
from omegaconf import DictConfig

from automl_lib.cli.common import clearml_avoid_task_reuse, maybe_clone_from_config
from automl_lib.cli.hydra_support import conf_root, to_clean_dict
from automl_lib.config.schemas import (
    DataEditingConfig,
    DataRegistrationConfig,
    InferenceConfig,
    PreprocessingConfig,
    TrainingConfig,
)
from automl_lib.integrations.clearml.context import resolve_run_id
from automl_lib.pipeline.controller import run_pipeline


def _write_phase(path: Path, payload: Dict[str, Any]) -> None:
    from omegaconf import OmegaConf  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


@hydra.main(version_base=None, config_path=str(conf_root()), config_name="app/pipeline")
def main(cfg: DictConfig) -> None:
    root = to_clean_dict(cfg)
    pipeline_cfg = dict(root.get("pipeline") or {})

    training_raw = dict(root.get("training") or {})
    preproc_raw = dict(root.get("preprocessing") or {})
    infer_raw = dict(root.get("inference") or {})
    datareg_raw = dict(root.get("data_registration") or {})
    editing_raw = dict(root.get("data_editing") or {})

    # Validate each phase config with existing pydantic schemas (keeps behavior consistent).
    training_cfg = TrainingConfig.model_validate(training_raw)
    _ = PreprocessingConfig.model_validate(preproc_raw)
    # In pipeline runs, inference.model_id is typically provided by training output.
    # Validate inference config by injecting a placeholder model_id if missing.
    infer_for_validate = dict(infer_raw)
    if not infer_for_validate.get("model_id") and not infer_for_validate.get("model_path"):
        infer_for_validate["model_id"] = "00000000000000000000000000000000"
    _ = InferenceConfig.model_validate(infer_for_validate)
    _ = DataRegistrationConfig.model_validate(datareg_raw)
    _ = DataEditingConfig.model_validate(editing_raw)

    if maybe_clone_from_config(training_cfg, phase="pipeline", output_info=None):
        return
    clearml_avoid_task_reuse()

    run_id = resolve_run_id(from_config=getattr(training_cfg.run, "id", None), from_env=None)

    base_dir = Path("outputs") / "pipeline" / str(run_id)
    cfg_dir = base_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved phase configs so PipelineController step tasks can read stable paths.
    persist = bool(pipeline_cfg.get("persist_resolved_configs", True))
    if persist:
        _write_phase(cfg_dir / "training.yaml", training_raw)
        _write_phase(cfg_dir / "preprocessing.yaml", preproc_raw)
        _write_phase(cfg_dir / "inference.yaml", infer_raw)
        _write_phase(cfg_dir / "data_registration.yaml", datareg_raw)
        _write_phase(cfg_dir / "data_editing.yaml", editing_raw)

    result = run_pipeline(
        cfg_dir / "training.yaml" if persist else Path("config.yaml"),
        mode=str(pipeline_cfg.get("mode") or "clearml"),
        data_registration_config=(cfg_dir / "data_registration.yaml") if persist else None,
        data_editing_config=(cfg_dir / "data_editing.yaml") if persist else None,
        preprocessing_config=(cfg_dir / "preprocessing.yaml") if persist else None,
        inference_config=(cfg_dir / "inference.yaml") if persist else None,
        training_config_data=training_raw,
        data_registration_config_data=datareg_raw,
        data_editing_config_data=editing_raw,
        preprocessing_config_data=preproc_raw,
        inference_config_data=infer_raw,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
