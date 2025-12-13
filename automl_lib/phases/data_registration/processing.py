import os
from pathlib import Path
from typing import Optional

from automl_lib.clearml import (
    find_first_dataset_id_by_tag,
    hash_tag_for_path,
    register_dataset_from_path,
    disable_resource_monitoring,
)
from automl_lib.config.loaders import load_data_registration_config
from automl_lib.types import DatasetInfo

from automl_lib.phases.data_registration.clearml_integration import create_data_registration_task


def run_data_registration_processing(config_path: Path, parent_task_id: Optional[str] = None):
    config_path = Path(config_path)
    cfg = load_data_registration_config(config_path)
    clearml_cfg = cfg.clearml

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    dataset_id_existing = None
    if clearml_cfg and clearml_cfg.enabled:
        dataset_id_existing = clearml_cfg.raw_dataset_id or cfg.data.dataset_id

    # If ClearML is disabled or raw registration is not requested, pass-through.
    if not (clearml_cfg and clearml_cfg.enabled and clearml_cfg.register_raw_dataset):
        return DatasetInfo(dataset_id=dataset_id_existing, task_id=None, csv_path=cfg.data.csv_path).model_dump()

    # If already provided, avoid redundant registration.
    if dataset_id_existing:
        return DatasetInfo(dataset_id=str(dataset_id_existing), task_id=None, csv_path=cfg.data.csv_path).model_dump()

    if not cfg.data.csv_path:
        raise ValueError("data.csv_path is required for data_registration")
    csv_path = Path(cfg.data.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Avoid task reuse for this phase (do not touch step tasks inside PipelineController)
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""

    hash_tag = hash_tag_for_path(csv_path)
    existing = find_first_dataset_id_by_tag(hash_tag, clearml_cfg.dataset_project if clearml_cfg else None)
    if existing:
        print(f"dataset with same hash found: {existing}")
        return DatasetInfo(dataset_id=existing, task_id=None, csv_path=str(csv_path)).model_dump()

    task_info = create_data_registration_task(cfg.model_dump(), parent_task_id=parent_task_id)
    task = task_info.get("task")

    dataset_id = register_dataset_from_path(
        name="raw-dataset",
        path=csv_path,
        dataset_project=clearml_cfg.dataset_project if clearml_cfg else None,
        parent_ids=None,
        tags=["raw-csv", "phase:data-registration", hash_tag],
        output_uri=clearml_cfg.base_output_uri if clearml_cfg else None,
    )
    if not dataset_id:
        raise RuntimeError("ClearML dataset registration failed")

    try:
        if task:
            task.upload_artifact("raw_dataset_id", artifact_object=str(dataset_id), wait_on_upload=True)
    except Exception:
        pass
    try:
        if task:
            task.flush(wait_for_uploads=True)
    except Exception:
        pass
    try:
        if task and os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
            task.close()
    except Exception:
        pass

    return DatasetInfo(dataset_id=dataset_id, task_id=(task.id if task else None), csv_path=str(csv_path)).model_dump()
