import os
from pathlib import Path
from typing import Optional

from automl_lib.clearml import (
    find_first_dataset_id_by_tag,
    add_tags_to_dataset,
    hash_tag_for_path,
    register_dataset_from_path,
    disable_resource_monitoring,
)
from automl_lib.config.loaders import load_data_registration_config
from automl_lib.types import DatasetInfo
from automl_lib.clearml.context import (
    build_run_context,
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    set_run_id_env,
)
from automl_lib.clearml.naming import dataset_name, build_tags

from automl_lib.phases.data_registration.clearml_integration import create_data_registration_task


def run_data_registration_processing(
    config_path: Path,
    parent_task_id: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
):
    config_path = Path(config_path)
    cfg = load_data_registration_config(config_path)
    try:
        from automl_lib.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    clearml_cfg = cfg.clearml
    run_id = resolve_run_id(explicit=run_id, from_config=getattr(cfg.run, "id", None), from_env=get_run_id_env())
    set_run_id_env(run_id)

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
        return DatasetInfo(
            dataset_id=dataset_id_existing,
            task_id=None,
            csv_path=cfg.data.csv_path,
            run_id=run_id,
        ).model_dump()

    # If already provided, avoid redundant registration.
    if dataset_id_existing:
        return DatasetInfo(
            dataset_id=str(dataset_id_existing),
            task_id=None,
            csv_path=cfg.data.csv_path,
            run_id=run_id,
        ).model_dump()

    if not cfg.data.csv_path:
        raise ValueError("data.csv_path is required for data_registration")
    csv_path = Path(cfg.data.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dataset_key = resolve_dataset_key(explicit=getattr(cfg.run, "dataset_key", None), csv_path=str(csv_path))
    ctx = build_run_context(
        run_id=run_id,
        dataset_key=dataset_key,
        project_root=(clearml_cfg.project_name if clearml_cfg else None),
        dataset_project=(clearml_cfg.dataset_project if clearml_cfg else None),
        user=getattr(cfg.run, "user", None),
    )
    base_tags = list(getattr(clearml_cfg, "tags", []) or []) if clearml_cfg else []
    ds_tags = build_tags(ctx, phase="data_registration", extra=[*base_tags, "raw-csv", hash_tag_for_path(csv_path)])

    # Avoid task reuse for this phase (do not touch step tasks inside PipelineController)
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
        current = os.environ.get("CLEARML_TASK_ID")
        if not (current and str(current).strip()):
            os.environ.pop("CLEARML_TASK_ID", None)
            os.environ["CLEARML_TASK_ID"] = ""

    hash_tag = hash_tag_for_path(csv_path)
    existing = find_first_dataset_id_by_tag(hash_tag, clearml_cfg.dataset_project if clearml_cfg else None)
    if existing:
        print(f"dataset with same hash found: {existing}")
        try:
            add_tags_to_dataset(existing, ds_tags)
        except Exception:
            pass
        return DatasetInfo(dataset_id=existing, task_id=None, csv_path=str(csv_path), run_id=run_id).model_dump()

    task_info = create_data_registration_task(cfg.model_dump(), parent_task_id=parent_task_id)
    task = task_info.get("task")

    dataset_id = register_dataset_from_path(
        name=dataset_name("raw", ctx),
        path=csv_path,
        dataset_project=clearml_cfg.dataset_project if clearml_cfg else None,
        parent_ids=None,
        tags=ds_tags,
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

    return DatasetInfo(
        dataset_id=dataset_id,
        task_id=(task.id if task else None),
        csv_path=str(csv_path),
        run_id=run_id,
    ).model_dump()
