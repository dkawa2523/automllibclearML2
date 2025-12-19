import os
from pathlib import Path
from typing import Any, Dict, Optional

from automl_lib.integrations.clearml import (
    find_first_dataset_id_by_tag,
    add_tags_to_dataset,
    hash_tag_for_path,
    register_dataset_from_path,
    disable_resource_monitoring,
    set_user_properties,
)
from automl_lib.config.loaders import load_data_registration_config
from automl_lib.types import DatasetInfo
from automl_lib.integrations.clearml.context import (
    build_run_context,
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    run_scoped_output_dir,
    set_run_id_env,
)
from automl_lib.integrations.clearml.naming import dataset_name, build_tags

from automl_lib.workflow.data_registration.clearml_integration import create_data_registration_task


def run_data_registration_processing(
    config_path: Path,
    parent_task_id: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
):
    config_path = Path(config_path)
    if isinstance(config_data, dict):
        from automl_lib.config.schemas import DataRegistrationConfig

        cfg = DataRegistrationConfig.model_validate(config_data)
    else:
        cfg = load_data_registration_config(config_path)
    try:
        from automl_lib.integrations.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    # Allow ClearML HyperParameters edits (clone -> edit -> run) to override YAML config.
    try:
        if cfg.clearml and bool(getattr(cfg.clearml, "enabled", False)):
            from automl_lib.integrations.clearml.hyperparams import (
                apply_data_registration_hyperparams,
                get_current_task_hyperparams,
            )

            params = get_current_task_hyperparams(cast=True)
            if isinstance(params, dict):
                cfg_dump = cfg.model_dump()
                applied = apply_data_registration_hyperparams(cfg_dump, params)
                if applied != cfg_dump:
                    cfg = type(cfg).model_validate(applied)
    except Exception as exc:
        raise ValueError(f"Invalid ClearML HyperParameters override for data_registration: {exc}") from exc
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

    cfg_for_task = cfg.model_dump()
    try:
        cfg_for_task.setdefault("run", {})
        cfg_for_task["run"]["id"] = run_id
        cfg_for_task["run"]["dataset_key"] = dataset_key
        if getattr(cfg.run, "user", None):
            cfg_for_task["run"]["user"] = cfg.run.user
    except Exception:
        pass
    try:
        cfg_for_task.setdefault("data", {})
        cfg_for_task["data"]["csv_path"] = str(csv_path)
    except Exception:
        pass

    task_info = create_data_registration_task(cfg_for_task, parent_task_id=parent_task_id)
    task = task_info.get("task")

    # Persist resolved full config for reproducibility (avoid HyperParameters pollution).
    try:
        import yaml  # type: ignore

        output_dir = run_scoped_output_dir(Path("outputs/data_registration"), run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = output_dir / "data_registration_config.yaml"
        cfg_path.write_text(
            yaml.safe_dump(cfg_for_task, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if task:
            task.upload_artifact(name=cfg_path.name, artifact_object=str(cfg_path))
    except Exception:
        pass

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
            set_user_properties(
                task,
                {
                    "run_id": run_id,
                    "dataset_key": dataset_key,
                    "dataset_role": "raw",
                    "raw_dataset_id": dataset_id,
                    "csv_path": str(csv_path),
                },
            )
    except Exception:
        pass
    # Reference info in Configuration Objects (avoid HyperParameters pollution)
    try:
        if task:
            task.connect_configuration(
                name="Dataset",
                configuration={
                    "run_id": str(run_id),
                    "dataset_key": str(dataset_key),
                    "dataset_role": "raw",
                    "raw_dataset_id": str(dataset_id),
                    "csv_path": str(csv_path),
                    "hash_tag": str(hash_tag),
                },
            )
    except Exception:
        pass
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
