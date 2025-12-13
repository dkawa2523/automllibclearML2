import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from automl_lib.clearml import (
    dataframe_from_dataset,
    find_first_dataset_id_by_tag,
    hash_tag_for_path,
    register_dataset_from_path,
    disable_resource_monitoring,
)
from automl_lib.clearml.logging import report_table
from automl_lib.config.loaders import load_data_editing_config
from automl_lib.types import DatasetInfo

from automl_lib.phases.data_editing.clearml_integration import create_data_editing_task


def run_data_editing_processing(
    config_path: Path,
    input_info: Optional[Dict[str, Any]] = None,
    parent_task_id: Optional[str] = None,
):
    config_path = Path(config_path)
    cfg = load_data_editing_config(config_path)
    clearml_cfg = cfg.clearml

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    # Avoid task reuse for this phase (do not touch step tasks inside PipelineController)
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""

    def _load_df(dataset_id: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
        if dataset_id:
            df = dataframe_from_dataset(dataset_id)
            if df is None:
                raise ValueError(f"Failed to load ClearML Dataset (dataset_id={dataset_id})")
            return df
        if not csv_path:
            raise ValueError("data.csv_path is required when dataset_id is not provided")
        return pd.read_csv(csv_path)

    if not (cfg.editing and cfg.editing.enable):
        dataset_id_passthrough = (input_info or {}).get("dataset_id")
        task_id_passthrough = (input_info or {}).get("task_id")
        csv_path_passthrough = (input_info or {}).get("csv_path") or cfg.data.csv_path
        if clearml_cfg and clearml_cfg.edited_dataset_id:
            dataset_id_passthrough = clearml_cfg.edited_dataset_id
        return DatasetInfo(
            dataset_id=str(dataset_id_passthrough) if dataset_id_passthrough else None,
            task_id=str(task_id_passthrough) if task_id_passthrough else None,
            csv_path=str(csv_path_passthrough) if csv_path_passthrough else None,
        ).model_dump()

    if clearml_cfg and clearml_cfg.edited_dataset_id:
        csv_src = (input_info or {}).get("csv_path") or cfg.data.csv_path
        return DatasetInfo(dataset_id=clearml_cfg.edited_dataset_id, task_id=(input_info or {}).get("task_id"), csv_path=csv_src).model_dump()

    parent_id = parent_task_id or (input_info or {}).get("task_id")
    task_info = create_data_editing_task(cfg.model_dump(), parent_task_id=parent_id)
    task = task_info.get("task")
    logger = task_info.get("logger")

    dataset_id_source = (
        (input_info or {}).get("dataset_id")
        or (clearml_cfg.raw_dataset_id if clearml_cfg else None)
        or cfg.data.dataset_id
    )
    csv_src = (input_info or {}).get("csv_path") or cfg.data.csv_path

    df_before = _load_df(str(dataset_id_source) if dataset_id_source else None, csv_src)
    df_after = df_before.copy()

    if cfg.editing.drop_columns:
        df_after = df_after.drop(
            columns=[c for c in cfg.editing.drop_columns if c in df_after.columns],
            errors="ignore",
        )
    if cfg.editing.rename_columns:
        df_after = df_after.rename(columns=cfg.editing.rename_columns)
    if cfg.editing.query:
        try:
            df_after = df_after.query(cfg.editing.query)
        except Exception:
            pass
    if cfg.editing.clip_values:
        for col, bounds in cfg.editing.clip_values.items():
            if col not in df_after.columns:
                continue
            vmin = bounds.get("min")
            vmax = bounds.get("max")
            try:
                df_after[col] = df_after[col].clip(lower=vmin, upper=vmax)
            except Exception:
                pass
    if cfg.editing.fillna is not None:
        try:
            df_after = df_after.fillna(cfg.editing.fillna)
        except Exception:
            pass
    if cfg.editing.fillna_columns:
        for col, val in cfg.editing.fillna_columns.items():
            if col in df_after.columns:
                try:
                    df_after[col] = df_after[col].fillna(val)
                except Exception:
                    pass

    out_path = Path(cfg.editing.output_path or "outputs/data_editing/edited.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_after.to_csv(out_path, index=False)

    summary_rows = [
        {"name": "rows_before", "value": int(df_before.shape[0])},
        {"name": "rows_after", "value": int(df_after.shape[0])},
        {"name": "cols_before", "value": int(df_before.shape[1])},
        {"name": "cols_after", "value": int(df_after.shape[1])},
    ]
    if logger:
        try:
            report_table(logger, title="data_edit_summary", df=pd.DataFrame(summary_rows), series="summary")
        except Exception:
            pass

    # If ClearML enabled, detect duplicates and register dataset.
    dataset_id = dataset_id_source if dataset_id_source else None
    if clearml_cfg and clearml_cfg.enabled:
        hash_tag = hash_tag_for_path(out_path)
        existing = find_first_dataset_id_by_tag(hash_tag, clearml_cfg.dataset_project)
        if existing:
            print(f"edited dataset with same hash found: {existing}")
            try:
                if task:
                    task.upload_artifact("duplicate_dataset_id", artifact_object=str(existing), wait_on_upload=True)
            except Exception:
                pass
            dataset_id = existing
        else:
            registered = register_dataset_from_path(
                name="edited-dataset",
                path=out_path,
                dataset_project=clearml_cfg.dataset_project,
                parent_ids=[str(dataset_id_source)] if dataset_id_source else None,
                tags=["edited", "phase:data-editing", hash_tag],
                output_uri=clearml_cfg.base_output_uri,
            )
            if registered:
                dataset_id = registered

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

    return DatasetInfo(dataset_id=str(dataset_id) if dataset_id else None, task_id=(task.id if task else None), csv_path=str(out_path)).model_dump()
