"""
Core processing for preprocessing phase.
特徴量型判定、前処理パイプラインの構築・適用を担当する。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from automl_lib.clearml import dataframe_from_dataset, register_dataset_from_path, disable_resource_monitoring
from automl_lib.config.loaders import load_preprocessing_config
from automl_lib.data import get_feature_types
from automl_lib.preprocessing import generate_preprocessors
from automl_lib.types import DatasetInfo

from automl_lib.phases.preprocessing.clearml_integration import create_preprocessing_task


def run_preprocessing_processing(config_path: Path, input_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config_path = Path(config_path)
    cfg = load_preprocessing_config(config_path)
    clearml_cfg = cfg.clearml

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    dataset_id_source = (
        (input_info or {}).get("dataset_id")
        or (clearml_cfg.edited_dataset_id if clearml_cfg else None)
        or (clearml_cfg.raw_dataset_id if clearml_cfg else None)
        or cfg.data.dataset_id
    )
    csv_src = (input_info or {}).get("csv_path") or cfg.data.csv_path

    if clearml_cfg and clearml_cfg.preprocessed_dataset_id:
        return DatasetInfo(
            dataset_id=clearml_cfg.preprocessed_dataset_id,
            task_id=(input_info or {}).get("task_id"),
            csv_path=csv_src,
        ).model_dump()

    if not (clearml_cfg and clearml_cfg.enabled and clearml_cfg.enable_preprocessing):
        return DatasetInfo(
            dataset_id=str(dataset_id_source) if dataset_id_source else None,
            task_id=(input_info or {}).get("task_id"),
            csv_path=csv_src,
        ).model_dump()

    if not dataset_id_source:
        raise ValueError("preprocessing requires data.dataset_id (existing ClearML Dataset ID)")

    df_raw = dataframe_from_dataset(str(dataset_id_source))
    if df_raw is None:
        raise ValueError(f"Failed to load ClearML Dataset (dataset_id={dataset_id_source})")

    target_col = cfg.data.target_column or df_raw.columns[-1]
    feature_cols = cfg.data.feature_columns or [c for c in df_raw.columns if c != target_col]
    X_df = df_raw[feature_cols].copy()

    feature_types = get_feature_types(X_df)
    preprocessors = generate_preprocessors(cfg.preprocessing, feature_types)
    if not preprocessors:
        return DatasetInfo(
            dataset_id=str(dataset_id_source),
            task_id=(input_info or {}).get("task_id"),
            csv_path=csv_src,
        ).model_dump()

    preproc_name, transformer = preprocessors[0]
    transformed = transformer.fit_transform(X_df)
    try:
        feature_names = transformer.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(transformed.shape[1])]
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(transformed):  # type: ignore[attr-defined]
            transformed = transformed.toarray()
    except Exception:
        pass

    df_preproc = pd.DataFrame(transformed, columns=feature_names)
    df_preproc[target_col] = df_raw[target_col].reset_index(drop=True)

    task_info = create_preprocessing_task(cfg.model_dump(), parent_task_id=(input_info or {}).get("task_id"))
    task = task_info.get("task")

    output_dir = Path(cfg.output.output_dir) if cfg.output else Path("outputs/train")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = output_dir / "preprocessed_features.csv"
    df_preproc.to_csv(csv_out, index=False)

    dataset_id = register_dataset_from_path(
        name="preprocessed-dataset",
        path=csv_out,
        dataset_project=clearml_cfg.dataset_project if clearml_cfg else None,
        parent_ids=[str(dataset_id_source)],
        tags=[f"preprocess:{preproc_name}", "phase:preprocessing"],
        output_uri=clearml_cfg.base_output_uri if clearml_cfg else None,
    )
    if not dataset_id:
        raise RuntimeError("Failed to register preprocessed dataset to ClearML")

    try:
        if task:
            task.upload_artifact("preprocessed_dataset_id", artifact_object=str(dataset_id), wait_on_upload=True)
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

    return DatasetInfo(dataset_id=dataset_id, task_id=(task.id if task else None), csv_path=str(csv_out)).model_dump()
