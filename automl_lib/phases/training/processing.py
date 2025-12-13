"""
Core processing for training phase.
本ファイルはモデル学習・評価などの解析処理を担当する予定のスタブ。
今後、auto_ml/train.py の実体をこちらに段階的に移設する。
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path

from automl_lib.clearml import disable_resource_monitoring
from automl_lib.training import run_automl
from automl_lib.types import TrainingInfo


def run_training_processing(
    config_path: Path,
    input_info: Optional[Dict[str, Any]] = None,
    parent_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run training phase inside automl_lib.

    Returns a dict compatible with downstream comparison:
    - dataset_id
    - task_id (training summary task id)
    - training_task_ids
    - metrics (optional; per-model records)
    """
    config_path = Path(config_path)

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    backup = {
        "AUTO_ML_PARENT_TASK_ID": os.environ.get("AUTO_ML_PARENT_TASK_ID"),
        "AUTO_ML_DATASET_ID": os.environ.get("AUTO_ML_DATASET_ID"),
        "AUTO_ML_PREPROCESSED_DATASET_ID": os.environ.get("AUTO_ML_PREPROCESSED_DATASET_ID"),
    }

    dataset_id = (input_info or {}).get("dataset_id")

    os.environ.pop("CLEARML_TASK_ID", None)
    os.environ["CLEARML_TASK_ID"] = ""

    parent_for_summary = parent_task_id or (input_info or {}).get("task_id")
    if parent_for_summary:
        os.environ["AUTO_ML_PARENT_TASK_ID"] = str(parent_for_summary)
    else:
        os.environ.pop("AUTO_ML_PARENT_TASK_ID", None)

    if dataset_id:
        os.environ["AUTO_ML_DATASET_ID"] = str(dataset_id)
        if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
            os.environ["AUTO_ML_PREPROCESSED_DATASET_ID"] = str(dataset_id)
    else:
        os.environ.pop("AUTO_ML_DATASET_ID", None)
        if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
            os.environ.pop("AUTO_ML_PREPROCESSED_DATASET_ID", None)

    try:
        res = run_automl(config_path)
    finally:
        for key, val in backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    training_task_ids = []
    summary_task_id = None
    metrics = None
    if isinstance(res, dict):
        training_task_ids = res.get("training_task_ids", []) or []
        summary_task_id = res.get("summary_task_id")
        metrics = res.get("metrics")
        # prefer dataset_id returned by training if present
        dataset_id = res.get("dataset_id") or dataset_id

    ret: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "task_id": summary_task_id,
        "training_task_ids": training_task_ids,
    }
    if metrics is not None:
        ret["metrics"] = metrics

    info = TrainingInfo(**ret)
    payload = info.model_dump()
    if metrics is None:
        payload.pop("metrics", None)
    return payload
