"""
Core processing for training phase.
本ファイルはモデル学習・評価などの解析処理を担当する予定のスタブ。
今後、auto_ml/train.py の実体をこちらに段階的に移設する。
"""

import os
import csv
from typing import Any, Dict, Optional
from pathlib import Path

from automl_lib.integrations.clearml import disable_resource_monitoring
from automl_lib.integrations.clearml.context import get_run_id_env, resolve_run_id, set_run_id_env
from automl_lib.integrations.clearml.properties import set_user_properties
from automl_lib.integrations.clearml.context import run_scoped_output_dir
from automl_lib.training import run_automl
from automl_lib.types import TrainingInfo


def run_training_processing(
    config_path: Path,
    input_info: Optional[Dict[str, Any]] = None,
    parent_task_id: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run training phase inside automl_lib.

    Returns a dict for downstream phases:
    - dataset_id
    - task_id (training summary task id)
    - training_task_ids
    - metrics (optional; per-model records)
    """
    config_path = Path(config_path)
    from automl_lib.config.loaders import load_training_config
    from automl_lib.config.schemas import TrainingConfig

    cfg = TrainingConfig.model_validate(config_data) if isinstance(config_data, dict) else load_training_config(config_path)
    try:
        from automl_lib.integrations.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    run_id = resolve_run_id(
        explicit=run_id,
        from_input=(input_info or {}).get("run_id"),
        from_config=getattr(cfg.run, "id", None),
        from_env=get_run_id_env(),
    )

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    backup = {
        "AUTO_ML_RUN_ID": os.environ.get("AUTO_ML_RUN_ID"),
    }
    set_run_id_env(run_id)

    config_path_for_run = config_path
    if isinstance(config_data, dict):
        # When PipelineController runs on a remote agent, step tasks cannot
        # rely on reading a config YAML from the controller filesystem.
        # Persist the resolved config inside the phase output dir and run from it.
        try:
            import yaml  # type: ignore

            cfg_dump = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()  # type: ignore[attr-defined]
            cfg_dump.setdefault("run", {})
            cfg_dump["run"]["id"] = run_id
            base_out = Path(getattr(cfg.output, "output_dir", "") or "outputs/training")
            out_dir = run_scoped_output_dir(base_out, run_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            config_path_for_run = out_dir / "training_config_runtime.yaml"
            config_path_for_run.write_text(
                yaml.safe_dump(cfg_dump, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
        except Exception:
            config_path_for_run = config_path

    dataset_id = (input_info or {}).get("dataset_id")

    # When running inside an existing ClearML task (e.g., cloned/enqueued execution),
    # CLEARML_TASK_ID is provided by the agent and must be preserved.
    current_task_id = os.environ.get("CLEARML_TASK_ID")
    if not (current_task_id and str(current_task_id).strip()):
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""

    parent_for_summary = parent_task_id or (input_info or {}).get("task_id")

    try:
        res = run_automl(
            config_path_for_run,
            dataset_id=(str(dataset_id) if dataset_id else None),
            parent_task_id=parent_for_summary,
        )
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

    recommended_model_id: Optional[str] = None
    recommended_model_task_id: Optional[str] = None
    recommended_model_name: Optional[str] = None
    try:
        base_out = Path(getattr(cfg.output, "output_dir", "") or "outputs/training")
        out_dir = run_scoped_output_dir(base_out, run_id)
        rec_path = out_dir / "recommended_model.csv"
        if rec_path.exists():
            with rec_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
            if isinstance(row, dict):
                recommended_model_id = str(row.get("model_id") or "").strip() or None
                recommended_model_task_id = str(row.get("task_id") or "").strip() or None
                recommended_model_name = str(row.get("model") or "").strip() or None
    except Exception:
        pass

    ret: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "task_id": summary_task_id,
        "training_task_ids": training_task_ids,
        "run_id": run_id,
    }
    if recommended_model_id:
        ret["recommended_model_id"] = recommended_model_id
    if recommended_model_task_id:
        ret["recommended_model_task_id"] = recommended_model_task_id
    if recommended_model_name:
        ret["recommended_model_name"] = recommended_model_name
    if metrics is not None:
        ret["metrics"] = metrics

    info = TrainingInfo(**ret)
    payload = info.model_dump()
    if metrics is None:
        payload.pop("metrics", None)

    # When running inside a PipelineController step, the step itself is a ClearML Task.
    # Attach a minimal pointer to the user-facing training summary task to avoid confusion in the UI.
    try:
        if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
            try:
                from clearml import Task  # type: ignore
            except Exception:
                Task = None  # type: ignore

            if Task is not None:
                step_task = Task.current_task()
                if step_task is not None:
                    step_task_id = ""
                    try:
                        step_task_id = str(getattr(step_task, "id", "") or "").strip()
                    except Exception:
                        step_task_id = ""
                    step_props: Dict[str, Any] = {
                        "run_id": run_id,
                        "dataset_id": dataset_id or "",
                    }
                    try:
                        sid = str(summary_task_id or "").strip()
                        if sid and sid != step_task_id:
                            step_props["training_summary_task_id"] = sid
                    except Exception:
                        pass
                    try:
                        if recommended_model_id:
                            step_props["recommended_model_id"] = str(recommended_model_id)
                        if recommended_model_task_id:
                            step_props["recommended_model_task_id"] = str(recommended_model_task_id)
                        if recommended_model_name:
                            step_props["recommended_model_name"] = str(recommended_model_name)
                    except Exception:
                        pass
                    set_user_properties(step_task, step_props)
                    try:
                        step_task.flush(wait_for_uploads=True)
                    except Exception:
                        pass
    except Exception:
        pass
    return payload
