from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from automl_lib.config.loaders import load_training_config
from automl_lib.config.schemas import TrainingConfig
from automl_lib.clearml.context import (
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    sanitize_name_token,
    set_run_id_env,
)


def run_pipeline(
    config_path: Path,
    *,
    mode: str = "auto",
    data_registration_config: Optional[Path] = None,
    data_editing_config: Optional[Path] = None,
    preprocessing_config: Optional[Path] = None,
    comparison_config: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = load_training_config(config_path)
    try:
        from automl_lib.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    run_id = resolve_run_id(from_config=getattr(cfg.run, "id", None), from_env=get_run_id_env())
    set_run_id_env(run_id)
    data_registration_config = _resolve_optional_config_path(data_registration_config, "config_dataregit.yaml")
    data_editing_config = _resolve_optional_config_path(data_editing_config, "config_editing.yaml")
    preprocessing_config = _resolve_optional_config_path(preprocessing_config, "config_preprocessing.yaml")
    comparison_config = _resolve_optional_config_path(comparison_config, "config_comparison.yaml")
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"in_process", "in-process", "local"}:
        return _run_in_process_pipeline(
            config_path,
            cfg,
            run_id=run_id,
            data_registration_config=data_registration_config,
            data_editing_config=data_editing_config,
            preprocessing_config=preprocessing_config,
            comparison_config=comparison_config,
        )
    if mode_norm in {"clearml", "controller"}:
        info = _run_clearml_pipeline_controller(
            config_path,
            cfg,
            run_id=run_id,
            data_registration_config=data_registration_config,
            data_editing_config=data_editing_config,
            preprocessing_config=preprocessing_config,
            comparison_config=comparison_config,
        )
        if info is None:
            raise RuntimeError("ClearML PipelineController execution failed or is disabled in config")
        return info
    # auto
    if cfg.clearml and cfg.clearml.enable_pipeline:
        info = _run_clearml_pipeline_controller(
            config_path,
            cfg,
            run_id=run_id,
            data_registration_config=data_registration_config,
            data_editing_config=data_editing_config,
            preprocessing_config=preprocessing_config,
            comparison_config=comparison_config,
        )
        if info is not None:
            return info
    return _run_in_process_pipeline(
        config_path,
        cfg,
        run_id=run_id,
        data_registration_config=data_registration_config,
        data_editing_config=data_editing_config,
        preprocessing_config=preprocessing_config,
        comparison_config=comparison_config,
    )


def _resolve_optional_config_path(explicit: Optional[Path], fallback_name: str) -> Optional[Path]:
    if explicit is not None:
        return Path(explicit)
    candidate = Path(fallback_name)
    return candidate if candidate.exists() else None


def _resolve_input_dataset_id(cfg: TrainingConfig) -> Optional[str]:
    if cfg.clearml:
        for cand in [
            cfg.clearml.preprocessed_dataset_id,
            cfg.clearml.edited_dataset_id,
            cfg.clearml.raw_dataset_id,
        ]:
            if cand:
                return str(cand)
    dataset_id = getattr(cfg.data, "dataset_id", None)
    return str(dataset_id) if dataset_id else None


def _run_in_process_pipeline(
    config_path: Path,
    cfg: TrainingConfig,
    *,
    run_id: str,
    data_registration_config: Optional[Path],
    data_editing_config: Optional[Path],
    preprocessing_config: Optional[Path],
    comparison_config: Optional[Path],
) -> Dict[str, Any]:
    from automl_lib.phases import run_data_editing, run_data_registration, run_preprocessing, run_reporting, run_training

    preprocessing_cfg_path = preprocessing_config or config_path
    comparison_cfg_path = comparison_config or config_path

    dataset_id = _resolve_input_dataset_id(cfg)
    input_info: Dict[str, Any] = {"csv_path": cfg.data.csv_path, "run_id": run_id}
    if dataset_id:
        input_info["dataset_id"] = dataset_id

    ret: Dict[str, Any] = {"mode": "in_process", "run_id": run_id}

    # Optional: data_registration (CSV -> ClearML Dataset)
    if cfg.clearml and cfg.clearml.register_raw_dataset:
        if not data_registration_config:
            raise ValueError("clearml.register_raw_dataset=true but data_registration config is not provided/found")
        datareg_info = run_data_registration(data_registration_config)
        ret["data_registration"] = datareg_info
        if isinstance(datareg_info, dict):
            input_info = dict(datareg_info)

    # Optional: data_editing (Dataset/CSV -> edited Dataset)
    if cfg.clearml and cfg.clearml.enable_data_editing:
        if not data_editing_config:
            raise ValueError("clearml.enable_data_editing=true but data_editing config is not provided/found")
        editing_info = run_data_editing(data_editing_config, input_info=input_info)
        ret["data_editing"] = editing_info
        if isinstance(editing_info, dict):
            input_info = dict(editing_info)

    dataset_id_for_pipeline = str(input_info.get("dataset_id") or dataset_id or "")
    if not dataset_id_for_pipeline:
        raise ValueError(
            "Pipeline requires an existing ClearML Dataset ID. "
            "Set data.dataset_id or enable clearml.register_raw_dataset / clearml.enable_data_editing."
        )
    input_info.setdefault("dataset_id", dataset_id_for_pipeline)
    input_info.setdefault("csv_path", cfg.data.csv_path)

    comparison_mode = getattr(cfg.clearml, "comparison_mode", "disabled") if cfg.clearml else None
    embed_comparison = bool(comparison_mode == "embedded")

    preproc_info = run_preprocessing(preprocessing_cfg_path, input_info=input_info)
    training_info = run_training(
        config_path,
        input_info=preproc_info,
        comparison_config_path=(str(comparison_cfg_path) if embed_comparison else None),
    )
    reporting_info = run_reporting(
        config_path,
        preprocessing_config_path=preprocessing_cfg_path,
        preprocessing_info=preproc_info if isinstance(preproc_info, dict) else None,
        training_info=training_info if isinstance(training_info, dict) else None,
        run_id=run_id,
    )

    ret.update(
        {
            "dataset_id": dataset_id_for_pipeline,
            "preprocessing": preproc_info,
            "training": training_info,
            "reporting": reporting_info,
        }
    )
    return ret


def _agent_queue(clearml_cfg, agent_key: str) -> Optional[str]:
    if not clearml_cfg:
        return None
    agents = getattr(clearml_cfg, "agents", None)
    if isinstance(agents, dict):
        v = agents.get(agent_key)
        return str(v) if v else None
    try:
        v = getattr(agents, agent_key, None)
        return str(v) if v else None
    except Exception:
        return None


def _run_clearml_pipeline_controller(
    config_path: Path,
    cfg: TrainingConfig,
    *,
    run_id: str,
    data_registration_config: Optional[Path],
    data_editing_config: Optional[Path],
    preprocessing_config: Optional[Path],
    comparison_config: Optional[Path],
) -> Optional[Dict[str, Any]]:
    """Use ClearML PipelineController to orchestrate phases."""

    if not (cfg.clearml and cfg.clearml.enabled):
        return None

    dataset_id = _resolve_input_dataset_id(cfg)

    # Avoid reusing a previous task id for the controller
    current_task_id = os.environ.get("CLEARML_TASK_ID")
    if not (current_task_id and str(current_task_id).strip()):
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""
    # Flag to tell sub-phases they run inside pipeline (avoid closing tasks too early)
    os.environ["AUTO_ML_PIPELINE_ACTIVE"] = "1"
    os.environ["AUTO_ML_RUN_ID"] = str(run_id)

    try:
        from clearml.automation.controller import PipelineController  # type: ignore
    except Exception as exc:
        print(f"[PipelineController] failed to import clearml controller: {exc}")
        return None

    # NOTE: PipelineController function steps execute in isolated worker contexts.
    # Use concrete module-level callables (avoid package __init__ wrappers) so imports are reliable.
    from automl_lib.phases.data_editing.processing import run_data_editing_processing
    from automl_lib.phases.data_registration.processing import run_data_registration_processing
    from automl_lib.phases.preprocessing.processing import run_preprocessing_processing
    from automl_lib.phases.training.processing import run_training_processing
    from automl_lib.phases.reporting.processing import run_reporting_processing

    preprocessing_cfg_path = preprocessing_config or config_path
    comparison_cfg_path = comparison_config or config_path

    dataset_key = resolve_dataset_key(
        explicit=getattr(cfg.run, "dataset_key", None),
        dataset_id=str(dataset_id) if dataset_id else None,
        csv_path=getattr(cfg.data, "csv_path", None),
    )
    base_pipeline_name = cfg.clearml.task_name or f"pipeline-{Path(config_path).stem}"
    # NOTE: ClearML PipelineController internally queries tasks by passing the name as a regex
    # (without escaping), so characters like `[` / `]` can break the server-side regex parser.
    # Keep the controller task name regex-safe and rely on tags/run_id for discoverability.
    safe_base = sanitize_name_token(base_pipeline_name, max_len=64)
    safe_run = sanitize_name_token(run_id, max_len=64)
    safe_ds = sanitize_name_token(dataset_key, max_len=64)
    pipeline_name = f"{safe_base} ds:{safe_ds} run:{safe_run}"
    project_name = cfg.clearml.project_name or "AutoML"
    print(f"[PipelineController] init controller name={pipeline_name} project={project_name}")

    try:
        # Reduce overhead / avoid hangs from repo & requirements scanning in controller tasks.
        os.environ.setdefault("CLEARML_SKIP_PIP_FREEZE", "1")
        pipe = PipelineController(
            name=pipeline_name,
            project=project_name,
            version="1.0",
        )
        pipe_task_id = None
        try:
            pipe_task_id = pipe.task.id  # type: ignore[attr-defined]
        except Exception:
            pipe_task_id = None

        default_queue = _agent_queue(cfg.clearml, "pipeline") or cfg.clearml.queue
        controller_queue = cfg.clearml.services_queue or _agent_queue(cfg.clearml, "pipeline") or cfg.clearml.queue
        if default_queue and not cfg.clearml.run_pipeline_locally:
            pipe.set_default_execution_queue(default_queue)
        print(
            f"[PipelineController] queues: controller={controller_queue}, default={default_queue}, "
            f"agents={cfg.clearml.agents}"
        )

        def _q(agent_key: str) -> Optional[str]:
            return _agent_queue(cfg.clearml, agent_key) or default_queue

        input_info: Dict[str, Any] = {"csv_path": cfg.data.csv_path, "run_id": run_id}
        if dataset_id:
            input_info["dataset_id"] = dataset_id

        last_step: Optional[str] = None
        last_result_expr: Optional[str] = None

        # Optional: data_registration (only when dataset_id is not already provided)
        if (not dataset_id) and cfg.clearml.register_raw_dataset:
            if not data_registration_config:
                print("[PipelineController] register_raw_dataset is enabled but config_dataregit.yaml was not found.")
                return None
            pipe.add_function_step(
                name="data_registration",
                parents=[],
                function=run_data_registration_processing,
                function_kwargs={"config_path": str(data_registration_config), "run_id": run_id},
                execution_queue=_q("data_registration"),
                function_return=["result"],
            )
            last_step = "data_registration"
            last_result_expr = "${data_registration.result}"

        # Optional: data_editing
        if cfg.clearml.enable_data_editing:
            if not data_editing_config:
                print("[PipelineController] enable_data_editing is enabled but config_editing.yaml was not found.")
                return None
            parents = [last_step] if last_step else []
            pipe.add_function_step(
                name="data_editing",
                parents=parents,
                function=run_data_editing_processing,
                function_kwargs={
                    "config_path": str(data_editing_config),
                    "input_info": (last_result_expr or input_info),
                    "run_id": run_id,
                },
                execution_queue=_q("data_editing"),
                function_return=["result"],
            )
            last_step = "data_editing"
            last_result_expr = "${data_editing.result}"

        # preprocessing requires dataset_id (either provided or created by previous steps)
        if (not dataset_id) and (last_result_expr is None):
            print(
                "[PipelineController] No dataset_id is available. "
                "Set data.dataset_id or enable clearml.register_raw_dataset / clearml.enable_data_editing."
            )
            return None

        preproc_parents = [last_step] if last_step else []
        pipe.add_function_step(
            name="preprocessing",
            parents=preproc_parents,
            function=run_preprocessing_processing,
            function_kwargs={
                "config_path": str(preprocessing_cfg_path),
                "input_info": (last_result_expr or input_info),
                "run_id": run_id,
            },
            execution_queue=_q("preprocessing"),
            function_return=["result"],
        )
        last_step = "preprocessing"
        last_result_expr = "${preprocessing.result}"

        comparison_mode = getattr(cfg.clearml, "comparison_mode", "disabled")

        training_kwargs: Dict[str, Any] = {
            "config_path": str(config_path),
            "input_info": last_result_expr,
            "run_id": run_id,
        }
        if comparison_mode == "embedded":
            training_kwargs["comparison_config_path"] = str(comparison_cfg_path)

        pipe.add_function_step(
            name="training",
            parents=[last_step],
            function=run_training_processing,
            function_kwargs=training_kwargs,
            execution_queue=_q("training"),
            function_return=["result"],
        )

        pipe.add_function_step(
            name="reporting",
            parents=["training"],
            function=run_reporting_processing,
            function_kwargs={
                "config_path": str(config_path),
                "preprocessing_config_path": str(preprocessing_cfg_path),
                "preprocessing_info": "${preprocessing.result}",
                "training_info": "${training.result}",
                "pipeline_task_id": str(pipe_task_id or ""),
                "run_id": run_id,
            },
            execution_queue=_q("reporting"),
            function_return=["result"],
        )
        last_step = "reporting"

        if cfg.clearml.run_pipeline_locally and hasattr(pipe, "start_locally"):
            print("[PipelineController] starting locally (run_pipeline_locally=True)")
            pipe.start_locally(run_pipeline_steps_locally=True)
        else:
            fallback_queue = controller_queue or default_queue or "services"
            print(f"[PipelineController] starting on queue={fallback_queue}")
            pipe.start(queue=fallback_queue)
            pipe.wait()

        print("[PipelineController] pipeline controller finished/queued successfully")
        return {
            "mode": "clearml_pipeline",
            "pipeline_task_id": pipe_task_id,
            "dataset_id": dataset_id,
            "run_id": run_id,
        }
    except Exception as exc:
        print(f"[PipelineController] failed to run pipeline: {exc}")
        return None
