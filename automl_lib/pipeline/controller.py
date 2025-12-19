from __future__ import annotations

import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from automl_lib.config.loaders import load_training_config
from automl_lib.config.schemas import TrainingConfig
from automl_lib.integrations.clearml.context import (
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    sanitize_name_token,
    set_run_id_env,
)


def _ensure_clearml_local_imports() -> None:
    """Ensure local PipelineController step subprocesses can import `automl_lib`.

    ClearML local step execution appends `sys.path[0]` to `PYTHONPATH`.
    When launched via `python -m ...` / `python -c ...`, `sys.path[0]` becomes an empty
    string, and step subprocesses fail with `Import error: No module named 'automl_lib'`.
    """

    try:
        repo_root = Path(__file__).resolve().parents[2]
        if not repo_root.exists():
            return
        repo_root_str = str(repo_root)

        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        if not sys.path or not sys.path[0]:
            if sys.path:
                sys.path[0] = repo_root_str
            else:
                sys.path.insert(0, repo_root_str)
        else:
            try:
                p0 = Path(str(sys.path[0])).resolve()
            except Exception:
                p0 = None
            if p0 is None or (not (p0 / "automl_lib").exists() and (repo_root / "automl_lib").exists()):
                sys.path.insert(0, repo_root_str)
                sys.path[0] = repo_root_str

        env_pp = os.environ.get("PYTHONPATH")
        if env_pp:
            parts = [p for p in env_pp.split(os.pathsep) if p]
            if repo_root_str not in parts:
                os.environ["PYTHONPATH"] = repo_root_str + os.pathsep + env_pp
        else:
            os.environ["PYTHONPATH"] = repo_root_str
    except Exception:
        return


def _should_skip_clearml_ping() -> bool:
    flag = str(os.environ.get("AUTO_ML_SKIP_CLEARML_PING", "")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _should_skip_clearml_queue_check() -> bool:
    flag = str(os.environ.get("AUTO_ML_SKIP_CLEARML_QUEUE_CHECK", "")).strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _parse_clearml_conf_value(text: str, key: str) -> Optional[str]:
    # clearml.conf example:
    # api_server:http://localhost:8008
    try:
        pat = re.compile(rf"(?m)^\\s*{re.escape(key)}\\s*:\\s*([^\\s#]+)\\s*$")
        m = pat.search(text)
        if not m:
            return None
        return str(m.group(1)).strip()
    except Exception:
        return None


def _get_clearml_api_server() -> Optional[str]:
    # Prefer env var (ClearML supports CLEARML_API_HOST)
    for key in ("CLEARML_API_HOST", "CLEARML_API_SERVER"):
        v = os.environ.get(key)
        if v and str(v).strip():
            return str(v).strip()
    cfg_path = os.environ.get("CLEARML_CONFIG_FILE")
    if not cfg_path:
        return None
    try:
        p = Path(str(cfg_path))
        if not p.exists():
            return None
        text = p.read_text(encoding="utf-8", errors="ignore")
        return _parse_clearml_conf_value(text, "api_server")
    except Exception:
        return None


def _check_clearml_api_reachable(api_server: str, *, timeout_seconds: float = 2.0) -> None:
    raw = str(api_server).strip()
    if "://" not in raw:
        raw = f"http://{raw}"
    parsed = urlparse(raw)
    host = parsed.hostname or ""
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    if not host:
        raise ValueError(f"Invalid ClearML api_server: {api_server}")
    try:
        with socket.create_connection((host, port), timeout=float(timeout_seconds)):
            return
    except Exception as exc:
        raise ValueError(
            f"ClearML API server is not reachable (api_server={api_server}). "
            "Start ClearML server or fix clearml.conf / CLEARML_API_HOST. "
            f"Reason: {exc}"
        ) from exc


def _read_obj_attr(obj: Any, name: str) -> Optional[Any]:
    try:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name)
    except Exception:
        return None


def _extract_queue_id_and_name(obj: Any) -> tuple[Optional[str], Optional[str]]:
    if obj is None:
        return None, None
    if isinstance(obj, str):
        s = obj.strip()
        return (s or None), (s or None)
    qid = _read_obj_attr(obj, "id")
    qname = _read_obj_attr(obj, "name")
    try:
        qid = str(qid).strip() if qid is not None else None
    except Exception:
        qid = None
    try:
        qname = str(qname).strip() if qname is not None else None
    except Exception:
        qname = None
    return (qid or None), (qname or None)


def _worker_has_queue(worker: Any, *, queue_id: str, queue_name: str) -> bool:
    queue_id = str(queue_id).strip()
    queue_name = str(queue_name).strip()
    if not queue_id and not queue_name:
        return False
    for attr in ("queue", "queues"):
        val = _read_obj_attr(worker, attr)
        if val is None:
            continue
        items = val if isinstance(val, (list, tuple, set)) else [val]
        for item in items:
            qid, qname = _extract_queue_id_and_name(item)
            if queue_id and qid and qid == queue_id:
                return True
            if queue_name and qname and qname == queue_name:
                return True
    return False


def _check_clearml_queues_have_agents(queue_names: list[str]) -> None:
    """Best-effort check that ClearML queues exist and have at least one worker/agent.

    Raises ValueError when the queue does not exist or has no registered workers.
    If the check cannot be performed (e.g., permissions), it will print a warning and continue.
    """

    queue_names = [str(q).strip() for q in (queue_names or []) if str(q).strip()]
    if not queue_names:
        return
    try:
        from clearml.backend_api.session import Session  # type: ignore
        from clearml.backend_api.services.v2_9 import queues as queues_service  # type: ignore
        from clearml.backend_api.services.v2_9 import workers as workers_service  # type: ignore
    except Exception:
        return

    try:
        session = Session()
        resp_q = session.send(queues_service.GetAllRequest())
        queues_list = getattr(resp_q, "queues", None) or []
        name_to_id: dict[str, str] = {}
        for q in queues_list:
            qid, qname = _extract_queue_id_and_name(q)
            if qid and qname and qname not in name_to_id:
                name_to_id[qname] = qid
        missing = [q for q in queue_names if q not in name_to_id]
        if missing:
            raise ValueError(
                "ClearML queue(s) not found: "
                + ", ".join(missing)
                + ". Create the queue in ClearML or fix clearml.queue / clearml.services_queue / clearml.agents.<phase>."
            )

        resp_w = session.send(workers_service.GetAllRequest())
        workers_list = getattr(resp_w, "workers", None) or []
        no_agents: list[str] = []
        for qname in queue_names:
            qid = name_to_id.get(qname, "")
            has_agent = any(_worker_has_queue(w, queue_id=qid, queue_name=qname) for w in workers_list)
            if not has_agent:
                no_agents.append(qname)
        if no_agents:
            raise ValueError(
                "ClearML queue(s) have no registered agents/workers: "
                + ", ".join(no_agents)
                + ". Start a clearml-agent on the queue (e.g., `clearml-agent daemon --queue <name>`) "
                "or change the queue in config. To skip this check set AUTO_ML_SKIP_CLEARML_QUEUE_CHECK=1."
            )
    except ValueError:
        raise
    except Exception as exc:
        print(f"[PipelineController] Warning: queue/agent preflight check failed; continuing. Reason: {exc}")


def run_pipeline(
    config_path: Path,
    *,
    mode: str = "clearml",
    data_registration_config: Optional[Path] = None,
    data_editing_config: Optional[Path] = None,
    preprocessing_config: Optional[Path] = None,
    inference_config: Optional[Path] = None,
    training_config_data: Optional[Dict[str, Any]] = None,
    data_registration_config_data: Optional[Dict[str, Any]] = None,
    data_editing_config_data: Optional[Dict[str, Any]] = None,
    preprocessing_config_data: Optional[Dict[str, Any]] = None,
    inference_config_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = TrainingConfig.model_validate(training_config_data) if isinstance(training_config_data, dict) else load_training_config(config_path)
    try:
        from automl_lib.integrations.clearml.overrides import apply_overrides, get_task_overrides

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
    inference_config = _resolve_optional_config_path(inference_config, "inference_config.yaml")

    # Prefer embedding resolved config dicts into step kwargs so remote PipelineController
    # steps do not rely on local filesystem paths (controller and steps run in different agents).
    try:
        from automl_lib.config.loaders import load_yaml

        if data_registration_config_data is None and data_registration_config and Path(data_registration_config).exists():
            data_registration_config_data = load_yaml(Path(data_registration_config))
        if data_editing_config_data is None and data_editing_config and Path(data_editing_config).exists():
            data_editing_config_data = load_yaml(Path(data_editing_config))
        if preprocessing_config_data is None and preprocessing_config and Path(preprocessing_config).exists():
            preprocessing_config_data = load_yaml(Path(preprocessing_config))
        if inference_config_data is None and inference_config and Path(inference_config).exists():
            inference_config_data = load_yaml(Path(inference_config))
    except Exception:
        pass

    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"clearml", "controller"}:
        raise ValueError("Only ClearML PipelineController is supported (use --mode clearml).")
    info = _run_clearml_pipeline_controller(
        config_path,
        cfg,
        run_id=run_id,
        data_registration_config=data_registration_config,
        data_editing_config=data_editing_config,
        preprocessing_config=preprocessing_config,
        inference_config=inference_config,
        data_registration_config_data=data_registration_config_data,
        data_editing_config_data=data_editing_config_data,
        preprocessing_config_data=preprocessing_config_data,
        inference_config_data=inference_config_data,
        training_config_data=(cfg.model_dump() if hasattr(cfg, "model_dump") else training_config_data),
    )
    if info is None:
        raise RuntimeError("ClearML PipelineController execution failed or is disabled in config")
    return info


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
    inference_config: Optional[Path],
    data_registration_config_data: Optional[Dict[str, Any]],
    data_editing_config_data: Optional[Dict[str, Any]],
    preprocessing_config_data: Optional[Dict[str, Any]],
    inference_config_data: Optional[Dict[str, Any]],
    training_config_data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Use ClearML PipelineController to orchestrate phases."""

    if not (cfg.clearml and cfg.clearml.enabled and cfg.clearml.enable_pipeline):
        return None

    _ensure_clearml_local_imports()

    # Ensure ClearML picks up the repo-local config file even in PipelineController subprocesses.
    try:
        from automl_lib.integrations.clearml.bootstrap import ensure_clearml_config_file

        ensure_clearml_config_file()
    except Exception:
        pass

    if not _should_skip_clearml_ping():
        api_server = _get_clearml_api_server()
        if api_server:
            _check_clearml_api_reachable(api_server, timeout_seconds=2.0)

    prev_env = {
        "AUTO_ML_PIPELINE_ACTIVE": os.environ.get("AUTO_ML_PIPELINE_ACTIVE"),
        "AUTO_ML_RUN_ID": os.environ.get("AUTO_ML_RUN_ID"),
        "CLEARML_TASK_ID": os.environ.get("CLEARML_TASK_ID"),
    }

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
        raise RuntimeError(f"Failed to import ClearML PipelineController. Reason: {exc}") from exc

    # NOTE: PipelineController function steps execute in isolated worker contexts.
    # Use concrete module-level callables (avoid package __init__ wrappers) so imports are reliable.
    from automl_lib.workflow.data_editing.processing import run_data_editing_processing
    from automl_lib.workflow.data_registration.processing import run_data_registration_processing
    from automl_lib.workflow.preprocessing.processing import run_preprocessing_processing
    from automl_lib.workflow.inference.processing import run_inference_processing
    from automl_lib.workflow.training.processing import run_training_processing

    preprocessing_cfg_path = preprocessing_config or config_path

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
        # PipelineController's controller task may start the ResourceMonitor by default (noisy on non-GPU envs).
        try:
            from automl_lib.integrations.clearml.utils import disable_resource_monitoring

            disable_resource_monitoring(getattr(pipe, "task", None))
        except Exception:
            pass

        # Pipeline controller HyperParameters (clone -> edit -> run should affect execution).
        # Keep editable inputs minimal to avoid UI confusion.
        try:
            pipe_task = getattr(pipe, "task", None)
            if pipe_task is not None:
                input_params: Dict[str, Any] = {
                    "dataset_id": str(dataset_id or ""),
                    "csv_path": str(getattr(cfg.data, "csv_path", "") or ""),
                }
                try:
                    pipe_task.connect(input_params, name="Input")
                except Exception:
                    pass

                try:
                    ds = str(input_params.get("dataset_id") or "").strip()
                    dataset_id = ds or None
                except Exception:
                    pass
                csv_path_input = ""
                try:
                    csv_path_input = str(input_params.get("csv_path") or "").strip()
                    if csv_path_input:
                        cfg_dump = cfg.model_dump() if hasattr(cfg, "model_dump") else {}  # type: ignore[attr-defined]
                        cfg_dump.setdefault("data", {})
                        cfg_dump["data"]["csv_path"] = csv_path_input
                        cfg = type(cfg).model_validate(cfg_dump)
                except Exception:
                    pass

                # Propagate the edited inputs to embedded config dicts (so step tasks do not rely on filesystem paths).
                try:
                    if csv_path_input:
                        if isinstance(data_registration_config_data, dict):
                            dr = dict(data_registration_config_data)
                            data = dict(dr.get("data") or {})
                            data["csv_path"] = csv_path_input
                            dr["data"] = data
                            data_registration_config_data = dr
                        if isinstance(data_editing_config_data, dict):
                            de = dict(data_editing_config_data)
                            data = dict(de.get("data") or {})
                            data["csv_path"] = csv_path_input
                            de["data"] = data
                            data_editing_config_data = de
                        if isinstance(preprocessing_config_data, dict):
                            pp = dict(preprocessing_config_data)
                            data = dict(pp.get("data") or {})
                            data["csv_path"] = csv_path_input
                            pp["data"] = data
                            preprocessing_config_data = pp

                    if isinstance(training_config_data, dict):
                        tr = dict(training_config_data)
                        data = dict(tr.get("data") or {})
                        if dataset_id is not None:
                            data["dataset_id"] = dataset_id
                        csv_path_final = str(getattr(cfg.data, "csv_path", "") or "").strip()
                        if csv_path_final:
                            data["csv_path"] = csv_path_final
                        tr["data"] = data
                        training_config_data = tr
                except Exception:
                    pass
        except Exception:
            pass

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

        if not cfg.clearml.run_pipeline_locally:
            if not (controller_queue or default_queue):
                raise ValueError(
                    "ClearML PipelineController remote execution requires a controller queue. "
                    "Set clearml.services_queue or clearml.queue (or clearml.agents.pipeline)."
                )
            required_steps: list[str] = []
            if (not dataset_id) and cfg.clearml.register_raw_dataset:
                required_steps.append("data_registration")
            if cfg.clearml.enable_data_editing:
                required_steps.append("data_editing")
            required_steps.extend(["preprocessing", "training"])
            if cfg.clearml.enable_inference:
                required_steps.append("inference")
            missing = [step for step in required_steps if not _q(step)]
            if missing:
                raise ValueError(
                    "ClearML PipelineController remote execution requires queue configuration. "
                    "Set clearml.queue (default) or clearml.agents.<phase> for: " + ", ".join(missing)
                )
            if not _should_skip_clearml_queue_check():
                queues_to_check: list[str] = []
                for q in [controller_queue, default_queue]:
                    if q:
                        queues_to_check.append(str(q))
                if (not dataset_id) and cfg.clearml.register_raw_dataset:
                    q = _q("data_registration")
                    if q:
                        queues_to_check.append(str(q))
                if cfg.clearml.enable_data_editing:
                    q = _q("data_editing")
                    if q:
                        queues_to_check.append(str(q))
                for step in ("preprocessing", "training"):
                    q = _q(step)
                    if q:
                        queues_to_check.append(str(q))
                if cfg.clearml.enable_inference:
                    q = _q("inference")
                    if q:
                        queues_to_check.append(str(q))
                # Deduplicate while preserving order
                uniq: list[str] = []
                seen: set[str] = set()
                for q in queues_to_check:
                    s = str(q).strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    uniq.append(s)
                _check_clearml_queues_have_agents(uniq)

        input_info: Dict[str, Any] = {"csv_path": cfg.data.csv_path, "run_id": run_id}
        if dataset_id:
            input_info["dataset_id"] = dataset_id

        last_step: Optional[str] = None
        last_result_expr: Optional[str] = None

        # Keep the full pipeline config (OmegaConf-compatible dict) in the controller task.
        try:
            pipe_task = getattr(pipe, "task", None)
            if pipe_task is not None:
                from automl_lib.config.loaders import load_yaml

                cfg_obj: Dict[str, Any] = {
                    "run": {"id": run_id},
                    "controller": {"task_id": str(pipe_task_id or "")},
                    "paths": {
                        "training_config": str(config_path),
                        "data_registration_config": str(data_registration_config) if data_registration_config else "",
                        "data_editing_config": str(data_editing_config) if data_editing_config else "",
                        "preprocessing_config": str(preprocessing_cfg_path) if preprocessing_cfg_path else "",
                        "inference_config": str(inference_config) if inference_config else "",
                    },
                    # Training config is already pydantic-validated and includes overrides.
                    "training": dict(training_config_data or (cfg.model_dump() if hasattr(cfg, "model_dump") else {})),
                }
                try:
                    if isinstance(data_registration_config_data, dict):
                        cfg_obj["data_registration"] = dict(data_registration_config_data)
                    elif data_registration_config:
                        cfg_obj["data_registration"] = load_yaml(Path(data_registration_config))
                except Exception:
                    pass
                try:
                    if isinstance(data_editing_config_data, dict):
                        cfg_obj["data_editing"] = dict(data_editing_config_data)
                    elif data_editing_config:
                        cfg_obj["data_editing"] = load_yaml(Path(data_editing_config))
                except Exception:
                    pass
                try:
                    if isinstance(preprocessing_config_data, dict):
                        cfg_obj["preprocessing"] = dict(preprocessing_config_data)
                    elif preprocessing_cfg_path:
                        cfg_obj["preprocessing"] = load_yaml(Path(preprocessing_cfg_path))
                except Exception:
                    pass
                try:
                    if isinstance(inference_config_data, dict):
                        cfg_obj["inference"] = dict(inference_config_data)
                    elif inference_config:
                        cfg_obj["inference"] = load_yaml(Path(inference_config))
                except Exception:
                    pass
                try:
                    pipe_task.connect_configuration(name="OmegaConf", configuration=cfg_obj)
                except Exception:
                    pass
        except Exception:
            pass

        # Optional: data_registration (only when dataset_id is not already provided)
        if (not dataset_id) and cfg.clearml.register_raw_dataset:
            if not data_registration_config:
                raise ValueError(
                    "clearml.register_raw_dataset is enabled but data_registration config was not found. "
                    "Provide --data-registration-config or create config_dataregit.yaml."
                )
            pipe.add_function_step(
                name="data_registration",
                parents=[],
                function=run_data_registration_processing,
                function_kwargs={
                    "config_path": str(data_registration_config),
                    "config_data": (
                        data_registration_config_data if isinstance(data_registration_config_data, dict) else None
                    ),
                    "run_id": run_id,
                },
                auto_connect_frameworks=False,
                auto_connect_arg_parser=False,
                task_type="data_processing",
                execution_queue=_q("data_registration"),
                function_return=["result"],
            )
            last_step = "data_registration"
            last_result_expr = "${data_registration.result}"

        # Optional: data_editing
        if cfg.clearml.enable_data_editing:
            if not data_editing_config:
                raise ValueError(
                    "clearml.enable_data_editing is enabled but data_editing config was not found. "
                    "Provide --data-editing-config or create config_editing.yaml."
                )
            parents = [last_step] if last_step else []
            pipe.add_function_step(
                name="data_editing",
                parents=parents,
                function=run_data_editing_processing,
                function_kwargs={
                    "config_path": str(data_editing_config),
                    "config_data": (data_editing_config_data if isinstance(data_editing_config_data, dict) else None),
                    "input_info": (last_result_expr or input_info),
                    "run_id": run_id,
                },
                auto_connect_frameworks=False,
                auto_connect_arg_parser=False,
                task_type="data_processing",
                execution_queue=_q("data_editing"),
                function_return=["result"],
            )
            last_step = "data_editing"
            last_result_expr = "${data_editing.result}"

        # preprocessing requires dataset_id (either provided or created by previous steps)
        if (not dataset_id) and (last_result_expr is None):
            raise ValueError(
                "No dataset_id is available. "
                "Set data.dataset_id or enable clearml.register_raw_dataset / clearml.enable_data_editing."
            )

        preproc_parents = [last_step] if last_step else []
        pipe.add_function_step(
            name="preprocessing",
            parents=preproc_parents,
            function=run_preprocessing_processing,
            function_kwargs={
                "config_path": str(preprocessing_cfg_path),
                "config_data": (preprocessing_config_data if isinstance(preprocessing_config_data, dict) else None),
                "input_info": (last_result_expr or input_info),
                "run_id": run_id,
            },
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
            task_type="data_processing",
            execution_queue=_q("preprocessing"),
            function_return=["result"],
        )
        last_step = "preprocessing"
        last_result_expr = "${preprocessing.result}"

        training_kwargs: Dict[str, Any] = {
            "config_path": str(config_path),
            "config_data": (training_config_data if isinstance(training_config_data, dict) else None),
            "input_info": last_result_expr,
            "run_id": run_id,
        }

        pipe.add_function_step(
            name="training",
            parents=[last_step],
            function=run_training_processing,
            function_kwargs=training_kwargs,
            auto_connect_frameworks=False,
            auto_connect_arg_parser=False,
            task_type="training",
            execution_queue=_q("training"),
            function_return=["result"],
        )
        last_step = "training"
        last_result_expr = "${training.result}"

        if cfg.clearml.enable_inference:
            if not inference_config:
                raise ValueError(
                    "clearml.enable_inference is enabled but inference config was not found. "
                    "Provide --inference-config or create inference_config.yaml."
                )
            pipe.add_function_step(
                name="inference",
                parents=[last_step],
                function=run_inference_processing,
                function_kwargs={
                    "config_path": str(inference_config),
                    "config_data": (inference_config_data if isinstance(inference_config_data, dict) else None),
                    "input_info": last_result_expr,
                    "run_id": run_id,
                },
                auto_connect_frameworks=False,
                auto_connect_arg_parser=False,
                task_type="inference",
                execution_queue=_q("inference"),
                function_return=["result"],
            )
            last_step = "inference"
            last_result_expr = "${inference.result}"

        if cfg.clearml.run_pipeline_locally and hasattr(pipe, "start_locally"):
            print("[PipelineController] starting locally (run_pipeline_locally=True)")
            pipe.start_locally(run_pipeline_steps_locally=True)
        else:
            fallback_queue = controller_queue or default_queue
            if not fallback_queue:
                raise ValueError(
                    "ClearML PipelineController remote execution requires a queue. "
                    "Set clearml.services_queue or clearml.queue."
                )
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
        raise RuntimeError(f"ClearML PipelineController failed: {exc}") from exc
    finally:
        # Restore env to avoid leaking pipeline context into subsequent runs in the same process.
        for key, old in prev_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(old)
