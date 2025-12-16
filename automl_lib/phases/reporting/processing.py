from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from automl_lib.clearml import disable_resource_monitoring, report_table, upload_artifacts
from automl_lib.clearml.context import (
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    run_scoped_output_dir,
    set_run_id_env,
)
from automl_lib.config.loaders import load_preprocessing_config, load_training_config
from automl_lib.phases.reporting.clearml_integration import create_reporting_task
from automl_lib.types import DatasetInfo, ReportingInfo, TrainingInfo


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _read_text_or_pickle_string(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        s = path.read_text(encoding="utf-8").strip()
        if s:
            return s
    except Exception:
        pass
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            obj = json.loads(raw)
            if isinstance(obj, str) and obj.strip():
                return obj.strip()
            if isinstance(obj, dict):
                for k in ["dataset_id", "preprocessed_dataset_id", "id"]:
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    except Exception:
        pass
    try:
        import pickle

        obj = pickle.loads(path.read_bytes())
        if isinstance(obj, str) and obj.strip():
            return obj.strip()
        if isinstance(obj, bytes):
            try:
                s = obj.decode("utf-8").strip()
                if s:
                    return s
            except Exception:
                pass
        if isinstance(obj, dict):
            for k in ["dataset_id", "preprocessed_dataset_id", "id"]:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return None


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_(empty)_\n"
    head = df.head(max_rows)
    cols = [str(c) for c in head.columns]
    lines: List[str] = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in head.iterrows():
        vals = []
        for c in cols:
            try:
                v = row.get(c)
            except Exception:
                v = ""
            s = "" if v is None else str(v)
            s = s.replace("|", "\\|")
            vals.append(s)
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > len(head):
        lines.append(f"\n_(showing first {len(head)} of {len(df)} rows)_\n")
    return "\n".join(lines) + "\n"


def _short_list(values: Optional[List[str]], *, max_items: int = 20) -> str:
    if not values:
        return ""
    items = [str(v) for v in values if str(v).strip()]
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f" ... (+{len(items) - max_items})"


def _format_seconds(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if v < 0:
        return f"{v:.3f}s"
    if v < 1:
        return f"{v * 1000:.0f}ms"
    if v < 60:
        return f"{v:.2f}s"
    m = v / 60.0
    if m < 60:
        return f"{m:.1f}m ({v:.0f}s)"
    h = m / 60.0
    return f"{h:.1f}h ({m:.0f}m)"


def _format_bytes(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return ""
    if v < 0:
        return f"{v:.0f}B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while v >= 1024 and idx < len(units) - 1:
        v /= 1024.0
        idx += 1
    if idx == 0:
        return f"{v:.0f}{units[idx]}"
    return f"{v:.2f}{units[idx]}"


def _download_task_artifact(task_id: str, artifact_name: str, dest_path: Path) -> bool:
    if dest_path.exists():
        return True
    if not (task_id and str(task_id).strip()):
        return False
    if not (artifact_name and str(artifact_name).strip()):
        return False

    try:
        from automl_lib.clearml.bootstrap import ensure_clearml_config_file

        ensure_clearml_config_file()
        from clearml import Task  # type: ignore
    except Exception:
        return False

    try:
        task = Task.get_task(task_id=str(task_id))
    except Exception:
        return False
    if not task:
        return False

    try:
        artifacts = getattr(task, "artifacts", None)
        if not isinstance(artifacts, dict):
            return False
        art = artifacts.get(str(artifact_name))
        if not art:
            return False
    except Exception:
        return False

    local_path = None
    try:
        local_path = art.get_local_copy()
    except Exception:
        try:
            local_path = art.get_local_copy(extract_archive=True)
        except Exception:
            local_path = None

    if not local_path:
        return False
    src = Path(str(local_path))
    if not src.exists():
        return False

    # Some artifacts can be extracted into a directory; try to locate the expected file.
    if src.is_dir():
        candidate = src / str(artifact_name)
        if candidate.exists():
            src = candidate
        else:
            files = [p for p in src.rglob("*") if p.is_file()]
            if len(files) == 1:
                src = files[0]
            else:
                return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(src, dest_path)
    except Exception:
        try:
            dest_path.write_bytes(src.read_bytes())
        except Exception:
            return False
    return dest_path.exists()


def _find_task_id_by_tags(
    *,
    project_name: Optional[str],
    tags: List[str],
    required_artifacts: Optional[List[str]] = None,
) -> Optional[str]:
    if not tags:
        return None
    try:
        from automl_lib.clearml.bootstrap import ensure_clearml_config_file

        ensure_clearml_config_file()
        from clearml import Task  # type: ignore
    except Exception:
        return None

    def _has_required_artifacts(task_obj) -> bool:
        if not required_artifacts:
            return False
        try:
            artifacts = getattr(task_obj, "artifacts", None)
            if not isinstance(artifacts, dict):
                return False
            return any(str(name) in artifacts for name in required_artifacts)
        except Exception:
            return False

    best_fallback: Optional[str] = None
    # Try within the expected project first, then fall back to global search.
    for proj in [project_name, None] if project_name else [None]:
        try:
            tasks = Task.get_tasks(project_name=proj, tags=tags, allow_archived=True)
        except Exception:
            tasks = []
        for t in tasks:
            tid = None
            try:
                tid = getattr(t, "id", None)
            except Exception:
                tid = None
            if not tid:
                continue
            tid_str = str(tid)
            if best_fallback is None:
                best_fallback = tid_str
            if not required_artifacts:
                return tid_str
            # Prefer tasks that actually contain the artifacts we need (e.g. training-summary vs child tasks).
            if _has_required_artifacts(t):
                return tid_str
            try:
                full = Task.get_task(task_id=tid_str)
                if full and _has_required_artifacts(full):
                    return tid_str
            except Exception:
                pass
    return best_fallback


def _get_task_output_url(task_id: str) -> Optional[str]:
    if not (task_id and str(task_id).strip()):
        return None
    try:
        from automl_lib.clearml.bootstrap import ensure_clearml_config_file

        ensure_clearml_config_file()
        from clearml import Task  # type: ignore
    except Exception:
        return None
    try:
        t = Task.get_task(task_id=str(task_id))
    except Exception:
        return None
    if not t:
        return None
    try:
        return str(t.get_output_log_web_page())
    except Exception:
        return None


def _get_dataset_meta(dataset_id: str) -> Optional[Dict[str, Any]]:
    if not (dataset_id and str(dataset_id).strip()):
        return None
    try:
        from automl_lib.clearml.bootstrap import ensure_clearml_config_file

        ensure_clearml_config_file()
        from clearml import Dataset  # type: ignore
    except Exception:
        return None
    try:
        ds = Dataset.get(dataset_id=str(dataset_id), include_archived=True)
    except Exception:
        return None
    if not ds:
        return None
    url = None
    try:
        task = getattr(ds, "_task", None)
        if task:
            url = task.get_output_log_web_page()
    except Exception:
        url = None
    try:
        return {
            "dataset_id": str(getattr(ds, "id", "") or dataset_id),
            "name": str(getattr(ds, "name", "") or ""),
            "project": str(getattr(ds, "project", "") or ""),
            "version": str(getattr(ds, "version", "") or ""),
            "tags": list(getattr(ds, "tags", []) or []),
            "url": str(url) if url else "",
        }
    except Exception:
        return None


def _pareto_front_mask(df: pd.DataFrame, objectives: List[tuple[str, str]]) -> Optional[pd.Series]:
    """
    Compute Pareto front mask for given objectives.

    objectives: list of (column_name, direction) where direction is 'min' or 'max'.
    Returns a boolean Series aligned to df.index, or None if insufficient columns/rows.
    """

    cols: List[str] = []
    dirs: List[str] = []
    for col, direction in objectives:
        if col in df.columns:
            cols.append(str(col))
            dirs.append(str(direction).strip().lower())
    if len(cols) < 2:
        return None

    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    work = df.copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=cols)
    if work.empty:
        return None

    arr = work[cols].to_numpy(dtype=float)
    # Convert to minimization
    for i, d in enumerate(dirs):
        if d == "max":
            arr[:, i] = -arr[:, i]

    n = arr.shape[0]
    pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                pareto[i] = False
                break

    mask = pd.Series(False, index=df.index)
    mask.loc[work.index] = pareto.tolist()
    return mask


def run_reporting_processing(
    config_path: Path,
    *,
    preprocessing_config_path: Optional[Path] = None,
    preprocessing_info: Optional[Dict[str, Any]] = None,
    training_info: Optional[Dict[str, Any]] = None,
    pipeline_task_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a ClearML "reporting task" (and a local markdown artifact) for a run.
    Intended to be used as the last step in a ClearML pipeline.
    """

    config_path = Path(config_path)
    cfg = load_training_config(config_path)
    try:
        from automl_lib.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass

    pre_info = preprocessing_info or {}
    tr_info = training_info or {}
    run_id = resolve_run_id(
        explicit=run_id,
        from_input=(tr_info.get("run_id") or pre_info.get("run_id")),
        from_config=getattr(cfg.run, "id", None),
        from_env=get_run_id_env(),
    )
    set_run_id_env(run_id)

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    dataset_id = tr_info.get("dataset_id") or pre_info.get("dataset_id") or getattr(cfg.data, "dataset_id", None)
    csv_path = pre_info.get("csv_path") or getattr(cfg.data, "csv_path", None)

    dataset_key = resolve_dataset_key(
        explicit=getattr(cfg.run, "dataset_key", None),
        dataset_id=str(dataset_id) if dataset_id else None,
        csv_path=csv_path,
    )

    # Resolve preprocessing output_dir for metadata lookup
    preproc_out_base = Path("outputs/preprocessing")
    if preprocessing_config_path:
        try:
            pre_cfg = load_preprocessing_config(Path(preprocessing_config_path))
            if pre_cfg.output and getattr(pre_cfg.output, "output_dir", None):
                preproc_out_base = Path(pre_cfg.output.output_dir)
        except Exception:
            preproc_out_base = Path("outputs/preprocessing")
    preproc_out_dir = run_scoped_output_dir(preproc_out_base, run_id)

    # Training output_dir for artifacts lookup
    train_out_dir = run_scoped_output_dir(Path(cfg.output.output_dir), run_id)

    # Reporting output_dir
    reporting_base = Path(cfg.output.output_dir).parent / "reporting"
    output_dir = run_scoped_output_dir(reporting_base, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    rep_cfg = getattr(cfg, "reporting", None)
    top_k = 5
    max_plot_candidates = 200
    max_failures_rows = 50
    include_failures = True
    include_tradeoff_plots = True
    include_task_links = True
    resolve_from_clearml = True
    project_suffix = "reports"
    try:
        top_k = int(getattr(rep_cfg, "top_k", 5))
    except Exception:
        top_k = 5
    try:
        max_plot_candidates = int(getattr(rep_cfg, "max_plot_candidates", 200))
    except Exception:
        max_plot_candidates = 200
    try:
        max_failures_rows = int(getattr(rep_cfg, "max_failures_rows", 50))
    except Exception:
        max_failures_rows = 50
    try:
        include_failures = bool(getattr(rep_cfg, "include_failures", True))
    except Exception:
        include_failures = True
    try:
        include_tradeoff_plots = bool(getattr(rep_cfg, "include_tradeoff_plots", True))
    except Exception:
        include_tradeoff_plots = True
    try:
        include_task_links = bool(getattr(rep_cfg, "include_task_links", True))
    except Exception:
        include_task_links = True
    try:
        resolve_from_clearml = bool(getattr(rep_cfg, "resolve_from_clearml", True))
    except Exception:
        resolve_from_clearml = True
    try:
        project_suffix = str(getattr(rep_cfg, "project_suffix", "reports") or "reports").strip() or "reports"
    except Exception:
        project_suffix = "reports"

    if top_k < 1:
        top_k = 1
    if max_plot_candidates < 1:
        max_plot_candidates = 1
    if max_failures_rows < 1:
        max_failures_rows = 1

    clearml_available = bool(cfg.clearml and getattr(cfg.clearml, "enabled", False))
    resolve_clearml = bool(clearml_available and resolve_from_clearml)
    include_links = bool(resolve_clearml and include_task_links)

    # Build a slim config dict for task creation / params
    cfg_dump = cfg.model_dump()
    try:
        cfg_dump.setdefault("run", {})
        cfg_dump["run"]["id"] = run_id
        cfg_dump["run"]["dataset_key"] = dataset_key
    except Exception:
        pass
    try:
        cfg_dump.setdefault("data", {})
        if dataset_id:
            cfg_dump["data"]["dataset_id"] = str(dataset_id)
        if csv_path:
            cfg_dump["data"]["csv_path"] = str(csv_path)
    except Exception:
        pass
    cfg_dump["reporting"] = {
        "top_k": top_k,
        "max_plot_candidates": max_plot_candidates,
        "include_failures": include_failures,
        "max_failures_rows": max_failures_rows,
        "include_tradeoff_plots": include_tradeoff_plots,
        "resolve_from_clearml": resolve_from_clearml,
        "include_task_links": include_task_links,
        "project_suffix": project_suffix,
        "pipeline_task_id": str(pipeline_task_id or ""),
        "preprocessing_task_id": "",
        "training_summary_task_id": "",
    }

    pre_task_id = str(pre_info.get("task_id") or "")
    tr_task_id = str(tr_info.get("task_id") or "")
    if resolve_clearml:
        results_csv_name = str(getattr(cfg.output, "results_csv", None) or "results_summary.csv")
        project_base = None
        try:
            from automl_lib.clearml.context import build_run_context
            from automl_lib.clearml.naming import build_project_path

            naming_cfg = getattr(cfg.clearml, "naming", None)
            project_mode = getattr(naming_cfg, "project_mode", "root")
            ctx = build_run_context(
                run_id=run_id,
                dataset_key=dataset_key,
                project_root=(cfg.clearml.project_name if cfg.clearml else None),
                dataset_project=(cfg.clearml.dataset_project if cfg.clearml else None),
                user=getattr(cfg.run, "user", None),
            )
            project_base = build_project_path(ctx, project_mode=project_mode)
        except Exception:
            project_base = str(cfg.clearml.project_name) if cfg.clearml and cfg.clearml.project_name else None

        if not pre_task_id:
            pre_task_id = _find_task_id_by_tags(
                project_name=project_base,
                tags=[f"run:{run_id}", "phase:preprocessing"],
                required_artifacts=["schema_raw.json", "preprocess_pipeline.json", "preprocessing_timing.json"],
            ) or ""
        if not tr_task_id:
            tr_task_id = _find_task_id_by_tags(
                project_name=project_base,
                tags=[f"run:{run_id}", "phase:training"],
                required_artifacts=[
                    "recommended_model.csv",
                    "recommendation_rationale.json",
                    "model_tasks_ranked.csv",
                    "model_task_failures.csv",
                    results_csv_name,
                ],
            ) or ""

    cfg_dump["reporting"]["preprocessing_task_id"] = str(pre_task_id or "")
    cfg_dump["reporting"]["training_summary_task_id"] = str(tr_task_id or "")

    # Create reporting task
    parent_for_report = pipeline_task_id or tr_task_id or pre_task_id or None
    task_info = create_reporting_task(cfg_dump, parent_task_id=str(parent_for_report) if parent_for_report else None)
    task = task_info.get("task")
    logger = task_info.get("logger")

    if task:
        # Extra linkage tag (helps trace the report back to its pipeline controller task).
        if pipeline_task_id:
            try:
                task.add_tags([f"pipeline:{pipeline_task_id}"])
            except Exception:
                pass

    # If we don't have local outputs (remote/distributed pipeline), try to download required artifacts.
    if resolve_clearml:
        try:
            if pre_task_id:
                _download_task_artifact(pre_task_id, "schema_raw.json", preproc_out_dir / "schema_raw.json")
                _download_task_artifact(
                    pre_task_id,
                    "preprocess_pipeline.json",
                    preproc_out_dir / "preprocess_pipeline.json",
                )
                _download_task_artifact(
                    pre_task_id,
                    "preprocessing_timing.json",
                    preproc_out_dir / "preprocessing_timing.json",
                )
                _download_task_artifact(
                    pre_task_id,
                    "preprocessed_dataset_id",
                    preproc_out_dir / "preprocessed_dataset_id.artifact",
                )
                _download_task_artifact(
                    pre_task_id,
                    "preprocessed_dataset_id.txt",
                    preproc_out_dir / "preprocessed_dataset_id.txt",
                )
        except Exception:
            pass
        results_csv_name = str(getattr(cfg.output, "results_csv", None) or "results_summary.csv")
        try:
            if tr_task_id:
                _download_task_artifact(tr_task_id, "recommended_model.csv", train_out_dir / "recommended_model.csv")
                _download_task_artifact(
                    tr_task_id,
                    "recommendation_rationale.md",
                    train_out_dir / "recommendation_rationale.md",
                )
                _download_task_artifact(
                    tr_task_id,
                    "recommendation_rationale.json",
                    train_out_dir / "recommendation_rationale.json",
                )
                _download_task_artifact(tr_task_id, "model_tasks_ranked.csv", train_out_dir / "model_tasks_ranked.csv")
                _download_task_artifact(
                    tr_task_id,
                    "model_task_failures.csv",
                    train_out_dir / "model_task_failures.csv",
                )
                _download_task_artifact(tr_task_id, results_csv_name, train_out_dir / results_csv_name)
        except Exception:
            pass

    # Collect preprocessing metadata
    schema = _read_json(preproc_out_dir / "schema_raw.json") or {}
    pipeline_meta = _read_json(preproc_out_dir / "preprocess_pipeline.json") or {}
    timing = _read_json(preproc_out_dir / "preprocessing_timing.json") or {}

    target_col = str(schema.get("target_column") or getattr(cfg.data, "target_column", "") or "")
    feature_cols = schema.get("feature_columns") if isinstance(schema.get("feature_columns"), list) else None
    numeric_cols = None
    categorical_cols = None
    try:
        ft = schema.get("feature_types") if isinstance(schema.get("feature_types"), dict) else {}
        numeric_cols = ft.get("numeric") if isinstance(ft.get("numeric"), list) else None
        categorical_cols = ft.get("categorical") if isinstance(ft.get("categorical"), list) else None
    except Exception:
        numeric_cols = None
        categorical_cols = None

    dataset_id_source = str(timing.get("dataset_id_source") or getattr(cfg.data, "dataset_id", "") or "")
    preprocessed_dataset_id = (
        str(timing.get("dataset_id_preprocessed") or "").strip()
        or _read_text_or_pickle_string(preproc_out_dir / "preprocessed_dataset_id.txt")
        or _read_text_or_pickle_string(preproc_out_dir / "preprocessed_dataset_id.artifact")
        or None
    )
    try:
        if preprocessed_dataset_id and not (preproc_out_dir / "preprocessed_dataset_id.txt").exists():
            (preproc_out_dir / "preprocessed_dataset_id.txt").write_text(str(preprocessed_dataset_id), encoding="utf-8")
    except Exception:
        pass
    enable_preprocessing = bool(getattr(getattr(cfg, "clearml", None), "enable_preprocessing", False))
    dataset_id_preprocessed = str(
        (
            pre_info.get("dataset_id")
            or tr_info.get("dataset_id")
            or preprocessed_dataset_id
            or getattr(getattr(cfg, "clearml", None), "preprocessed_dataset_id", None)
            or ""
        )
        or ""
    )
    if not dataset_id_preprocessed:
        fallback_id = str(dataset_id or "")
        if fallback_id:
            if (not enable_preprocessing) or (not dataset_id_source) or (fallback_id != dataset_id_source):
                dataset_id_preprocessed = fallback_id
    dataset_id_for_reporting = dataset_id_preprocessed or str(dataset_id or "") or None

    ds_source_meta = _get_dataset_meta(dataset_id_source) if (include_links and dataset_id_source) else None
    ds_pre_meta = _get_dataset_meta(dataset_id_preprocessed) if (include_links and dataset_id_preprocessed) else None

    dataset_summary = {
        "run_id": run_id,
        "dataset_key": dataset_key,
        "dataset_id_source": dataset_id_source,
        "dataset_source_name": (ds_source_meta.get("name") if ds_source_meta else ""),
        "dataset_source_project": (ds_source_meta.get("project") if ds_source_meta else ""),
        "dataset_source_version": (ds_source_meta.get("version") if ds_source_meta else ""),
        "dataset_source_url": (ds_source_meta.get("url") if ds_source_meta else ""),
        "dataset_id_preprocessed": dataset_id_preprocessed,
        "dataset_preprocessed_name": (ds_pre_meta.get("name") if ds_pre_meta else ""),
        "dataset_preprocessed_project": (ds_pre_meta.get("project") if ds_pre_meta else ""),
        "dataset_preprocessed_version": (ds_pre_meta.get("version") if ds_pre_meta else ""),
        "dataset_preprocessed_url": (ds_pre_meta.get("url") if ds_pre_meta else ""),
        "csv_path": str(csv_path or ""),
        "target_column": target_col,
        "n_rows": schema.get("n_rows"),
        "n_features_raw": schema.get("n_features_raw"),
        "n_features_preprocessed": schema.get("n_features_preprocessed"),
        "n_numeric": len(numeric_cols) if numeric_cols else None,
        "n_categorical": len(categorical_cols) if categorical_cols else None,
        "feature_columns_sample": _short_list([str(c) for c in (feature_cols or [])]),
    }
    df_dataset = pd.DataFrame([dataset_summary])

    preproc_summary = {
        "run_id": run_id,
        "dataset_key": dataset_key,
        "selected_preprocessor": str(pipeline_meta.get("selected_preprocessor") or ""),
        "preprocessed_dataset_id": dataset_id_preprocessed,
        "preprocessing_task_id": str(pre_task_id or ""),
        "n_rows": schema.get("n_rows"),
        "n_features_raw": schema.get("n_features_raw"),
        "n_features_preprocessed": schema.get("n_features_preprocessed"),
        "time_total_seconds": timing.get("total_seconds"),
        "time_load_dataset_seconds": timing.get("load_dataset_seconds"),
        "time_fit_transform_seconds": timing.get("fit_transform_seconds"),
        "time_register_dataset_seconds": timing.get("register_dataset_seconds"),
    }
    df_preproc = pd.DataFrame([preproc_summary])

    # Training / recommendation artifacts
    rationale_md_path = train_out_dir / "recommendation_rationale.md"
    rationale_json_path = train_out_dir / "recommendation_rationale.json"
    rec_csv_path = train_out_dir / "recommended_model.csv"
    ranked_csv_path = train_out_dir / "model_tasks_ranked.csv"
    failures_csv_path = train_out_dir / "model_task_failures.csv"
    results_csv_path = train_out_dir / (getattr(cfg.output, "results_csv", None) or "results_summary.csv")

    missing_artifacts: List[str] = []
    df_missing = pd.DataFrame()
    try:
        expected = [
            ("preprocessing/schema_raw.json", preproc_out_dir / "schema_raw.json"),
            ("preprocessing/preprocess_pipeline.json", preproc_out_dir / "preprocess_pipeline.json"),
            ("preprocessing/preprocessing_timing.json", preproc_out_dir / "preprocessing_timing.json"),
            ("preprocessing/preprocessed_dataset_id.txt", preproc_out_dir / "preprocessed_dataset_id.txt"),
            ("training/recommended_model.csv", rec_csv_path),
            ("training/recommendation_rationale.md", rationale_md_path),
            ("training/recommendation_rationale.json", rationale_json_path),
            ("training/model_tasks_ranked.csv", ranked_csv_path),
            ("training/model_task_failures.csv", failures_csv_path),
            ("training/results_csv", results_csv_path),
        ]
        for label, path in expected:
            try:
                if not Path(path).exists():
                    missing_artifacts.append(str(label))
            except Exception:
                missing_artifacts.append(str(label))
        if missing_artifacts:
            df_missing = pd.DataFrame({"missing": missing_artifacts})
    except Exception:
        missing_artifacts = []
        df_missing = pd.DataFrame()

    recommend_metric = ""
    recommend_goal = ""
    try:
        rationale = _read_json(rationale_json_path) or {}
        recommend_metric = str(rationale.get("recommend_metric") or "")
        recommend_goal = str(rationale.get("recommend_goal") or "")
    except Exception:
        recommend_metric = ""
        recommend_goal = ""

    df_top5 = pd.DataFrame()
    df_rec = pd.DataFrame()
    df_ranked = pd.DataFrame()
    try:
        if rec_csv_path.exists():
            df_rec = pd.read_csv(rec_csv_path)
    except Exception:
        df_rec = pd.DataFrame()
    try:
        if ranked_csv_path.exists():
            df_ranked = pd.read_csv(ranked_csv_path)
    except Exception:
        df_top5 = pd.DataFrame()
        df_ranked = pd.DataFrame()

    if not recommend_metric and not df_rec.empty:
        try:
            recommend_metric = str(df_rec.iloc[0].get("primary_metric") or "").strip()
        except Exception:
            recommend_metric = recommend_metric

    # Add Pareto flag to ranked candidates (used in decision tables/plots).
    if not df_ranked.empty and recommend_metric and recommend_metric in df_ranked.columns:
        try:
            goal_norm = str(recommend_goal or "").strip().lower()
            metric_dir = "min" if goal_norm == "min" else "max"
            df_ranked["pareto"] = False
            df_subset = df_ranked.head(max_plot_candidates).copy()
            pareto_mask = _pareto_front_mask(
                df_subset,
                objectives=[
                    (recommend_metric, metric_dir),
                    ("train_seconds", "min"),
                    ("predict_seconds", "min"),
                    ("model_size_bytes", "min"),
                ],
            )
            if pareto_mask is not None:
                df_ranked.loc[df_subset.index, "pareto"] = pareto_mask.loc[df_subset.index].astype(bool)
        except Exception:
            pass

    try:
        if not df_ranked.empty:
            df_top5 = df_ranked.head(top_k)
    except Exception:
        df_top5 = pd.DataFrame()

    # Write markdown report
    md_lines: List[str] = []
    md_lines.append(f"# AutoML Report — ds:{dataset_key} run:{run_id}")
    md_lines.append("")
    md_lines.append("## Run")
    md_lines.append("")
    md_lines.append(f"- run_id: `{run_id}`")
    md_lines.append(f"- dataset_key: `{dataset_key}`")
    if pipeline_task_id:
        md_lines.append(f"- pipeline_task_id: `{pipeline_task_id}`")
    if pre_task_id:
        md_lines.append(f"- preprocessing_task_id: `{pre_task_id}`")
    if tr_task_id:
        md_lines.append(f"- training_summary_task_id: `{tr_task_id}`")
    if include_links and task:
        try:
            md_lines.append(f"- reporting_task_id: `{task.id}`")
        except Exception:
            pass
    md_lines.append("")

    if include_links:
        pipeline_url = _get_task_output_url(str(pipeline_task_id or "")) if pipeline_task_id else None
        pre_url = _get_task_output_url(pre_task_id) if pre_task_id else None
        train_url = _get_task_output_url(tr_task_id) if tr_task_id else None
        report_url = None
        try:
            if task:
                report_url = task.get_output_log_web_page()
        except Exception:
            report_url = None

        if any([pipeline_url, pre_url, train_url, report_url]):
            md_lines.append("## Links")
            md_lines.append("")
            if pipeline_task_id:
                md_lines.append(f"- pipeline: `{pipeline_task_id}`" + (f" ({pipeline_url})" if pipeline_url else ""))
            if pre_task_id:
                md_lines.append(f"- preprocessing: `{pre_task_id}`" + (f" ({pre_url})" if pre_url else ""))
            if tr_task_id:
                md_lines.append(f"- training-summary: `{tr_task_id}`" + (f" ({train_url})" if train_url else ""))
            if task:
                try:
                    md_lines.append(f"- reporting: `{task.id}`" + (f" ({report_url})" if report_url else ""))
                except Exception:
                    pass
            if ds_source_meta and ds_source_meta.get("url"):
                md_lines.append(
                    f"- dataset(source): `{dataset_id_source}`"
                    + (f" ({ds_source_meta.get('name')})" if ds_source_meta.get("name") else "")
                    + f" ({ds_source_meta.get('url')})"
                )
            if ds_pre_meta and ds_pre_meta.get("url"):
                md_lines.append(
                    f"- dataset(preprocessed): `{dataset_id_preprocessed}`"
                    + (f" ({ds_pre_meta.get('name')})" if ds_pre_meta.get("name") else "")
                    + f" ({ds_pre_meta.get('url')})"
                )
            md_lines.append("")

    md_lines.append("## Dataset")
    md_lines.append("")
    dataset_cols = [
        c
        for c in [
            "dataset_id_source",
            "dataset_source_name",
            "dataset_source_project",
            "dataset_source_version",
            "dataset_id_preprocessed",
            "dataset_preprocessed_name",
            "dataset_preprocessed_project",
            "dataset_preprocessed_version",
            "target_column",
            "n_rows",
            "n_features_raw",
            "n_features_preprocessed",
            "feature_columns_sample",
        ]
        if c in df_dataset.columns
    ]
    md_lines.append(_markdown_table(df_dataset[dataset_cols] if dataset_cols else df_dataset, max_rows=1).strip())
    md_lines.append("")
    if numeric_cols:
        md_lines.append(f"- numeric_columns (sample): {_short_list([str(c) for c in numeric_cols])}")
    if categorical_cols:
        md_lines.append(f"- categorical_columns (sample): {_short_list([str(c) for c in categorical_cols])}")
    md_lines.append("")

    md_lines.append("## Preprocessing")
    md_lines.append("")
    preproc_cols = [
        c
        for c in [
            "selected_preprocessor",
            "preprocessed_dataset_id",
            "preprocessing_task_id",
            "n_rows",
            "n_features_raw",
            "n_features_preprocessed",
            "time_total_seconds",
            "time_load_dataset_seconds",
            "time_fit_transform_seconds",
            "time_register_dataset_seconds",
        ]
        if c in df_preproc.columns
    ]
    md_lines.append(_markdown_table(df_preproc[preproc_cols] if preproc_cols else df_preproc, max_rows=1).strip())
    md_lines.append("")

    md_lines.append("## Recommendation")
    md_lines.append("")
    if not df_rec.empty:
        md_lines.append(_markdown_table(df_rec, max_rows=1).strip())
    else:
        md_lines.append("_(no recommended_model.csv found)_")
    md_lines.append("")
    if rationale_md_path.exists():
        try:
            md_lines.append(rationale_md_path.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    md_lines.append("")

    md_lines.append("## Decision Summary")
    md_lines.append("")
    if recommend_metric:
        try:
            goal_norm = str(recommend_goal or "").strip().lower()
            metric_dir = "min" if goal_norm == "min" else "max"
            md_lines.append(
                f"- primary_metric: `{recommend_metric}` ({'lower' if metric_dir == 'min' else 'higher'} is better)"
            )
        except Exception:
            md_lines.append(f"- primary_metric: `{recommend_metric}`")
    md_lines.append("- speed: lower is better (`train_seconds`, `predict_seconds`)")
    md_lines.append("- size: lower is better (`model_size_bytes`)")
    if "pareto" in df_ranked.columns:
        md_lines.append("- `pareto=True` indicates a good trade-off candidate (within top plotted candidates).")
    md_lines.append("")
    df_decision = pd.DataFrame()
    df_alts = pd.DataFrame()
    try:
        if not df_ranked.empty:
            rec_idx = None
            if "is_recommended" in df_ranked.columns:
                try:
                    s = df_ranked["is_recommended"]
                    if str(s.dtype) == "bool":
                        mask = s
                    else:
                        mask = s.astype(str).str.lower().isin(["1", "true", "yes", "y"])
                    rec_rows = df_ranked[mask]
                    if not rec_rows.empty:
                        rec_idx = rec_rows.index[0]
                except Exception:
                    rec_idx = None
            if rec_idx is None:
                rec_idx = df_ranked.index[0]
            cols = [
                c
                for c in [
                    "rank",
                    "is_recommended",
                    "pareto",
                    "model",
                    "preprocessor",
                    "metric_source",
                    recommend_metric,
                    "composite_score",
                    "train_seconds",
                    "predict_seconds",
                    "model_size_bytes",
                    "num_features",
                    "task_id",
                    "url",
                ]
                if c and c in df_ranked.columns
            ]
            df_decision = df_ranked.loc[[rec_idx], cols] if cols else df_ranked.loc[[rec_idx]]
    except Exception:
        df_decision = pd.DataFrame()
    if not df_decision.empty:
        md_lines.append(_markdown_table(df_decision, max_rows=1).strip())
        try:
            row = df_decision.iloc[0].to_dict()
            model_name = str(row.get("model") or "").strip()
            preproc_name = str(row.get("preprocessor") or "").strip()
            metric_val = row.get(recommend_metric) if recommend_metric else None
            metric_text = str(metric_val) if metric_val is not None and str(metric_val) != "nan" else ""
            train_text = _format_seconds(row.get("train_seconds"))
            pred_text = _format_seconds(row.get("predict_seconds"))
            size_text = _format_bytes(row.get("model_size_bytes"))
            parts = []
            if model_name:
                parts.append(f"model={model_name}")
            if preproc_name:
                parts.append(f"preproc={preproc_name}")
            if metric_text and recommend_metric:
                parts.append(f"{recommend_metric}={metric_text}")
            if train_text:
                parts.append(f"train={train_text}")
            if pred_text:
                parts.append(f"predict={pred_text}")
            if size_text:
                parts.append(f"size={size_text}")
            if parts:
                md_lines.append("- recommended: " + ", ".join(parts))
        except Exception:
            pass
    else:
        md_lines.append("_(no ranking data found)_")
    md_lines.append("")

    # Alternatives (within top-N candidates) for speed/size trade-offs.
    try:
        if not df_ranked.empty:
            pool_n = min(len(df_ranked), max(top_k, 20))
            df_pool = df_ranked.head(pool_n).copy()
            for c in [recommend_metric, "train_seconds", "predict_seconds", "model_size_bytes"]:
                if c and c in df_pool.columns:
                    df_pool[c] = pd.to_numeric(df_pool[c], errors="coerce")
            rec_task_id = ""
            try:
                if not df_decision.empty and "task_id" in df_decision.columns:
                    rec_task_id = str(df_decision.iloc[0].get("task_id") or "").strip()
            except Exception:
                rec_task_id = ""
            if rec_task_id and "task_id" in df_pool.columns:
                df_pool = df_pool[df_pool["task_id"].astype(str) != rec_task_id]

            choices: List[Dict[str, Any]] = []

            def _pick_min(label: str, col: str) -> None:
                if col not in df_pool.columns:
                    return
                use = df_pool.dropna(subset=[col]).copy()
                if use.empty:
                    return
                by = [col] + (["rank"] if "rank" in use.columns else [])
                use = use.sort_values(by=by, ascending=True, na_position="last")
                d = use.iloc[0].to_dict()
                d["choice"] = label
                choices.append(d)

            # Best Pareto (if available)
            try:
                if "pareto" in df_pool.columns:
                    mask = df_pool["pareto"].astype(str).str.lower().isin(["1", "true", "yes", "y"])
                    use = df_pool[mask].copy()
                    if not use.empty:
                        by = ["rank"] if "rank" in use.columns else [recommend_metric] if recommend_metric in use.columns else None
                        if by:
                            use = use.sort_values(by=by, ascending=True, na_position="last")
                        d = use.iloc[0].to_dict()
                        d["choice"] = "pareto_best"
                        choices.append(d)
            except Exception:
                pass

            _pick_min("fastest_train", "train_seconds")
            _pick_min("fastest_predict", "predict_seconds")
            _pick_min("smallest_model", "model_size_bytes")

            if choices:
                df_alts = pd.DataFrame(choices)
                if "task_id" in df_alts.columns:
                    df_alts = df_alts.drop_duplicates(subset=["task_id"], keep="first")
                elif all(c in df_alts.columns for c in ["model", "preprocessor"]):
                    df_alts = df_alts.drop_duplicates(subset=["model", "preprocessor"], keep="first")
                cols = [
                    c
                    for c in [
                        "choice",
                        "rank",
                        "pareto",
                        "model",
                        "preprocessor",
                        recommend_metric,
                        "train_seconds",
                        "predict_seconds",
                        "model_size_bytes",
                        "task_id",
                        "url",
                    ]
                    if c and c in df_alts.columns
                ]
                df_alts = df_alts[cols] if cols else df_alts
    except Exception:
        df_alts = pd.DataFrame()

    if not df_alts.empty:
        md_lines.append("### Alternatives (speed / size)")
        md_lines.append("")
        md_lines.append(_markdown_table(df_alts, max_rows=10).strip())
        md_lines.append("")

    md_lines.append(f"## Top {top_k} candidates (accuracy / speed / size)")
    md_lines.append("")
    if recommend_metric:
        md_lines.append(f"- ranking_metric: `{recommend_metric}` ({recommend_goal or 'auto'})")
        md_lines.append("")
    if not df_top5.empty:
        # Keep only a concise set of columns for decision making.
        cols = [
            c
            for c in [
                "rank",
                "is_recommended",
                "pareto",
                "model",
                "preprocessor",
                "metric_source",
                recommend_metric,
                "composite_score",
                "train_seconds",
                "predict_seconds",
                "model_size_bytes",
                "num_features",
                "url",
            ]
            if c and c in df_top5.columns
        ]
        md_lines.append(_markdown_table(df_top5[cols] if cols else df_top5, max_rows=top_k).strip())
    else:
        md_lines.append("_(no model_tasks_ranked.csv found)_")
    md_lines.append("")

    try:
        if include_failures and failures_csv_path.exists():
            df_fail = pd.read_csv(failures_csv_path)
            if not df_fail.empty:
                md_lines.append("## Failures / Skipped")
                md_lines.append("")
                cols = [c for c in ["model", "preprocessor", "status", "error"] if c in df_fail.columns]
                md_lines.append(_markdown_table(df_fail[cols] if cols else df_fail, max_rows=max_failures_rows).strip())
                md_lines.append("")
    except Exception:
        pass

    if missing_artifacts:
        md_lines.append("## Debug: Missing artifacts")
        md_lines.append("")
        md_lines.append("This can happen when running remotely and required artifacts were not uploaded.")
        md_lines.append("")
        for item in missing_artifacts:
            md_lines.append(f"- {item}")
        md_lines.append("")

    report_md = output_dir / "report.md"
    report_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    # ClearML reporting outputs
    if logger:
        try:
            report_table(logger, "01_dataset/summary", df_dataset, series="dataset")
            report_table(logger, "02_preprocessing/summary", df_preproc, series="preprocessing")
            if not df_rec.empty:
                report_table(logger, "03_training/recommended", df_rec, series="training")
            if not df_top5.empty:
                report_table(logger, f"03_training/top{top_k}", df_top5, series="training")
            if not df_decision.empty:
                report_table(logger, "04_decision/recommended_summary", df_decision, series="decision")
            if not df_alts.empty:
                report_table(logger, "04_decision/alternatives", df_alts, series="decision")
            if include_failures and failures_csv_path.exists():
                try:
                    df_fail = pd.read_csv(failures_csv_path)
                    if not df_fail.empty:
                        report_table(
                            logger,
                            "99_debug/model_task_failures",
                            df_fail.head(max_failures_rows),
                            series="debug",
                        )
                except Exception:
                    pass
            if not df_missing.empty:
                report_table(logger, "99_debug/missing_artifacts", df_missing, series="debug")
        except Exception:
            pass
        if include_tradeoff_plots and not df_ranked.empty and recommend_metric and recommend_metric in df_ranked.columns:
            try:
                import plotly.express as px  # type: ignore

                df_plot = df_ranked.head(max_plot_candidates).copy()
                df_plot[recommend_metric] = pd.to_numeric(df_plot[recommend_metric], errors="coerce")
                for col in ["train_seconds", "predict_seconds", "model_size_bytes", "num_features"]:
                    if col in df_plot.columns:
                        df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

                goal_norm = str(recommend_goal or "").strip().lower()
                metric_dir = "min" if goal_norm == "min" else "max"
                pareto_mask = _pareto_front_mask(
                    df_plot,
                    objectives=[
                        (recommend_metric, metric_dir),
                        ("train_seconds", "min"),
                        ("predict_seconds", "min"),
                        ("model_size_bytes", "min"),
                    ],
                )
                df_plot["pareto"] = pareto_mask if pareto_mask is not None else False

                # Optional: show Pareto candidates as a table
                try:
                    df_pareto = df_plot[df_plot["pareto"] == True]  # noqa: E712
                    if not df_pareto.empty:
                        cols = [
                            c
                            for c in [
                                "rank",
                                "is_recommended",
                                "model",
                                "preprocessor",
                                recommend_metric,
                                "train_seconds",
                                "predict_seconds",
                                "model_size_bytes",
                                "num_features",
                                "task_id",
                                "url",
                            ]
                            if c in df_pareto.columns
                        ]
                        report_table(
                            logger,
                            "04_decision/pareto_candidates",
                            df_pareto[cols].head(top_k) if cols else df_pareto.head(top_k),
                            series="decision",
                        )
                except Exception:
                    pass

                color_col = "pareto" if "pareto" in df_plot.columns else ("model" if "model" in df_plot.columns else None)
                symbol_col = "is_recommended" if "is_recommended" in df_plot.columns else None
                size_col = "model_size_bytes" if "model_size_bytes" in df_plot.columns else None
                hover_cols = [c for c in ["rank", "task_id", "url", "pareto", "is_recommended"] if c in df_plot.columns]

                def _save_plot(fig, *, title: str, html_name: str) -> None:
                    logger.report_plotly(title=title, series="decision", iteration=0, figure=fig)
                    try:
                        fig.write_html(str(output_dir / html_name))
                    except Exception:
                        pass

                # metric vs train_seconds
                if "train_seconds" in df_plot.columns:
                    use = df_plot.dropna(subset=[recommend_metric, "train_seconds"]).copy()
                    if not use.empty:
                        fig = px.scatter(
                            use,
                            x="train_seconds",
                            y=recommend_metric,
                            color=(color_col if color_col in use.columns else None),
                            symbol=(symbol_col if symbol_col and symbol_col in use.columns else None),
                            size=(size_col if size_col and size_col in use.columns else None),
                            hover_data=hover_cols,
                            title=f"Trade-off: {recommend_metric} vs train_seconds",
                        )
                        if metric_dir == "min":
                            fig.update_yaxes(autorange="reversed")
                        _save_plot(fig, title="04_decision/tradeoff_metric_vs_train_seconds", html_name="tradeoff_metric_vs_train_seconds.html")

                # metric vs predict_seconds
                if "predict_seconds" in df_plot.columns:
                    use = df_plot.dropna(subset=[recommend_metric, "predict_seconds"]).copy()
                    if not use.empty:
                        fig = px.scatter(
                            use,
                            x="predict_seconds",
                            y=recommend_metric,
                            color=(color_col if color_col in use.columns else None),
                            symbol=(symbol_col if symbol_col and symbol_col in use.columns else None),
                            size=(size_col if size_col and size_col in use.columns else None),
                            hover_data=hover_cols,
                            title=f"Trade-off: {recommend_metric} vs predict_seconds",
                        )
                        if metric_dir == "min":
                            fig.update_yaxes(autorange="reversed")
                        _save_plot(fig, title="04_decision/tradeoff_metric_vs_predict_seconds", html_name="tradeoff_metric_vs_predict_seconds.html")

                # metric vs model_size_bytes (log x)
                if "model_size_bytes" in df_plot.columns:
                    use = df_plot.dropna(subset=[recommend_metric, "model_size_bytes"]).copy()
                    if not use.empty:
                        fig = px.scatter(
                            use,
                            x="model_size_bytes",
                            y=recommend_metric,
                            color=(color_col if color_col in use.columns else None),
                            symbol=(symbol_col if symbol_col and symbol_col in use.columns else None),
                            hover_data=hover_cols,
                            title=f"Trade-off: {recommend_metric} vs model_size_bytes",
                        )
                        fig.update_xaxes(type="log")
                        if metric_dir == "min":
                            fig.update_yaxes(autorange="reversed")
                        _save_plot(fig, title="04_decision/tradeoff_metric_vs_model_size_bytes", html_name="tradeoff_metric_vs_model_size_bytes.html")
            except Exception:
                pass
        try:
            text = report_md.read_text(encoding="utf-8")
            logger.report_text(text.replace("\n", "<br/>"), title="00_report/report_md")
        except Exception:
            pass

    try:
        if task:
            upload_artifacts(task, [report_md])
            # Also attach referenced artifacts when available
            extras = [
                rationale_md_path,
                rationale_json_path,
                rec_csv_path,
                ranked_csv_path,
                failures_csv_path,
                results_csv_path,
            ]
            try:
                extras.extend(output_dir.glob("tradeoff_*.html"))
            except Exception:
                pass
            upload_artifacts(task, [p for p in extras if p.exists()])
            task.flush(wait_for_uploads=True)
    except Exception:
        pass

    # Keep compatibility with phase IO types where possible
    try:
        DatasetInfo(
            dataset_id=str(dataset_id_for_reporting) if dataset_id_for_reporting else None,
            task_id=(task.id if task else None),
            csv_path=str(csv_path) if csv_path else None,
            run_id=run_id,
        )
        TrainingInfo(
            dataset_id=tr_info.get("dataset_id"),
            task_id=tr_info.get("task_id"),
            training_task_ids=list(tr_info.get("training_task_ids") or []),
            metrics=(tr_info.get("metrics") if isinstance(tr_info.get("metrics"), list) else None),
            run_id=run_id,
        )
    except Exception:
        pass

    info = ReportingInfo(
        task_id=(task.id if task else None),
        dataset_id=(str(dataset_id_for_reporting) if dataset_id_for_reporting else None),
        output_dir=str(output_dir),
        report_md=str(report_md),
        run_id=run_id,
    ).model_dump()
    return info
