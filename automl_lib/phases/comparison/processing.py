"""
Processing for comparison phase.
ClearML 上の学習タスク群からメトリクスを集計し、比較タスクを作成する。
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from automl_lib.clearml import disable_resource_monitoring
from automl_lib.clearml.context import get_run_id_env, resolve_dataset_key, resolve_run_id, set_run_id_env
from automl_lib.clearml.logging import report_scalar, report_table
from automl_lib.config.loaders import load_comparison_config
from automl_lib.phases.comparison.clearml_integration import create_comparison_task, finalize_comparison_task
from automl_lib.phases.comparison.visualization import (
    render_comparison_visuals,
    render_model_summary_visuals,
    render_win_summary_visuals,
)
from automl_lib.phases.comparison.meta import build_comparison_metadata
from automl_lib.types import ComparisonInfo


def run_comparison_processing(
    config_path: Path,
    training_info: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    parent_task_id: Optional[Union[str, Sequence[str]]] = None,
    *,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute comparison phase based on training_info (expects training_task_ids).
    """
    config_path = Path(config_path)

    cfg = load_comparison_config(config_path)
    try:
        from automl_lib.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    clearml_cfg = cfg.clearml
    ranking_cfg = getattr(cfg, "ranking", None)
    from_input = None
    try:
        if isinstance(training_info, dict):
            from_input = training_info.get("run_id")
        elif isinstance(training_info, list) and len(training_info) == 1 and isinstance(training_info[0], dict):
            from_input = training_info[0].get("run_id")
    except Exception:
        from_input = None
    run_id = resolve_run_id(
        explicit=run_id,
        from_input=from_input,
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

    desired_metrics: List[str] = []
    ranking_metrics = []
    if ranking_cfg and getattr(ranking_cfg, "metrics", None):
        try:
            ranking_metrics = [str(m).strip().lower() for m in (ranking_cfg.metrics or []) if str(m).strip()]
        except Exception:
            ranking_metrics = []
    if ranking_metrics:
        desired_metrics = ranking_metrics
    elif clearml_cfg and clearml_cfg.comparison_metrics:
        desired_metrics = [str(m).strip().lower() for m in clearml_cfg.comparison_metrics if str(m).strip()]
    else:
        desired_metrics = ["r2", "rmse", "mae"]

    primary_metric = None
    if ranking_cfg and getattr(ranking_cfg, "primary_metric", None):
        try:
            primary_metric = str(ranking_cfg.primary_metric).strip().lower()
        except Exception:
            primary_metric = None
    if not primary_metric:
        if clearml_cfg and clearml_cfg.comparison_metrics:
            try:
                primary_metric = str(clearml_cfg.comparison_metrics[0]).strip().lower()
            except Exception:
                primary_metric = None
    if not primary_metric and desired_metrics:
        primary_metric = desired_metrics[0]

    goal = None
    if ranking_cfg and getattr(ranking_cfg, "goal", None):
        try:
            goal = str(ranking_cfg.goal).strip().lower()
        except Exception:
            goal = None

    composite_cfg = getattr(ranking_cfg, "composite", None) if ranking_cfg else None
    composite_metrics = None
    composite_weights = None
    composite_enabled = True
    composite_require_all = False
    if composite_cfg:
        try:
            composite_enabled = bool(getattr(composite_cfg, "enabled", True))
        except Exception:
            composite_enabled = True
        try:
            composite_require_all = bool(getattr(composite_cfg, "require_all_metrics", False))
        except Exception:
            composite_require_all = False
        try:
            composite_metrics = list(getattr(composite_cfg, "metrics", []) or [])
        except Exception:
            composite_metrics = None
        try:
            composite_weights = dict(getattr(composite_cfg, "weights", {}) or {})
        except Exception:
            composite_weights = None

    # Ensure metrics needed by composite/primary are collected from tasks.
    extras: List[str] = []
    if composite_metrics:
        extras.extend([str(m).strip().lower() for m in composite_metrics if str(m).strip()])
    elif composite_weights:
        extras.extend([str(k).strip().lower() for k in composite_weights.keys() if str(k).strip()])
    if primary_metric and primary_metric not in {"", "none", "null", "composite_score"}:
        extras.append(primary_metric)
    for m in extras:
        if m and m not in desired_metrics:
            desired_metrics.append(m)

    # Avoid task reuse for comparison (except inside pipeline step task)
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
        current = os.environ.get("CLEARML_TASK_ID")
        if not (current and str(current).strip()):
            os.environ.pop("CLEARML_TASK_ID", None)
            os.environ["CLEARML_TASK_ID"] = ""

    def _normalize_training_infos(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
        if isinstance(value, dict):
            return [value]
        return []

    def _normalize_parent_ids(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v]
        return [str(value)]

    allow_task_name_lookup = bool(clearml_cfg and clearml_cfg.enabled)

    def _maybe_task_name(task_id: str) -> Optional[str]:
        if not allow_task_name_lookup:
            return None
        try:
            from clearml import Task  # type: ignore

            t = Task.get_task(task_id=str(task_id))
            name = getattr(t, "name", None)
            return str(name) if name else None
        except Exception:
            return None

    training_infos = _normalize_training_infos(training_info)
    parent_ids = _normalize_parent_ids(parent_task_id)

    run_specs: List[Tuple[Optional[Dict[str, Any]], Optional[str]]] = []
    if training_infos:
        run_specs = [(ti, None) for ti in training_infos]
        if len(run_specs) == 1 and parent_ids:
            run_specs[0] = (run_specs[0][0], parent_ids[0])
    elif parent_ids:
        run_specs = [(None, pid) for pid in parent_ids]
    else:
        run_specs = [(None, None)]

    rows: List[Dict[str, Any]] = []
    for idx, (run_training_info, run_parent_id_override) in enumerate(run_specs, start=1):
        run_parent_id = run_parent_id_override or (run_training_info or {}).get("task_id")
        run_label = (run_training_info or {}).get("run_label")
        if run_label:
            run_label = str(run_label)
        elif run_parent_id:
            run_label = _maybe_task_name(str(run_parent_id)) or str(run_parent_id)
        else:
            run_label = f"run_{idx}"

        task_ids: List[str] = []
        if run_training_info and run_training_info.get("training_task_ids"):
            task_ids = list(run_training_info.get("training_task_ids") or [])
        if not task_ids:
            if run_parent_id and clearml_cfg and clearml_cfg.project_name:
                project_parent = clearml_cfg.project_name or "AutoML"
                train_models_project = f"{project_parent}/train_models"
                task_ids = _find_child_training_tasks(str(run_parent_id), train_models_project)

        run_rows: List[Dict[str, Any]] = []
        if run_training_info and run_training_info.get("metrics"):
            raw_rows = list(run_training_info.get("metrics") or [])
            run_rows = [r for r in raw_rows if isinstance(r, dict)]
        if (not run_rows) and task_ids:
            run_rows = _collect_metrics_from_tasks(task_ids, desired_metrics)

        for row in run_rows:
            enriched = dict(row)
            if len(run_specs) > 1:
                enriched["run_id"] = run_label
                if run_parent_id:
                    enriched["run_task_id"] = str(run_parent_id)
            rows.append(enriched)

    output_dir = Path(cfg.output.output_dir)
    meta = build_comparison_metadata(
        rows,
        output_dir=output_dir,
        metric_cols=desired_metrics,
        primary_metric=primary_metric,
        goal=goal,
        group_col=("run_id" if len(run_specs) > 1 else None),
        top_k=(getattr(ranking_cfg, "top_k", None) if ranking_cfg else None),
        composite_enabled=composite_enabled,
        composite_metrics=composite_metrics,
        composite_weights=composite_weights,
        composite_require_all=composite_require_all,
    )

    task = None
    logger = None
    if clearml_cfg and clearml_cfg.enabled:
        parent_id = None
        if len(run_specs) == 1:
            ti0, pid0 = run_specs[0]
            parent_id = pid0 or (ti0 or {}).get("task_id")
        config_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()  # type: ignore[attr-defined]
        try:
            dataset_id_for_key = None
            if len(training_infos) == 1:
                dataset_id_for_key = (training_infos[0] or {}).get("dataset_id")
            dataset_key = resolve_dataset_key(
                explicit=getattr(cfg.run, "dataset_key", None),
                dataset_id=str(dataset_id_for_key) if dataset_id_for_key else None,
                csv_path=None,
            )
            if len(training_infos) > 1:
                dataset_key = "multi"
        except Exception:
            dataset_key = "unknown"
        try:
            config_dict.setdefault("run", {})
            config_dict["run"]["id"] = run_id
            config_dict["run"]["dataset_key"] = dataset_key
        except Exception:
            pass
        try:
            if dataset_id_for_key:
                config_dict.setdefault("data", {})
                config_dict["data"]["dataset_id"] = str(dataset_id_for_key)
        except Exception:
            pass
        task = create_comparison_task(config_dict, parent_task_id=str(parent_id) if parent_id else None)
        try:
            logger = task.get_logger() if task else None
        except Exception:
            logger = None

    if logger and isinstance(meta.get("ranked_df"), pd.DataFrame):
        ranked_for_vis = meta["ranked_df"]
        # Basic stats scalars (useful for comparing comparison runs)
        try:
            report_scalar(logger, title="comparison/n_candidates", series="value", value=float(len(ranked_for_vis)))
        except Exception:
            pass
        try:
            if "model" in ranked_for_vis.columns:
                report_scalar(
                    logger,
                    title="comparison/n_models",
                    series="value",
                    value=float(ranked_for_vis["model"].nunique(dropna=True)),
                )
        except Exception:
            pass
        try:
            if "preprocessor" in ranked_for_vis.columns:
                report_scalar(
                    logger,
                    title="comparison/n_preprocessors",
                    series="value",
                    value=float(ranked_for_vis["preprocessor"].nunique(dropna=True)),
                )
        except Exception:
            pass
        top_k = None
        if ranking_cfg and getattr(ranking_cfg, "top_k", None):
            try:
                top_k = int(ranking_cfg.top_k)
            except Exception:
                top_k = None
        if top_k and hasattr(ranked_for_vis, "head"):
            try:
                ranked_for_vis = ranked_for_vis.head(top_k)
            except Exception:
                pass

        render_comparison_visuals(logger, ranked_for_vis, metric_cols=desired_metrics)

        best = meta.get("best") or {}
        if isinstance(best, dict):
            try:
                best_row = best.get("best_row") or {}
                metric = best.get("primary_metric")
                if metric and isinstance(best_row, dict) and metric in best_row:
                    report_scalar(logger, title=f"best_{metric}", series="best", value=float(best_row[metric]))
            except Exception:
                pass
            # Standardized scalars for comparing comparison-runs.
            try:
                best_row = best.get("best_row") or {}
                metric = str(best.get("primary_metric") or "").strip().lower()
                if metric and isinstance(best_row, dict) and metric in best_row and best_row[metric] is not None:
                    report_scalar(
                        logger,
                        title=f"comparison/best_{metric}",
                        series="value",
                        value=float(best_row[metric]),
                    )
            except Exception:
                pass
            try:
                ranked_topk_df = meta.get("ranked_topk_df")
                metric = str(best.get("primary_metric") or "").strip().lower()
                if (
                    isinstance(ranked_topk_df, pd.DataFrame)
                    and not ranked_topk_df.empty
                    and metric
                    and metric in ranked_topk_df.columns
                ):
                    vals = pd.to_numeric(ranked_topk_df[metric], errors="coerce")
                    mean_val = float(vals.mean())
                    if mean_val == mean_val:  # not NaN
                        report_scalar(
                            logger,
                            title=f"comparison/topk_mean_{metric}",
                            series="value",
                            value=mean_val,
                        )
            except Exception:
                pass

        if isinstance(meta.get("best_by_group_df"), pd.DataFrame):
            try:
                report_table(logger, title="best_by_run", df=meta["best_by_group_df"], series="best_by_run")
            except Exception:
                pass
        if isinstance(meta.get("best_by_model_df"), pd.DataFrame):
            try:
                report_table(logger, title="best_by_model", df=meta["best_by_model_df"], series="best_by_model")
            except Exception:
                pass
        if isinstance(meta.get("win_summary_df"), pd.DataFrame):
            try:
                report_table(logger, title="win_summary", df=meta["win_summary_df"], series="summary")
            except Exception:
                pass
            try:
                render_win_summary_visuals(logger, meta["win_summary_df"])
            except Exception:
                pass
        if isinstance(meta.get("model_summary_df"), pd.DataFrame):
            metric = None
            if isinstance(best, dict):
                metric = best.get("primary_metric")
            metric = str(metric or primary_metric or (desired_metrics[0] if desired_metrics else "rmse")).strip().lower()
            render_model_summary_visuals(logger, meta["model_summary_df"], primary_metric=metric)
        if isinstance(meta.get("recommended_model"), dict):
            try:
                rec = meta.get("recommended_model") or {}
                stats = rec.get("stats") if isinstance(rec, dict) else None
                row = dict(stats) if isinstance(stats, dict) else {}
                if isinstance(rec, dict):
                    row.setdefault("selected_model", rec.get("selected_model"))
                    row.setdefault("strategy", rec.get("strategy"))
                    row.setdefault("primary_metric", rec.get("primary_metric"))
                    row.setdefault("goal", rec.get("goal"))
                if row:
                    report_table(logger, title="recommended_model", df=pd.DataFrame([row]), series="summary")
            except Exception:
                pass

    try:
        finalize_comparison_task(task, artifact_paths=meta.get("artifacts", []))
    except Exception:
        pass

    info = ComparisonInfo(
        task_id=(task.id if task else None),
        artifacts=list(meta.get("artifacts", []) or []),
        run_id=run_id,
    )

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

    return info.model_dump()


def _collect_metrics_from_tasks(task_ids: List[str], metric_names: List[str]) -> List[Dict[str, Any]]:
    try:
        from clearml import Task  # type: ignore
    except Exception:
        return []

    def _last_scalar_value(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return float(value)
            except Exception:
                return None
        if isinstance(value, dict):
            for key in ["y", "value", "last"]:
                if key in value:
                    return _last_scalar_value(value.get(key))
        if isinstance(value, list) and value:
            return _last_scalar_value(value[-1])
        if isinstance(value, tuple) and value:
            return _last_scalar_value(value[-1])
        try:
            return float(value)
        except Exception:
            return None

    def _get_reported_scalar(task_obj, title: str, series: Optional[str] = None) -> Optional[float]:
        try:
            scalars = task_obj.get_reported_scalars()
        except Exception:
            scalars = None
        if not isinstance(scalars, dict):
            return None
        title_key = title
        if title_key not in scalars:
            # best-effort fallback (case variations)
            for cand in [title.lower(), title.upper()]:
                if cand in scalars:
                    title_key = cand
                    break
        data = scalars.get(title_key)
        if not isinstance(data, dict) or not data:
            return None
        if series:
            series_key = series
            if series_key not in data:
                for cand in [series.lower(), series.upper()]:
                    if cand in data:
                        series_key = cand
                        break
            return _last_scalar_value(data.get(series_key))
        # no series specified: take first
        first = next(iter(data.values()))
        return _last_scalar_value(first)

    rows: List[Dict[str, Any]] = []
    for tid in task_ids:
        try:
            t = Task.get_task(task_id=tid)
        except Exception:
            continue
        try:
            sv = t.get_reported_single_values()
        except Exception:
            sv = {}
        name = t.name if t else tid
        model = name
        preproc = ""
        if " - " in name:
            model = name.split(" - ")[0]
            preproc = name.split(" - ")[1]
        else:
            # New naming: "train [<preproc>] [<model>] [<run_id>]" or "train [<model>] [<run_id>]"
            try:
                lowered = str(name).strip().lower()
                if lowered.startswith("train "):
                    parts = re.findall(r"\\[([^\\]]+)\\]", str(name))
                    if len(parts) == 2:
                        model = parts[0]
                    elif len(parts) >= 3:
                        preproc = parts[0]
                        model = parts[1]
            except Exception:
                pass
        row = {"task_id": tid, "task_name": name, "model": model, "preprocessor": preproc}
        for m in metric_names:
            key = str(m).strip().lower()
            val = sv.get(key) or sv.get(key.upper()) or sv.get(key.lower())
            if val is None:
                # New convention: title="metric/<metric>", series="value"
                val = _get_reported_scalar(t, f"metric/{key}", "value")
            if val is None:
                # Legacy convention: title="train", series="R2" etc.
                val = _get_reported_scalar(t, "train", key.upper()) or _get_reported_scalar(t, "train", key)
            row[m] = val
        rows.append(row)
    return rows


def _find_child_training_tasks(parent_task_id: str, project_name: Optional[str]) -> List[str]:
    try:
        from clearml import Task  # type: ignore
    except Exception:
        return []
    try:
        tasks = Task.get_tasks(project_name=project_name, task_filter={"parent": parent_task_id})
    except Exception:
        return []
    ids: List[str] = []
    if tasks:
        for t in tasks:
            try:
                ids.append(t.id)
            except Exception:
                continue
    return ids
