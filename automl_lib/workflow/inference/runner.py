from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from automl_lib.integrations.clearml import ensure_local_dataset_copy, find_first_csv
from automl_lib.integrations.clearml.context import (
    build_run_context,
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    sanitize_name_token,
    set_run_id_env,
)
from automl_lib.integrations.clearml.manager import ClearMLManager, _import_clearml, build_clearml_config_from_dict
from automl_lib.integrations.clearml.naming import build_project_path, build_tags, task_name
from automl_lib.integrations.clearml.properties import set_user_properties
from automl_lib.workflow.inference.provenance import collect_model_provenance, resolve_dataset_id_for_range
from automl_lib.workflow.inference.visualization import (
    build_contour_plot,
    build_feature_importance_plot,
    build_loss_history_plot,
    build_parallel_coordinates_plot,
    build_single_conditions_table,
    build_topk_table,
    build_training_position_plot,
)


def _child_task_type(mode: str, task_types_cls: Any) -> Any:
    task_type = getattr(task_types_cls, "inference", None)
    if str(mode).strip().lower() == "optimize":
        task_type = getattr(task_types_cls, "optimizer", None) or getattr(task_types_cls, "optimization", None) or task_type
    return task_type


def _create_child_task(
    *,
    project_name: str,
    name: str,
    mode: str,
    parent_task_id: Optional[str],
    tags: list[str],
    requested_queue: Optional[str],
) -> Any:
    _, _, Task, TaskTypes = _import_clearml()
    if Task is None or TaskTypes is None:
        return None
    try:
        task_obj = Task.create(
            project_name=project_name,
            task_name=name,
            task_type=_child_task_type(mode, TaskTypes),
            # This child task is used for logging (not remote execution). Keep creation lightweight/stable.
            add_task_init_call=False,
            detect_repository=False,
        )
        if parent_task_id:
            try:
                if hasattr(task_obj, "add_parent"):
                    task_obj.add_parent(parent_task_id)
                else:
                    task_obj.set_parent(parent_task_id)
            except Exception:
                pass
        try:
            task_obj.add_tags(tags)
        except Exception:
            pass
        if requested_queue:
            try:
                task_obj.set_parameter("requested_queue", str(requested_queue))
            except Exception:
                pass
        return task_obj
    except Exception:
        return None


def _variables_section_from_config(input_conf: Dict[str, Any]) -> Dict[str, Any]:
    """Convert input.variables (list[dict]) into a dict keyed by variable name for ClearML HyperParameters UI."""

    raw = input_conf.get("variables")
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        spec = dict(item)
        spec.pop("name", None)
        out[name] = spec
    return out


def _write_json(path: Path, payload: Any) -> None:
    def _json_default(obj: Any) -> Any:
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        try:
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
        except Exception:
            pass
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _resolved_single_row(*, input_conf: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """Return the resolved single input row (inline dict preferred; fall back to output input.json)."""

    if isinstance(input_conf.get("single"), dict):
        return dict(input_conf["single"])
    try:
        p = Path(output_dir) / "input.json"
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("row"), dict):
                return dict(raw.get("row") or {})
    except Exception:
        pass
    return {}


def _load_training_df_for_range(*, dataset_id: str, output_dir: Path) -> Optional[pd.DataFrame]:
    """Load a best-effort *raw* training dataframe for range visualization."""

    if not dataset_id:
        return None
    local_copy = ensure_local_dataset_copy(str(dataset_id), output_dir / "clearml_training_dataset")
    if not local_copy:
        return None

    processed_names = {"data_processed.csv", "preprocessed_features.csv"}

    def _pick_csv(root: Path) -> Optional[Path]:
        try:
            candidates = [p for p in Path(root).rglob("*.csv") if p.name not in processed_names]
            if candidates:
                candidates.sort(key=lambda p: (len(p.parts), str(p)))
                return candidates[0]
        except Exception:
            pass
        return find_first_csv(Path(root))

    csv_path = _pick_csv(Path(local_copy))
    # If the dataset only contains processed tables, try parent_dataset_id from manifest.json.
    try:
        if csv_path and csv_path.name in processed_names:
            manifest_path = Path(local_copy) / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                parent_id = manifest.get("parent_dataset_id") if isinstance(manifest, dict) else None
                if parent_id:
                    parent_copy = ensure_local_dataset_copy(str(parent_id), output_dir / "clearml_training_dataset_parent")
                    if parent_copy:
                        csv_parent = _pick_csv(Path(parent_copy))
                        if csv_parent:
                            csv_path = csv_parent
    except Exception:
        pass

    if not csv_path or not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def _dataset_lineage_from_manifest(*, dataset_id: str, output_dir: Path) -> Dict[str, Any]:
    """Extract dataset lineage (parent_dataset_id / preprocessing_task_id) from manifest.json if present."""

    out: Dict[str, Any] = {}
    if not dataset_id:
        return out
    local_copy = ensure_local_dataset_copy(str(dataset_id), output_dir / "clearml_dataset_lineage")
    if not local_copy:
        return out
    manifest_path = Path(local_copy) / "manifest.json"
    if not manifest_path.exists():
        return out
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    if not isinstance(manifest, dict):
        return out
    for k in ["parent_dataset_id", "preprocessing_task_id", "contract_version", "selected_preprocessor", "created_at"]:
        v = manifest.get(k)
        if v is None:
            continue
        out[str(k)] = v
    out["dataset_id"] = str(dataset_id)
    return out


def run_inference_workflow(
    config_data: Dict[str, Any],
    *,
    input_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run inference and (optionally) log to ClearML.

    Notes:
    - The actual inference execution is ClearML-independent and lives in `automl_lib.inference.run`.
    - ClearML Task creation/logging stays in the workflow layer so phase-level UI changes are easy to track.
    """

    run_cfg = config_data.get("run") if isinstance(config_data.get("run"), dict) else {}
    run_id = resolve_run_id(
        from_config=(run_cfg or {}).get("id"),
        from_input=(input_info or {}).get("run_id"),
        from_env=get_run_id_env(),
    )
    set_run_id_env(run_id)

    input_conf = config_data.get("input") if isinstance(config_data.get("input"), dict) else {}
    mode = str(input_conf.get("mode") or "single").strip().lower()
    if mode == "csv":
        mode = "batch"
    if mode == "params":
        mode = "optimize"
    search_conf = config_data.get("search") if isinstance(config_data.get("search"), dict) else {}

    clearml_cfg_raw = config_data.get("clearml") if isinstance(config_data.get("clearml"), dict) else {}
    clearml_cfg = build_clearml_config_from_dict(clearml_cfg_raw)

    # Prefer pipeline/training-provided dataset_id for naming/traceability when inference input has no dataset reference.
    dataset_id_for_key = (
        str(input_conf.get("dataset_id") or "").strip()
        or str((input_info or {}).get("dataset_id") or "").strip()
        or None
    )
    dataset_key = resolve_dataset_key(
        explicit=(run_cfg or {}).get("dataset_key"),
        dataset_id=(dataset_id_for_key if dataset_id_for_key else None),
        csv_path=(str(input_conf.get("csv_path")) if input_conf.get("csv_path") else None)
        or (str(input_conf.get("single_json")) if input_conf.get("single_json") else None),
    )
    ctx = build_run_context(
        run_id=run_id,
        dataset_key=dataset_key,
        project_root=(clearml_cfg.project_name if clearml_cfg else None),
        dataset_project=(clearml_cfg.dataset_project if clearml_cfg else None),
        user=(run_cfg or {}).get("user"),
    )
    project_mode = getattr(getattr(clearml_cfg, "naming", None), "project_mode", "root") if clearml_cfg else "root"
    naming_cfg = getattr(clearml_cfg, "naming", None) if clearml_cfg else None
    project_root = build_project_path(ctx, project_mode=project_mode)
    prediction_suffix = getattr(naming_cfg, "prediction_runs_suffix", "Prediction_runs")
    prediction_runs_project = build_project_path(ctx, project_mode=project_mode, suffix=prediction_suffix)

    # ----------------------------
    # Execute core inference (no ClearML side-effects)
    # ----------------------------
    from automl_lib.inference.run import run_inference_core

    core = run_inference_core(config_data) or {}
    output_dir = core.get("output_dir")
    if not isinstance(output_dir, Path):
        raise RuntimeError(f"Inference core did not return a valid output_dir: {output_dir}")
    artifacts = core.get("artifacts") if isinstance(core.get("artifacts"), list) else []
    artifact_paths = [p for p in artifacts if isinstance(p, Path) and p.exists()]
    user_props = dict(core.get("user_props") or {}) if isinstance(core.get("user_props"), dict) else {}
    model_meta = core.get("model_meta") if isinstance(core.get("model_meta"), dict) else {}
    model_id = str(core.get("model_id") or "").strip()
    model_name = str(core.get("model_name") or "").strip() or "model"
    # Ensure UI/search stability: workflow-level ctx is the source of truth.
    user_props["run_id"] = ctx.run_id
    user_props["dataset_key"] = ctx.dataset_key

    # No ClearML: return the phase output only.
    if not (clearml_cfg and bool(getattr(clearml_cfg, "enabled", False))):
        return {
            "task_id": None,
            "child_task_ids": [],
            "output_dir": str(output_dir),
            "artifacts": [str(p) for p in artifact_paths],
            "mode": str(mode),
        }

    parent_for_inference = str((input_info or {}).get("task_id") or "").strip() or None
    provenance = collect_model_provenance(model_id)
    dataset_id_for_range_plot = resolve_dataset_id_for_range(
        input_info=input_info,
        input_conf=input_conf,
        provenance=provenance,
    )
    if dataset_id_for_range_plot:
        user_props.setdefault("dataset_id", str(dataset_id_for_range_plot))
    try:
        if dataset_id_for_range_plot:
            provenance["dataset_lineage"] = _dataset_lineage_from_manifest(
                dataset_id=str(dataset_id_for_range_plot),
                output_dir=Path(output_dir),
            )
    except Exception:
        pass
    df_train = None
    try:
        if dataset_id_for_range_plot:
            df_train = _load_training_df_for_range(dataset_id=str(dataset_id_for_range_plot), output_dir=Path(output_dir))
    except Exception:
        df_train = None

    # ----------------------------------------
    # Mode: optimize -> inference-summary + Prediction_runs children (TopK)
    # ----------------------------------------
    if mode == "optimize":
        summary_task_obj = None
        if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
            _, _, TaskCls, _ = _import_clearml()
            if TaskCls is not None:
                try:
                    summary_task_obj = TaskCls.current_task()
                except Exception:
                    summary_task_obj = None
                if summary_task_obj is not None:
                    try:
                        if hasattr(summary_task_obj, "move_to_project"):
                            summary_task_obj.move_to_project(new_project_name=project_root)
                    except Exception:
                        pass

        summary_task_name = task_name("inference_summary", ctx)
        summary_mgr = ClearMLManager(
            clearml_cfg,
            task_name=summary_task_name,
            task_type="inference",
            default_project=project_root,
            project=project_root,
            parent=parent_for_inference,
            existing_task=summary_task_obj,
            extra_tags=build_tags(ctx, phase="inference", model=model_name, extra=["optimize"]),
        )
        summary_task_id = str(summary_mgr.task.id) if getattr(summary_mgr, "task", None) is not None else None

        # Load full trials for plots/TopK (core already wrote trials.csv).
        df_trials = None
        try:
            trials_path = Path(output_dir) / "trials.csv"
            if trials_path.exists():
                df_trials = pd.read_csv(trials_path)
        except Exception:
            df_trials = None

        goal = str(search_conf.get("goal") or "max")
        top_k = int(search_conf.get("top_k") or 10)
        if top_k < 1:
            top_k = 1
        # Rank ALL trials by prediction (used for child task creation).
        ranked_df = (
            build_topk_table(df_trials=df_trials, goal=goal, top_k=int(max(1, len(df_trials))))
            if isinstance(df_trials, pd.DataFrame)
            else pd.DataFrame()
        )
        topk_df = ranked_df.head(top_k) if (isinstance(ranked_df, pd.DataFrame) and not ranked_df.empty) else pd.DataFrame()

        try:
            summary_mgr.connect_params_sections(
                {
                    "Model": {"model_id": model_id, "model_name": model_name},
                    "Input": {
                        "mode": mode,
                        "dataset_id": str(dataset_id_for_range_plot or ""),
                        "csv_path": str(input_conf.get("csv_path") or ""),
                        "params_path": str(input_conf.get("params_path") or ""),
                    },
                    "Variables": _variables_section_from_config(input_conf),
                    "Search": {
                        "method": str(search_conf.get("method") or ""),
                        "n_trials": int(search_conf.get("n_trials") or 0),
                        "goal": str(search_conf.get("goal") or ""),
                        "top_k": int(search_conf.get("top_k") or 10),
                    },
                    "Output": {"output_dir": str(output_dir)},
                }
            )
        except Exception:
            pass
        try:
            summary_mgr.connect_configuration(model_meta, name="Model")
            summary_mgr.connect_configuration(provenance, name="Provenance")
            summary_mgr.connect_configuration(config_data, name="OmegaConf")
        except Exception:
            pass

        # Plots: numbered for UI ordering.
        try:
            if summary_mgr.logger is not None:
                if isinstance(topk_df, pd.DataFrame) and not topk_df.empty:
                    summary_mgr.report_table("01_Overview/TopK", topk_df, series="topk")
                if isinstance(df_trials, pd.DataFrame) and not df_trials.empty:
                    summary_mgr.report_table("02_Overview/TrialsPreview", df_trials.head(100), series="trials")
                # Training position plot (TopK points)
                try:
                    if df_train is not None and isinstance(topk_df, pd.DataFrame) and not topk_df.empty:
                        fig_pos = build_training_position_plot(
                            df_train=df_train,
                            df_points=topk_df.drop(columns=["rank"], errors="ignore"),
                            title="03_Data/Input Position vs Training (TopK)",
                            point_name="topk",
                        )
                        if fig_pos is not None:
                            summary_mgr.logger.report_plotly(title="03_Data/InputPosition", series="pca", iteration=0, figure=fig_pos)
                except Exception:
                    pass
                try:
                    if isinstance(df_trials, pd.DataFrame) and not df_trials.empty:
                        fig_loss = build_loss_history_plot(
                            df_trials=df_trials,
                            goal=str(search_conf.get("goal") or "max"),
                            title="04_Optimization/LossHistory",
                        )
                        if fig_loss is not None:
                            summary_mgr.logger.report_plotly(title="04_Optimization/LossHistory", series="loss", iteration=0, figure=fig_loss)
                except Exception:
                    pass
                try:
                    if isinstance(df_trials, pd.DataFrame) and not df_trials.empty:
                        fig_par, meta = build_parallel_coordinates_plot(df_trials=df_trials, title="05_Optimization/ParallelCoordinates")
                        if fig_par is not None:
                            summary_mgr.logger.report_plotly(title="05_Optimization/ParallelCoordinates", series="pc", iteration=0, figure=fig_par)
                        if meta:
                            summary_mgr.connect_configuration(meta, name="ParallelCoordinatesMeta")
                except Exception:
                    pass
                try:
                    if isinstance(df_trials, pd.DataFrame) and not df_trials.empty:
                        fig_imp = build_feature_importance_plot(df_trials=df_trials, title="06_Optimization/FeatureImportance")
                        if fig_imp is not None:
                            summary_mgr.logger.report_plotly(title="06_Optimization/FeatureImportance", series="importance", iteration=0, figure=fig_imp)
                except Exception:
                    pass
                try:
                    if isinstance(df_trials, pd.DataFrame) and not df_trials.empty:
                        fig_cont = build_contour_plot(df_trials=df_trials, title="07_Optimization/Contour")
                        if fig_cont is not None:
                            summary_mgr.logger.report_plotly(title="07_Optimization/Contour", series="contour", iteration=0, figure=fig_cont)
                except Exception:
                    pass
        except Exception:
            pass

        # Summary artifacts + properties
        try:
            summary_mgr.upload_artifacts(artifact_paths)
        except Exception:
            pass
        try:
            if getattr(summary_mgr, "task", None) is not None:
                set_user_properties(summary_mgr.task, user_props)
        except Exception:
            pass

        # Create Prediction_runs child tasks for ALL trials (content matches single inference).
        child_task_ids: list[str] = []
        topk_items: list[Dict[str, Any]] = []
        if isinstance(ranked_df, pd.DataFrame) and not ranked_df.empty:
            safe_model = sanitize_name_token(model_name, max_len=48)
            pred_root = Path(output_dir) / "Prediction_runs"
            pred_root.mkdir(parents=True, exist_ok=True)
            rank_width = max(2, len(str(len(ranked_df))))
            for _, row in ranked_df.iterrows():
                try:
                    rank = int(row.get("rank")) if "rank" in row else 0
                except Exception:
                    rank = 0
                try:
                    trial_index = int(row.get("trial_index")) if "trial_index" in row else None
                except Exception:
                    trial_index = None
                try:
                    pred_val = float(row.get("prediction")) if "prediction" in row else None
                except Exception:
                    pred_val = None
                # Build a "single input" dict from the row (drop non-input columns).
                drop_cols = {"rank", "prediction", "trial_index"}
                single_row = {k: row.get(k) for k in row.index if str(k) not in drop_cols}

                # Artifacts for this prediction run.
                rank_tok = str(rank).zfill(rank_width) if rank else "unknown"
                run_dir = pred_root / f"rank_{rank_tok}" if rank else pred_root / "rank_unknown"
                in_path = run_dir / "input.json"
                out_path = run_dir / "output.json"
                _write_json(
                    in_path,
                    {
                        "source": "optimize:trial",
                        "rank": rank,
                        "trial_index": trial_index,
                        "row": single_row,
                    },
                )
                _write_json(out_path, {"prediction": pred_val, "rank": rank, "trial_index": trial_index})
                child_artifacts = [p for p in [in_path, out_path] if p.exists()]

                child_name = f"predict rank:{rank_tok} model:{safe_model}" if rank else f"predict model:{safe_model}"
                extra_tags = ["prediction_run", "child", f"rank:{rank_tok}"]
                if trial_index is not None:
                    extra_tags.append(f"trial:{trial_index}")
                if rank and rank <= top_k:
                    extra_tags.append("topk")
                child_tags = build_tags(ctx, phase="inference", model=model_name, extra=extra_tags)
                task_obj = _create_child_task(
                    project_name=prediction_runs_project,
                    name=child_name,
                    mode="single",
                    parent_task_id=summary_task_id,
                    tags=child_tags,
                    requested_queue=(clearml_cfg.queue if (clearml_cfg.queue and not clearml_cfg.run_tasks_locally) else None),
                )
                child_clearml_cfg = clearml_cfg if task_obj is not None else None
                child_mgr = ClearMLManager(
                    child_clearml_cfg,
                    task_name=child_name,
                    task_type="inference",
                    default_project=prediction_runs_project,
                    project=prediction_runs_project,
                    parent=summary_task_id,
                    existing_task=task_obj,
                    extra_tags=child_tags,
                )
                child_id = str(child_mgr.task.id) if getattr(child_mgr, "task", None) is not None else ""
                if child_id:
                    child_task_ids.append(child_id)
                    if rank and rank <= top_k:
                        topk_items.append(
                            {
                                "rank": rank,
                                "trial_index": trial_index,
                                "prediction": pred_val,
                                "task_id": child_id,
                            }
                        )
                try:
                    child_mgr.connect_params_sections(
                        {
                            "Model": {"model_id": model_id, "model_name": model_name},
                            "Input": {"mode": "single", "dataset_id": str(dataset_id_for_range_plot or "")},
                            "SingleInput": dict(single_row),
                            "Search": {"rank": rank, "trial_index": trial_index, "prediction": pred_val},
                        }
                    )
                except Exception:
                    pass
                try:
                    child_mgr.connect_configuration(model_meta, name="Model")
                    child_mgr.connect_configuration(provenance, name="Provenance")
                except Exception:
                    pass
                try:
                    if child_mgr.logger is not None:
                        df_one = build_single_conditions_table(row=dict(single_row), prediction=pred_val)
                        child_mgr.report_table("01_Overview/ConditionsAndPrediction", df_one, series="single")
                        if df_train is not None:
                            fig_pos = build_training_position_plot(
                                df_train=df_train,
                                df_points=df_one.drop(columns=["prediction"], errors="ignore"),
                                title="02_Data/Input Position vs Training",
                                point_name="input",
                            )
                            if fig_pos is not None:
                                child_mgr.logger.report_plotly(title="02_Data/InputPosition", series="pca", iteration=0, figure=fig_pos)
                except Exception:
                    pass
                try:
                    child_mgr.upload_artifacts(child_artifacts)
                except Exception:
                    pass
                try:
                    if getattr(child_mgr, "task", None) is not None:
                        props = dict(user_props)
                        props.update({"rank": rank, "trial_index": trial_index, "prediction": pred_val})
                        set_user_properties(child_mgr.task, props)
                except Exception:
                    pass
                child_mgr.close()

        try:
            summary_mgr.connect_configuration(
                {
                    "mode": "all_trials",
                    "total": int(len(ranked_df)) if isinstance(ranked_df, pd.DataFrame) else 0,
                    "created": int(len(child_task_ids)),
                    "top_k": int(top_k),
                    "top_k_items": json.loads(json.dumps(topk_items, default=str)),
                    "project": str(prediction_runs_project),
                },
                name="PredictionRuns",
            )
        except Exception:
            pass
        summary_mgr.close()

        return {
            "task_id": summary_task_id,
            "child_task_ids": child_task_ids,
            "output_dir": str(output_dir),
            "artifacts": [str(p) for p in artifact_paths],
            "mode": str(mode),
        }

    # ----------------------------------------
    # Mode: single/batch -> one task only (no inference-summary)
    # ----------------------------------------
    safe_model = sanitize_name_token(model_name, max_len=48)
    task_title = f"infer {mode} model:{safe_model}"
    extra = [str(mode)]
    if mode == "single":
        extra.append("prediction_run")
    task_tags = build_tags(ctx, phase="inference", model=model_name, extra=extra)

    mgr = ClearMLManager(
        clearml_cfg,
        task_name=task_title,
        task_type="inference",
        default_project=project_root,
        project=project_root,
        parent=parent_for_inference,
        extra_tags=task_tags,
    )
    task_id = str(mgr.task.id) if getattr(mgr, "task", None) is not None else None

    # HyperParameters: per-variable for single, and preview info for batch.
    try:
        sections: Dict[str, Any] = {
            "Model": {"model_id": model_id, "model_name": model_name},
            "Input": {
                "mode": mode,
                "dataset_id": str(dataset_id_for_range_plot or ""),
                "csv_path": str(input_conf.get("csv_path") or ""),
                "single_json": str(input_conf.get("single_json") or ""),
            },
            "Output": {"output_dir": str(output_dir)},
        }
        if mode == "single":
            sections["SingleInput"] = _resolved_single_row(input_conf=input_conf, output_dir=Path(output_dir))
        mgr.connect_params_sections(sections)
    except Exception:
        pass
    try:
        mgr.connect_configuration(model_meta, name="Model")
        mgr.connect_configuration(provenance, name="Provenance")
        mgr.connect_configuration(config_data, name="OmegaConf")
    except Exception:
        pass

    # Plots
    try:
        if mgr.logger is not None:
            if mode == "single":
                row = _resolved_single_row(input_conf=input_conf, output_dir=Path(output_dir))
                pred = core.get("prediction")
                df_one = build_single_conditions_table(row=dict(row or {}), prediction=(float(pred) if pred is not None else None))
                mgr.report_table("01_Overview/ConditionsAndPrediction", df_one, series="single")
                if df_train is not None:
                    fig_pos = build_training_position_plot(
                        df_train=df_train,
                        df_points=df_one.drop(columns=["prediction"], errors="ignore"),
                        title="02_Data/Input Position vs Training",
                        point_name="input",
                    )
                    if fig_pos is not None:
                        mgr.logger.report_plotly(title="02_Data/InputPosition", series="pca", iteration=0, figure=fig_pos)
            elif mode == "batch":
                preview = core.get("preview_df")
                if preview is not None:
                    mgr.report_table("01_Overview/PredictionsPreview", preview, series="predictions")
                    if df_train is not None:
                        try:
                            fig_pos = build_training_position_plot(
                                df_train=df_train,
                                df_points=preview.drop(columns=["prediction"], errors="ignore"),
                                title="02_Data/Input Position vs Training (Preview)",
                                point_name="preview",
                            )
                            if fig_pos is not None:
                                mgr.logger.report_plotly(title="02_Data/InputPosition", series="pca", iteration=0, figure=fig_pos)
                        except Exception:
                            pass
    except Exception:
        pass

    # Artifacts + properties
    try:
        mgr.upload_artifacts(artifact_paths)
    except Exception:
        pass
    try:
        if getattr(mgr, "task", None) is not None:
            set_user_properties(mgr.task, user_props)
    except Exception:
        pass

    mgr.close()
    return {
        "task_id": task_id,
        "child_task_ids": [],
        "output_dir": str(output_dir),
        "artifacts": [str(p) for p in artifact_paths],
        "mode": str(mode),
    }
