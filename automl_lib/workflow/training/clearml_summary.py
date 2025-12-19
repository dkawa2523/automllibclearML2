from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from automl_lib.config.schemas import TrainingConfig
from automl_lib.integrations.clearml.properties import set_user_properties


def normalize_summary_plots_mode(cfg: TrainingConfig) -> str:
    try:
        mode = str(getattr(cfg.clearml, "summary_plots", "best") if cfg.clearml else "none").strip().lower()
    except Exception:
        mode = "best"
    if mode not in {"none", "best", "all"}:
        mode = "best"
    # Policy: even if users request "all", mirror only the recommended model's detailed plots to the summary task.
    if mode == "all":
        mode = "best"
    return mode


def set_min_user_properties(
    clearml_mgr: Any,
    *,
    run_id: str,
    dataset_id: Optional[str],
    dataset_key: str,
) -> None:
    try:
        task = getattr(clearml_mgr, "task", None)
        if not task:
            return
        set_user_properties(
            task,
            {
                "run_id": str(run_id),
                "dataset_id": str(dataset_id or ""),
                "dataset_key": str(dataset_key),
            },
        )
    except Exception:
        pass


def _should_close_task(clearml_mgr: Any) -> bool:
    should_close = os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1"
    task = getattr(clearml_mgr, "task", None)
    if not should_close and task:
        step_task_id = (os.environ.get("CLEARML_TASK_ID") or "").strip()
        try:
            should_close = (not step_task_id) or (str(task.id) != step_task_id)
        except Exception:
            should_close = False
    return bool(should_close)


def finalize_task(clearml_mgr: Any) -> None:
    try:
        task = getattr(clearml_mgr, "task", None)
        if task:
            try:
                task.flush(wait_for_uploads=True)
            except Exception:
                pass
    except Exception:
        pass
    try:
        if _should_close_task(clearml_mgr):
            clearml_mgr.close()
    except Exception:
        pass


def fail_training_summary(
    clearml_mgr: Any,
    *,
    output_dir: Path,
    run_id: str,
    dataset_id: Optional[str],
    dataset_key: str,
    exc: BaseException,
) -> None:
    """Best-effort: mark training-summary task failed + persist traceback."""

    output_dir = Path(output_dir)
    err_path: Optional[Path] = None
    try:
        tb = traceback.format_exc()
        err_path = output_dir / "training_summary_error.txt"
        err_path.write_text(tb, encoding="utf-8", errors="ignore")
    except Exception:
        err_path = None

    task = getattr(clearml_mgr, "task", None)
    try:
        if task and hasattr(task, "mark_failed"):
            msg = str(exc)
            if len(msg) > 500:
                msg = msg[:500] + "..."
            task.mark_failed(
                ignore_errors=True,
                status_reason=str(exc),
                status_message=msg,
                force=True,
            )
    except Exception:
        pass

    try:
        if task:
            set_user_properties(
                task,
                {
                    "run_id": str(run_id),
                    "dataset_id": str(dataset_id or ""),
                    "dataset_key": str(dataset_key),
                    "status": "failed",
                    "error": str(exc),
                },
            )
    except Exception:
        pass

    try:
        if err_path is not None and err_path.exists():
            clearml_mgr.upload_artifacts([err_path])
    except Exception:
        pass

    finalize_task(clearml_mgr)


def persist_full_config(
    clearml_mgr: Any,
    *,
    cfg: TrainingConfig,
    run_id: str,
    dataset_key: str,
    dataset_id_for_load: Optional[str],
    output_dir: Path,
) -> Path:
    """Persist the resolved config as YAML and (if enabled) store it under Configuration Objects/OmegaConf."""

    output_dir = Path(output_dir)
    full_config_path = output_dir / "training_config.yaml"
    full_config_obj: Optional[Dict[str, Any]] = None
    try:
        import yaml  # type: ignore

        cfg_dump = cfg.model_dump()
        cfg_dump.setdefault("run", {})
        cfg_dump["run"]["id"] = run_id
        cfg_dump["run"]["dataset_key"] = dataset_key
        cfg_dump.setdefault("data", {})
        if dataset_id_for_load:
            cfg_dump["data"]["dataset_id"] = str(dataset_id_for_load)
        if getattr(cfg.data, "csv_path", None):
            cfg_dump["data"]["csv_path"] = str(cfg.data.csv_path)
        full_config_obj = json.loads(json.dumps(cfg_dump, default=str))
        full_config_path.write_text(yaml.safe_dump(cfg_dump, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        pass
    try:
        if full_config_obj is not None and getattr(clearml_mgr, "task", None):
            clearml_mgr.connect_configuration(full_config_obj, name="OmegaConf")
    except Exception:
        pass
    return full_config_path


def connect_training_hyperparams(
    clearml_mgr: Any,
    *,
    cfg: TrainingConfig,
    dataset_id_for_load: Optional[str],
) -> None:
    """Register phase-relevant HyperParameters (clone -> edit -> run must affect execution)."""

    try:
        if not getattr(clearml_mgr, "task", None):
            return
        models_spec: List[Dict[str, Any]] = []
        for spec in cfg.models:
            try:
                if hasattr(spec, "enable") and not spec.enable:
                    continue
            except Exception:
                pass
            try:
                models_spec.append(spec.model_dump())
            except Exception:
                models_spec.append({"name": getattr(spec, "name", ""), "params": getattr(spec, "params", {})})

        clearml_mgr.connect_params_sections(
            {
                "Training": {
                    "dataset_id": dataset_id_for_load or "",
                    "target_column": str(cfg.data.target_column or ""),
                    "test_size": float(cfg.data.test_size),
                    "random_seed": int(cfg.data.random_seed),
                    "cv_folds": cfg.cross_validation.n_folds,
                    "cv_shuffle": bool(cfg.cross_validation.shuffle),
                },
                "Models": {"models": models_spec},
                "Ensembles": cfg.ensembles.model_dump() if hasattr(cfg.ensembles, "model_dump") else {},
                "CrossValidation": cfg.cross_validation.model_dump()
                if hasattr(cfg.cross_validation, "model_dump")
                else {},
                "Evaluation": cfg.evaluation.model_dump() if hasattr(cfg.evaluation, "model_dump") else {},
                "Optimization": cfg.optimization.model_dump() if hasattr(cfg.optimization, "model_dump") else {},
            }
        )
    except Exception:
        pass


def connect_child_tasks_overview(
    clearml_mgr: Any,
    *,
    df_links: Optional[pd.DataFrame],
    df_links_ranked: Optional[pd.DataFrame],
    recommend_metric: Optional[str],
) -> None:
    try:
        if not getattr(clearml_mgr, "task", None):
            return
        df_src = (
            df_links_ranked
            if (df_links_ranked is not None and isinstance(df_links_ranked, pd.DataFrame) and not df_links_ranked.empty)
            else df_links
        )
        if df_src is None or not isinstance(df_src, pd.DataFrame) or df_src.empty:
            return

        top_k = 200
        df_show = df_src.head(top_k)
        cols_keep = [
            "rank",
            "is_recommended",
            "model",
            "preprocessor",
            "params",
            "task_id",
            "model_id",
            "status",
            "error",
            "metric_source",
            "train_seconds",
            "predict_seconds",
            "model_size_bytes",
            "num_features",
            "url",
        ]
        if recommend_metric and recommend_metric in df_show.columns:
            cols_keep.insert(2, str(recommend_metric))
        cols_keep = [c for c in cols_keep if c in df_show.columns]
        records = df_show[cols_keep].to_dict(orient="records")
        child_tasks_obj = {
            "recommend_metric": recommend_metric,
            "top_k": int(min(top_k, len(df_src))),
            "total": int(len(df_src)),
            "truncated": bool(len(df_src) > top_k),
            "items": json.loads(json.dumps(records, default=str)),
        }
        clearml_mgr.connect_configuration(child_tasks_obj, name="ChildTasks")
    except Exception:
        pass


def set_training_summary_user_properties(
    clearml_mgr: Any,
    *,
    run_id: str,
    dataset_id_for_load: Optional[str],
    dataset_key: str,
    dataset_role: str,
    target_column: str,
    primary_metric: str,
    preproc_manifest_src: Optional[Path],
    preproc_recipe_src: Optional[Path],
    recommended_df: Optional[pd.DataFrame],
    metric_for_bar: Optional[str],
) -> None:
    try:
        task = getattr(clearml_mgr, "task", None)
        if not task:
            return

        props: Dict[str, Any] = {
            "run_id": str(run_id),
            "dataset_id": str(dataset_id_for_load or ""),
            "dataset_key": str(dataset_key),
            "dataset_role": str(dataset_role),
            "target_column": str(target_column or ""),
            "primary_metric": str(primary_metric or ""),
        }

        try:
            if dataset_role == "preprocessed" and preproc_manifest_src and preproc_manifest_src.exists():
                rec = json.loads(preproc_manifest_src.read_text(encoding="utf-8"))
                if isinstance(rec, dict):
                    props["selected_preprocessor"] = str(rec.get("selected_preprocessor") or "")
                    props["source_dataset_id"] = str(rec.get("parent_dataset_id") or "")
        except Exception:
            pass

        try:
            if preproc_recipe_src and preproc_recipe_src.exists():
                recipe = json.loads(preproc_recipe_src.read_text(encoding="utf-8"))
                if isinstance(recipe, dict):
                    props["preproc_contract_version"] = str(recipe.get("contract_version") or "")
                    target = recipe.get("target") if isinstance(recipe.get("target"), dict) else {}
                    selected = recipe.get("selected_settings") if isinstance(recipe.get("selected_settings"), dict) else {}
                    props["preproc_target_transform"] = str(target.get("transform") or "")
                    props["preproc_scaling"] = str(selected.get("scaling") or "")
                    props["preproc_encoding"] = str(selected.get("categorical_encoding") or "")
                    props["preproc_numeric_imputation"] = str(selected.get("numeric_imputation") or "")
                    props["preproc_categorical_imputation"] = str(selected.get("categorical_imputation") or "")
                    poly = selected.get("polynomial_degree")
                    props["preproc_polynomial_degree"] = "" if poly is None else str(poly)
        except Exception:
            pass

        if recommended_df is not None and isinstance(recommended_df, pd.DataFrame) and not recommended_df.empty:
            try:
                rec_row = recommended_df.iloc[0].to_dict()
            except Exception:
                rec_row = {}
            props["recommended_model_name"] = str(rec_row.get("model") or "")
            props["recommended_model_id"] = str(rec_row.get("model_id") or "")
            props["recommended_model_task_id"] = str(rec_row.get("task_id") or "")
            try:
                key = str(metric_for_bar or "").strip()
                top1 = float(pd.to_numeric(rec_row.get(key), errors="coerce")) if key else float("nan")
                if top1 == top1:
                    props["leaderboard_top1_score"] = top1
            except Exception:
                pass

        set_user_properties(task, props)
    except Exception:
        pass


def connect_dataset_and_preprocessing_objects(
    clearml_mgr: Any,
    *,
    dataset_id_for_load: Optional[str],
    dataset_key: str,
    problem_type: str,
    X: Any,
    X_train: Any,
    X_test: Any,
    preprocessors: List[Any],
    feature_types: Dict[str, Any],
    has_preproc_contract: bool,
    preproc_bundle_available: bool,
    preproc_manifest_src: Optional[Path],
    preproc_schema_src: Optional[Path],
    preproc_recipe_src: Optional[Path],
    preproc_summary_src: Optional[Path],
    model_candidates: List[str],
) -> None:
    try:
        if not getattr(clearml_mgr, "task", None):
            return

        feature_types_summary: Dict[str, Any] = {}
        for key, cols in (feature_types or {}).items():
            try:
                feature_types_summary[str(key)] = {"n": int(len(cols or [])), "sample": list(cols or [])[:20]}
            except Exception:
                pass

        dataset_conf: Dict[str, Any] = {
            "dataset_id": dataset_id_for_load or "",
            "dataset_key": dataset_key,
            "problem_type": problem_type,
            "counts": {
                "n_total": int(len(X)) if X is not None else 0,
                "split": {
                    "n_train": int(len(X_train)) if X_train is not None else 0,
                    "n_test": int(len(X_test)) if X_test is not None else 0,
                },
            },
            "feature_types": feature_types_summary,
            "preprocessor_candidates": [name for name, _ in preprocessors],
            "model_candidates": model_candidates,
        }
        clearml_mgr.connect_configuration(dataset_conf, name="Dataset")

        def _read_text(path: Optional[Path], *, max_chars: int = 20000) -> str:
            if not path or not path.exists():
                return ""
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""
            if len(text) > max_chars:
                return text[:max_chars] + "..."
            return text

        preproc_conf: Dict[str, Any] = {
            "has_preprocessing_contract": bool(has_preproc_contract),
            "bundle_available": bool(preproc_bundle_available),
        }
        try:
            if preproc_manifest_src and preproc_manifest_src.exists():
                preproc_conf["manifest"] = json.loads(preproc_manifest_src.read_text(encoding="utf-8"))
        except Exception:
            pass
        try:
            if preproc_schema_src and preproc_schema_src.exists():
                preproc_conf["schema"] = json.loads(preproc_schema_src.read_text(encoding="utf-8"))
        except Exception:
            pass
        try:
            if preproc_recipe_src and preproc_recipe_src.exists():
                preproc_conf["recipe"] = json.loads(preproc_recipe_src.read_text(encoding="utf-8"))
        except Exception:
            pass
        summary_md = _read_text(preproc_summary_src)
        if summary_md:
            preproc_conf["summary_md"] = summary_md
        clearml_mgr.connect_configuration(preproc_conf, name="Preprocessing")
    except Exception:
        pass

