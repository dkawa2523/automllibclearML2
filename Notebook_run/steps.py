from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .clearml_utils import dataset_url, find_latest_task_by_prefix, task_url, task_user_properties
from .runtime import NotebookContext
from .ui import display_scrollable_df


def _parse_yaml_text(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyyaml が必要です: pip install pyyaml") from exc
    obj = yaml.safe_load(str(text))
    if not isinstance(obj, dict):
        raise ValueError("Edited YAML must be a dict")
    return obj


def _apply_default_if_unchanged(*, current: str, default_if_equals: str, new_default: Optional[str]) -> str:
    cur = str(current or "").strip()
    base = str(default_if_equals or "").strip()
    if not new_default:
        return cur
    if not cur or cur == base:
        return str(new_default).strip()
    return cur


def _read_recommended_model_csv(path: Path) -> Dict[str, Optional[str]]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
    except Exception:
        return {}
    if not isinstance(row, dict):
        return {}
    model_id = str(row.get("model_id") or "").strip() or None
    model_name = str(row.get("model") or "").strip() or None
    task_id = str(row.get("task_id") or "").strip() or None
    return {"model_id": model_id, "model_name": model_name, "task_id": task_id}


def _resolve_recommended_model_from_outputs(
    *,
    training_output_dir: Path,
    run_id: Optional[str],
) -> Dict[str, Optional[str]]:
    base = Path(training_output_dir)
    candidates: list[Path] = []
    if run_id:
        candidates.append(base / str(run_id) / "recommended_model.csv")
    candidates.append(base / "recommended_model.csv")
    for p in candidates:
        if not p.exists():
            continue
        rec = _read_recommended_model_csv(p)
        if rec.get("model_id"):
            rec["source_path"] = str(p)
            return rec
    try:
        globbed = list(base.glob("*/recommended_model.csv"))
        globbed.sort(key=lambda p: float(p.stat().st_mtime), reverse=True)
        for p in globbed:
            rec = _read_recommended_model_csv(p)
            if rec.get("model_id"):
                rec["source_path"] = str(p)
                return rec
    except Exception:
        pass
    return {}


def data_registration(ctx: NotebookContext, *, yaml_text: str, base_config_path: Path = Path("config_dataregit.yaml")) -> Dict[str, Any]:
    base_cfg = ctx.load_yaml(base_config_path)
    patch = _parse_yaml_text(yaml_text)
    merged = ctx.deep_update(base_cfg, patch)
    out_cfg_path = ctx.write_yaml(ctx.config_out_dir / "data_registration.yaml", merged)
    print(f"[data_registration] config saved: {out_cfg_path}")

    from automl_lib.cli.common import clearml_avoid_task_reuse
    from automl_lib.workflow.data_registration.processing import run_data_registration_processing

    clearml_avoid_task_reuse()
    info = run_data_registration_processing(out_cfg_path) or {}
    print("[data_registration] result:")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))

    dataset_id = str(info.get("dataset_id") or "").strip() or None
    task_id = str(info.get("task_id") or "").strip() or None
    csv_path = str(info.get("csv_path") or (merged.get("data") or {}).get("csv_path") or "").strip() or None
    run_id = str(info.get("run_id") or "").strip() or None

    if csv_path and Path(csv_path).exists():
        try:
            df = pd.read_csv(csv_path)
            display_scrollable_df(df, max_rows=50, max_height_px=320)
        except Exception:
            pass

    print(f"dataset_id: {dataset_id}")
    print(f"dataset_link: {dataset_url(dataset_id)}")
    print(f"task_id: {task_id}")
    print(f"task_link: {task_url(task_id)}")

    ctx.update_state(
        {
            "data_registration": {
                "dataset_id": dataset_id,
                "task_id": task_id,
                "csv_path": csv_path,
                "run_id": run_id,
            }
        }
    )
    print(f"[data_registration] state saved: {ctx.state_path}")
    return info


def pipeline_training(ctx: NotebookContext, *, yaml_text: str, base_config_path: Path = Path("config.yaml")) -> Dict[str, Any]:
    base_cfg = ctx.load_yaml(base_config_path)
    patch = _parse_yaml_text(yaml_text)
    merged = ctx.deep_update(base_cfg, patch)

    # Default dataset_id from previous data_registration, but do not override user changes.
    state = ctx.load_state()
    reg_dsid = str(((state.get("data_registration") or {}).get("dataset_id") or "")).strip() or None
    base_dsid = str(((base_cfg.get("data") or {}).get("dataset_id") or "")).strip()
    cur_dsid = str(((merged.get("data") or {}).get("dataset_id") or "")).strip()
    dsid = _apply_default_if_unchanged(current=cur_dsid, default_if_equals=base_dsid, new_default=reg_dsid)
    merged.setdefault("data", {})
    merged["data"]["dataset_id"] = dsid

    # This cell is preprocessing+training only (no inference).
    merged.setdefault("clearml", {})
    merged["clearml"]["enable_inference"] = False

    out_cfg_path = ctx.write_yaml(ctx.config_out_dir / "pipeline_training.yaml", merged)
    print(f"[pipeline] config saved: {out_cfg_path}")

    from automl_lib.cli.common import clearml_avoid_task_reuse
    from automl_lib.pipeline.controller import run_pipeline

    clearml_avoid_task_reuse()
    pipe_info = run_pipeline(out_cfg_path, mode="clearml") or {}
    print("[pipeline] controller result:")
    print(json.dumps(pipe_info, ensure_ascii=False, indent=2, default=str))

    run_id = str(pipe_info.get("run_id") or "").strip() or None
    pipeline_task_id = str(pipe_info.get("pipeline_task_id") or "").strip() or None
    print(f"pipeline_task_id: {pipeline_task_id}")
    print(f"pipeline_link: {task_url(pipeline_task_id)}")

    preproc_task = find_latest_task_by_prefix(run_id=run_id or "", prefix="preprocessing") if run_id else None
    train_summary_task = find_latest_task_by_prefix(run_id=run_id or "", prefix="training-summary") if run_id else None

    preproc_props = task_user_properties(preproc_task)
    train_props = task_user_properties(train_summary_task)

    preprocessed_dataset_id = (preproc_props.get("preprocessed_dataset_id") or "").strip() or None
    training_dataset_id = (train_props.get("dataset_id") or "").strip() or None
    recommended_model_id = (train_props.get("recommended_model_id") or "").strip() or None
    recommended_model_name = (train_props.get("recommended_model_name") or "").strip() or None
    training_summary_task_id = str(getattr(train_summary_task, "id", "") or "").strip() or None
    preprocessing_task_id = str(getattr(preproc_task, "id", "") or "").strip() or None

    training_output_dir = str(((merged.get("output") or {}).get("output_dir") or "outputs/train")).strip()

    # Fallback when ClearML task lookup fails: read local training artifacts.
    if not recommended_model_id:
        rec = _resolve_recommended_model_from_outputs(
            training_output_dir=Path(training_output_dir),
            run_id=run_id,
        )
        recommended_model_id = str(rec.get("model_id") or "").strip() or None
        if not recommended_model_name:
            recommended_model_name = str(rec.get("model_name") or "").strip() or None

    print("\n--- Preprocessing ---")
    print(f"preprocessing_task_id: {preprocessing_task_id}")
    print(f"preprocessing_task_link: {task_url(preprocessing_task_id)}")
    print(f"preprocessed_dataset_id: {preprocessed_dataset_id}")
    print(f"preprocessed_dataset_link: {dataset_url(preprocessed_dataset_id)}")

    print("\n--- Training Summary ---")
    print(f"training_summary_task_id: {training_summary_task_id}")
    print(f"training_summary_link: {task_url(training_summary_task_id)}")
    print(f"training_dataset_id: {training_dataset_id}")
    print(f"recommended_model_name: {recommended_model_name}")
    print(f"recommended_model_id: {recommended_model_id}")

    # Suggestion YAML for inference inputs (copy/paste friendly).
    target = str(((merged.get("data") or {}).get("target_column") or "target")).strip()
    suggest_single: Dict[str, Any] = {
        "model_id": str(recommended_model_id or ""),
        "model_name": str(recommended_model_name or "best_model"),
        "input": {"mode": "single", "single": {}},
    }
    suggest_opt: Dict[str, Any] = {
        "model_id": str(recommended_model_id or ""),
        "model_name": str(recommended_model_name or "best_model"),
        "input": {"mode": "optimize", "variables": []},
        "search": {"method": "tpe", "n_trials": 10, "goal": "max"},
    }

    def _to_builtin_number(value: Any) -> Any:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                value = value.item()
        except Exception:
            pass
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return float(value)
        return value

    def _nice_step(*, vtype: str, vmin: float, vmax: float) -> Any:
        span = float(vmax) - float(vmin)
        if not math.isfinite(span) or span <= 0:
            return 1 if vtype == "int" else 1.0
        if vtype == "int":
            step = max(1, int(round(span / 10.0)))
            return int(step)
        step = span / 10.0
        # Keep YAML readable (avoid long decimals).
        step_rounded = round(step, 6)
        return step_rounded if step_rounded > 0 else step

    try:
        from automl_lib.integrations.clearml import load_input_model
        from automl_lib.inference.model_utils import _get_required_feature_names
        import joblib  # type: ignore

        req: list[str] = []
        if recommended_model_id:
            local_model = load_input_model(str(recommended_model_id))
            if local_model:
                pipe = joblib.load(local_model)
                req = [str(c) for c in (_get_required_feature_names(pipe) or []) if str(c).strip()]
                if target in req:
                    req = [c for c in req if c != target]

        # Prefer dataset-driven examples (continuous features only).
        cfg_data = merged.get("data") or {}
        feature_columns = cfg_data.get("feature_columns")
        if not isinstance(feature_columns, list):
            feature_columns = None
        feature_columns = [str(c) for c in feature_columns] if feature_columns else None

        state = ctx.load_state()
        csv_path = (
            str(((state.get("data_registration") or {}).get("csv_path") or "")).strip()
            or str((cfg_data.get("csv_path") or "")).strip()
            or None
        )

        stats: Dict[str, Dict[str, Any]] = {}
        if csv_path and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            if feature_columns:
                candidate_cols = [c for c in feature_columns if c in df.columns]
            else:
                candidate_cols = [c for c in df.columns if c != target]

            # "連続値のみ" = numeric dtypes only (exclude categorical/object).
            numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
            for col in numeric_cols:
                s = pd.to_numeric(df[col], errors="coerce")
                if not s.notna().any():
                    continue
                vmin = _to_builtin_number(s.min())
                vmax = _to_builtin_number(s.max())
                if vmin is None or vmax is None:
                    continue
                col_is_int = pd.api.types.is_integer_dtype(df[col])
                stats[col] = {"min": vmin, "max": vmax, "type": "int" if col_is_int else "float"}

        if stats:
            ordered_cols = req[:] if req else list(stats.keys())
            ordered_cols = [c for c in ordered_cols if c in stats]
            suggest_single["input"]["single"] = {c: stats[c]["min"] for c in ordered_cols}
            suggest_opt["input"]["variables"] = [
                {
                    "name": c,
                    "type": stats[c]["type"],
                    "method": "range",
                    "min": stats[c]["min"],
                    "max": stats[c]["max"],
                    "step": _nice_step(
                        vtype=str(stats[c]["type"]),
                        vmin=float(stats[c]["min"]),
                        vmax=float(stats[c]["max"]),
                    ),
                }
                for c in ordered_cols
            ]
        elif req:
            # Fallback: show required columns without stats (still better than empty).
            req = list(req)[:20]
            suggest_single["input"]["single"] = {c: "" for c in req}
            suggest_opt["input"]["variables"] = [
                {"name": c, "type": "float", "method": "range", "min": 0, "max": 1, "step": 0.1}
                for c in req[:2]
            ]
    except Exception:
        pass

    print("\n--- Inference input examples (copy/paste) ---")
    print("# suggest_single_yaml")
    print(ctx.dump_yaml(suggest_single))
    print("# suggest_optimize_yaml")
    print(ctx.dump_yaml(suggest_opt))

    ctx.update_state(
        {
            "pipeline": {
                "run_id": run_id,
                "pipeline_task_id": pipeline_task_id,
                "preprocessing_task_id": preprocessing_task_id,
                "preprocessed_dataset_id": preprocessed_dataset_id,
                "training_summary_task_id": training_summary_task_id,
                "recommended_model_id": recommended_model_id,
                "recommended_model_name": recommended_model_name,
                "training_output_dir": training_output_dir,
                "suggest_single_yaml": ctx.dump_yaml(suggest_single),
                "suggest_optimize_yaml": ctx.dump_yaml(suggest_opt),
            }
        }
    )
    print(f"[pipeline] state saved: {ctx.state_path}")
    return pipe_info


def inference_single(ctx: NotebookContext, *, yaml_text: str, base_config_path: Path = Path("inference_config.yaml")) -> Dict[str, Any]:
    base_cfg = ctx.load_yaml(base_config_path)
    patch = _parse_yaml_text(yaml_text)

    # Guard: do not run if input.single is empty (avoid confusing no-op).
    row = ((patch.get("input") or {}).get("single") or {})
    if not isinstance(row, dict) or not row:
        raise ValueError(
            "input.single が空です。\n"
            "前処理+学習セルの出力 'suggest_single_yaml' をコピペして埋めてから実行してください。"
        )

    # Resolve default model_id from previous training (pipeline state or local artifacts).
    state = ctx.load_state()
    pipe_state = state.get("pipeline") or {}
    rec_mid = str((pipe_state.get("recommended_model_id") or "")).strip() or None
    rec_name = str((pipe_state.get("recommended_model_name") or "")).strip() or None
    run_id = str((pipe_state.get("run_id") or "")).strip() or None
    training_output_dir = str((pipe_state.get("training_output_dir") or "outputs/train")).strip()
    if not rec_mid:
        rec = _resolve_recommended_model_from_outputs(training_output_dir=Path(training_output_dir), run_id=run_id)
        rec_mid = str(rec.get("model_id") or "").strip() or None
        if not rec_name:
            rec_name = str(rec.get("model_name") or "").strip() or None
        if rec_mid:
            ctx.update_state({"pipeline": {"recommended_model_id": rec_mid, "recommended_model_name": rec_name}})

    # Backward compatible: accept recommended_model_id in edited YAML, but always write model_id.
    if not str(patch.get("model_id") or "").strip() and str(patch.get("recommended_model_id") or "").strip():
        patch["model_id"] = str(patch.get("recommended_model_id") or "").strip()
    patch.pop("recommended_model_id", None)

    user_model_id = str(patch.get("model_id") or "").strip() or None
    chosen_model_id = user_model_id or rec_mid
    if not chosen_model_id:
        raise ValueError(
            "model_id が空です。\n"
            "- 先に pipeline_training を成功させて recommended_model_id を取得するか\n"
            "- inference YAML に model_id（training-summary の USER PROPERTIES: recommended_model_id）を設定してください。"
        )

    patch["model_id"] = str(chosen_model_id)
    if rec_name and not str(patch.get("model_name") or "").strip():
        patch["model_name"] = rec_name

    merged = ctx.deep_update(base_cfg, patch)
    merged.pop("recommended_model_id", None)
    out_cfg_path = ctx.write_yaml(ctx.config_out_dir / "inference_single.yaml", merged)
    print(f"[inference single] config saved: {out_cfg_path}")

    from automl_lib.cli.common import clearml_avoid_task_reuse
    from automl_lib.workflow import run_inference

    clearml_avoid_task_reuse()
    info = run_inference(out_cfg_path) or {}
    print("[inference single] result:")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))

    task_id = str(info.get("task_id") or "").strip() or None
    output_dir = Path(str(info.get("output_dir") or "")) if info.get("output_dir") else None

    row = {}
    pred = None
    try:
        if output_dir and (output_dir / "input.json").exists():
            row = json.loads((output_dir / "input.json").read_text(encoding="utf-8")).get("row") or {}
        if output_dir and (output_dir / "output.json").exists():
            pred = json.loads((output_dir / "output.json").read_text(encoding="utf-8")).get("prediction")
    except Exception:
        pass

    df = pd.DataFrame([dict(row or {}, prediction=pred)])
    display_scrollable_df(df, max_rows=1, max_height_px=240)
    print(f"task_id: {task_id}")
    print(f"task_link: {task_url(task_id)}")

    ctx.update_state({"inference_single": {"task_id": task_id, "output_dir": str(output_dir) if output_dir else None}})
    print(f"[inference single] state saved: {ctx.state_path}")
    return info


def inference_optimize(ctx: NotebookContext, *, yaml_text: str, base_config_path: Path = Path("inference_config_optimize.yaml")) -> Dict[str, Any]:
    base_cfg = ctx.load_yaml(base_config_path)
    patch = _parse_yaml_text(yaml_text)

    vars_list = ((patch.get("input") or {}).get("variables") or [])
    if not isinstance(vars_list, list) or len(vars_list) == 0:
        raise ValueError(
            "input.variables が空です。\n"
            "前処理+学習セルの出力 'suggest_optimize_yaml' をコピペして埋めてから実行してください。"
        )

    # Resolve default model_id from previous training (pipeline state or local artifacts).
    state = ctx.load_state()
    pipe_state = state.get("pipeline") or {}
    rec_mid = str((pipe_state.get("recommended_model_id") or "")).strip() or None
    rec_name = str((pipe_state.get("recommended_model_name") or "")).strip() or None
    run_id = str((pipe_state.get("run_id") or "")).strip() or None
    training_output_dir = str((pipe_state.get("training_output_dir") or "outputs/train")).strip()
    if not rec_mid:
        rec = _resolve_recommended_model_from_outputs(training_output_dir=Path(training_output_dir), run_id=run_id)
        rec_mid = str(rec.get("model_id") or "").strip() or None
        if not rec_name:
            rec_name = str(rec.get("model_name") or "").strip() or None
        if rec_mid:
            ctx.update_state({"pipeline": {"recommended_model_id": rec_mid, "recommended_model_name": rec_name}})

    # Backward compatible: accept recommended_model_id in edited YAML, but always write model_id.
    if not str(patch.get("model_id") or "").strip() and str(patch.get("recommended_model_id") or "").strip():
        patch["model_id"] = str(patch.get("recommended_model_id") or "").strip()
    patch.pop("recommended_model_id", None)

    user_model_id = str(patch.get("model_id") or "").strip() or None
    chosen_model_id = user_model_id or rec_mid
    if not chosen_model_id:
        raise ValueError(
            "model_id が空です。\n"
            "- 先に pipeline_training を成功させて recommended_model_id を取得するか\n"
            "- inference YAML に model_id（training-summary の USER PROPERTIES: recommended_model_id）を設定してください。"
        )
    patch["model_id"] = str(chosen_model_id)
    if rec_name and not str(patch.get("model_name") or "").strip():
        patch["model_name"] = rec_name

    merged = ctx.deep_update(base_cfg, patch)
    merged.pop("recommended_model_id", None)
    out_cfg_path = ctx.write_yaml(ctx.config_out_dir / "inference_optimize.yaml", merged)
    print(f"[inference optimize] config saved: {out_cfg_path}")

    from automl_lib.cli.common import clearml_avoid_task_reuse
    from automl_lib.workflow import run_inference

    clearml_avoid_task_reuse()
    info = run_inference(out_cfg_path) or {}
    print("[inference optimize] result:")
    print(json.dumps(info, ensure_ascii=False, indent=2, default=str))

    summary_task_id = str(info.get("task_id") or "").strip() or None
    child_task_ids = list(info.get("child_task_ids") or [])
    output_dir = Path(str(info.get("output_dir") or "")) if info.get("output_dir") else None

    print(f"summary_task_id: {summary_task_id}")
    print(f"summary_link: {task_url(summary_task_id)}")

    df_trials = None
    if output_dir and (output_dir / "trials.csv").exists():
        try:
            df_trials = pd.read_csv(output_dir / "trials.csv")
        except Exception:
            df_trials = None

    goal = str(((merged.get("search") or {}).get("goal") or "max")).strip().lower()
    top_k = int(((merged.get("search") or {}).get("top_k") or 10))
    top_k = max(1, top_k)

    if df_trials is not None and not df_trials.empty and "prediction" in df_trials.columns:
        df = df_trials.copy()
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        df = df.dropna(subset=["prediction"])
        if "trial_index" not in df.columns:
            df["trial_index"] = range(len(df))
        df = df.sort_values("trial_index", kind="mergesort")
        if goal == "min":
            df["best_so_far"] = df["prediction"].cummin()
        else:
            df["best_so_far"] = df["prediction"].cummax()
        print("\n--- Optimization log (tail) ---")
        display_scrollable_df(df[["trial_index", "prediction", "best_so_far"]].tail(30), max_rows=30, max_height_px=320)

        ascending = True if goal == "min" else False
        df_rank = df.sort_values("prediction", ascending=ascending, kind="mergesort")
        df_topk = df_rank.head(top_k)
        print("\n--- TopK (conditions + prediction) ---")
        display_scrollable_df(df_topk, max_rows=min(top_k, 50), max_height_px=360)
    else:
        print("trials.csv が見つからない/空のため TopK 表示をスキップしました")

    print("\n--- Prediction_runs ---")
    print(f"child_tasks: {len(child_task_ids)}")
    try:
        if child_task_ids:
            from clearml import Task  # type: ignore

            t0 = Task.get_task(task_id=str(child_task_ids[0]))
            print(f"child_project: {getattr(t0, 'project', '')}")
            print(f"child_task_link_example: {t0.get_output_log_web_page()}")
    except Exception:
        pass

    ctx.update_state(
        {
            "inference_optimize": {
                "summary_task_id": summary_task_id,
                "child_task_ids": child_task_ids,
                "output_dir": str(output_dir) if output_dir else None,
            }
        }
    )
    print(f"[inference optimize] state saved: {ctx.state_path}")
    return info
