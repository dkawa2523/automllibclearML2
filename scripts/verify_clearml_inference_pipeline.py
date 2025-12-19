#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXPECTED_ARTIFACTS_BY_MODE = {
    "single": {"inference_config.json", "input.json", "output.json"},
    "batch": {"inference_config.json", "predictions.csv"},
    "optimize": {"inference_config.json", "trials.csv", "best_solution.json"},
}
OPTIONAL_ARTIFACTS = {"input_meta.json"}


def _task_name(task: Any) -> str:
    try:
        return str(getattr(task, "name", "") or "").strip()
    except Exception:
        return ""


def _task_status(task: Any) -> str:
    try:
        st = getattr(task, "get_status", lambda: "")()
        return str(st or "").strip().lower()
    except Exception:
        try:
            data = getattr(task, "data", None)
            st = getattr(data, "status", "") if data is not None else ""
            return str(st or "").strip().lower()
        except Exception:
            return ""


def _pick_latest(tasks: Sequence[Any]) -> Optional[Any]:
    if not tasks:
        return None
    try:
        return sorted(
            tasks,
            key=lambda t: float(getattr(getattr(t, "data", None), "last_update", 0) or 0),
        )[-1]
    except Exception:
        return tasks[-1]


def _artifact_names(task: Any) -> List[str]:
    names: set[str] = set()

    # `upload_artifact()` shows up under `task.artifacts` (not `get_registered_artifacts()`).
    try:
        arts = getattr(task, "artifacts", None)
        if arts is not None:
            try:
                for k in arts.keys():  # type: ignore[attr-defined]
                    key = str(k).strip()
                    if key:
                        names.add(key)
            except Exception:
                if isinstance(arts, dict):
                    for k in arts.keys():
                        key = str(k).strip()
                        if key:
                            names.add(key)
    except Exception:
        pass

    if names:
        return sorted(names)

    # Fallback: dynamically registered artifacts (register_artifact).
    try:
        arts = task.get_registered_artifacts()
        if isinstance(arts, dict):
            for k in arts.keys():
                key = str(k).strip()
                if key:
                    names.add(key)
    except Exception:
        pass

    return sorted(names)


def _user_properties(task: Any) -> Dict[str, str]:
    # Prefer ClearML's native "value_only" conversion.
    try:
        props = task.get_user_properties(value_only=True)
        if isinstance(props, dict):
            out: Dict[str, str] = {}
            for k, v in props.items():
                key = str(k).strip()
                if not key:
                    continue
                out[key] = str(v)
            return out
    except TypeError:
        pass
    except Exception:
        pass

    # Fallback: older ClearML versions may return a dict of details.
    try:
        props = task.get_user_properties()
    except Exception:
        return {}
    if not isinstance(props, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in props.items():
        key = str(k).strip()
        if not key:
            continue
        if isinstance(v, dict) and "value" in v:
            out[key] = str(v.get("value"))
        else:
            out[key] = str(v)
    return out


def _wait_for_task_by_prefix(*, run_id: str, prefix: str, timeout_seconds: float, poll_seconds: float) -> Any:
    from clearml import Task  # type: ignore

    deadline = time.time() + float(timeout_seconds)
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            tasks = Task.get_tasks(tags=[f"run:{run_id}"])
            candidates = [t for t in (tasks or []) if _task_name(t).startswith(prefix)]
            picked = _pick_latest(candidates)
            if picked is not None:
                return picked
        except Exception as exc:
            last_err = exc
        time.sleep(float(poll_seconds))
    if last_err is not None:
        raise RuntimeError(f"Failed to locate task prefix='{prefix}' (run_id={run_id}). Last error: {last_err}")
    raise RuntimeError(f"Failed to locate task prefix='{prefix}' (run_id={run_id}).")


def _wait_for_task_by_prefix_absent(*, run_id: str, prefix: str, timeout_seconds: float, poll_seconds: float) -> None:
    """Wait until there are no tasks with the given prefix for the run_id (best-effort)."""

    from clearml import Task  # type: ignore

    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        try:
            tasks = Task.get_tasks(tags=[f"run:{run_id}"])
            candidates = [t for t in (tasks or []) if _task_name(t).startswith(prefix)]
            if not candidates:
                return
        except Exception:
            return
        time.sleep(float(poll_seconds))
    # If tasks remain, keep going (we will fail later if expectations require absence).
    return


def _wait_for_inference_child_task(*, run_id: str, mode: str, timeout_seconds: float, poll_seconds: float) -> Any:
    from clearml import Task  # type: ignore

    mode = str(mode).strip().lower()
    deadline = time.time() + float(timeout_seconds)
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            tasks = Task.get_tasks(tags=[f"run:{run_id}"])
            # name: "infer <mode> model:..."
            candidates = [t for t in (tasks or []) if _task_name(t).startswith(f"infer {mode} ")]
            picked = _pick_latest(candidates)
            if picked is not None:
                return picked
        except Exception as exc:
            last_err = exc
        time.sleep(float(poll_seconds))
    if last_err is not None:
        raise RuntimeError(f"Failed to locate inference child task (run_id={run_id}). Last error: {last_err}")
    raise RuntimeError(f"Failed to locate inference child task (run_id={run_id}).")


def _count_trials_csv_rows(path: Path) -> Optional[int]:
    """Return number of trial rows with non-empty prediction, or None if unavailable."""

    try:
        import csv

        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = 0
            for r in reader:
                if not isinstance(r, dict):
                    continue
                pred = r.get("prediction")
                if pred is None:
                    continue
                if str(pred).strip() == "":
                    continue
                rows += 1
            return int(rows)
    except Exception:
        return None


def _wait_for_predict_children_count(
    *,
    run_id: str,
    expected: int,
    timeout_seconds: float,
    poll_seconds: float,
) -> List[Any]:
    """Wait until expected number of 'predict ...' tasks show up (best-effort)."""

    from clearml import Task  # type: ignore

    deadline = time.time() + float(timeout_seconds)
    last: List[Any] = []
    while time.time() < deadline:
        try:
            tasks = Task.get_tasks(tags=[f"run:{run_id}"])
            last = [t for t in (tasks or []) if _task_name(t).startswith("predict ")]
            if len(last) >= int(expected):
                return last
        except Exception:
            return last
        time.sleep(float(poll_seconds))
    return last


def _wait_for_expected_artifacts(
    task: Any,
    *,
    expected: Sequence[str],
    timeout_seconds: float,
    poll_seconds: float,
) -> List[str]:
    expected_set = {str(x).strip() for x in expected if str(x).strip()}
    deadline = time.time() + float(timeout_seconds)
    last: List[str] = []
    while time.time() < deadline:
        try:
            task.reload()
        except Exception:
            pass
        last = _artifact_names(task)
        got = set(last)
        missing = sorted(expected_set - got)
        if not missing:
            return last
        st = _task_status(task)
        if st in {"failed", "aborted", "stopped"}:
            raise RuntimeError(
                f"Task reached terminal status '{st}' before expected artifacts were registered. "
                f"Missing={missing} got={last}"
            )
        time.sleep(float(poll_seconds))
    raise RuntimeError(f"Timed out waiting for artifacts. Missing={sorted(expected_set - set(last))} got={last}")


def _run_pipeline(
    *,
    config_path: Path,
    inference_config_path: Path,
    training_config_data: Dict[str, Any],
    inference_config_data: Dict[str, Any],
) -> Dict[str, Any]:
    from automl_lib.pipeline.controller import run_pipeline

    return run_pipeline(
        config_path,
        mode="clearml",
        inference_config=inference_config_path,
        training_config_data=training_config_data,
        inference_config_data=inference_config_data,
    )


def _set_nested(d: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    cur: Any = d
    for key in path[:-1]:
        if not isinstance(cur, dict):
            return
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    if isinstance(cur, dict):
        cur[path[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a new ClearML pipeline with inference enabled and verify inference tasks/artifacts exist."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to training config YAML.")
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=Path("inference_config.yaml"),
        help="Path to inference config YAML.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0, help="Timeout waiting for tasks/artifacts.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval.")
    parser.add_argument("--run-id", type=str, default=None, help="If set, only verify the specified run_id.")
    args = parser.parse_args()

    if os.environ.get("CLEARML_OFFLINE_MODE"):
        raise SystemExit("CLEARML_OFFLINE_MODE is set; disable it to verify against a real ClearML server.")

    try:
        from automl_lib.config.loaders import load_yaml
        from automl_lib.integrations.clearml.bootstrap import ensure_clearml_config_file
        from automl_lib.integrations.clearml.context import generate_run_id, set_run_id_env
    except Exception as exc:
        raise SystemExit(f"Failed importing automl_lib. Run from repo root. Reason: {exc}") from exc

    ensure_clearml_config_file()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    inference_config_path = Path(args.inference_config)
    if not inference_config_path.exists():
        raise SystemExit(f"Inference config not found: {inference_config_path}")

    run_id = str(args.run_id).strip() if args.run_id else generate_run_id()
    inference_cfg = load_yaml(inference_config_path)
    if not isinstance(inference_cfg, dict):
        raise SystemExit(f"Invalid inference config YAML: {inference_config_path}")

    if not args.run_id:
        set_run_id_env(run_id)
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""

        training_cfg = load_yaml(config_path)
        if not isinstance(training_cfg, dict):
            raise SystemExit(f"Invalid training config YAML: {config_path}")

        # Force inference enabled for this run (keep run local by default to avoid requiring agents).
        _set_nested(training_cfg, ["clearml", "enable_inference"], True)
        _set_nested(training_cfg, ["clearml", "run_pipeline_locally"], True)
        _set_nested(training_cfg, ["clearml", "run_tasks_locally"], True)

        # Make it obvious if pipeline hand-off does not override model_id.
        inference_cfg["model_id"] = "00000000000000000000000000000000"

        print(f"[verify] Starting pipeline for run_id={run_id} config={config_path}")
        info = _run_pipeline(
            config_path=config_path,
            inference_config_path=inference_config_path,
            training_config_data=training_cfg,
            inference_config_data=inference_cfg,
        )
        print("[verify] Pipeline finished:")
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        run_id = str(info.get("run_id") or run_id)
    else:
        print(f"[verify] Verifying existing run_id={run_id}")

    training_summary = _wait_for_task_by_prefix(
        run_id=run_id,
        prefix="training-summary",
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )
    training_props = _user_properties(training_summary)
    recommended_model_id = str(training_props.get("recommended_model_id") or "").strip()
    if not recommended_model_id:
        raise SystemExit(
            "training-summary is missing USER PROPERTIES 'recommended_model_id'. "
            "Inference hand-off cannot be verified."
        )

    # Determine expected mode from the inference config (not from ClearML props),
    # because single/batch no longer create an inference-summary task.
    cfg_mode = ""
    try:
        cfg_mode = str(((inference_cfg or {}).get("input") or {}).get("mode") or "").strip().lower()
    except Exception:
        cfg_mode = ""
    if cfg_mode == "csv":
        cfg_mode = "batch"
    if cfg_mode == "params":
        cfg_mode = "optimize"
    mode = cfg_mode or "single"
    if mode not in EXPECTED_ARTIFACTS_BY_MODE:
        raise SystemExit(f"Unsupported inference mode in config: {mode}")

    expected = EXPECTED_ARTIFACTS_BY_MODE[mode]

    if mode in {"single", "batch"}:
        # No inference-summary task should be created.
        _wait_for_task_by_prefix_absent(
            run_id=run_id,
            prefix="inference-summary",
            timeout_seconds=10.0,
            poll_seconds=1.0,
        )

        inference_task = _wait_for_task_by_prefix(
            run_id=run_id,
            prefix=f"infer {mode} ",
            timeout_seconds=float(args.timeout_seconds),
            poll_seconds=float(args.poll_seconds),
        )
        inference_props = _user_properties(inference_task)

        artifacts = _wait_for_expected_artifacts(
            inference_task,
            expected=expected,
            timeout_seconds=float(args.timeout_seconds),
            poll_seconds=float(args.poll_seconds),
        )

        inferred_model_id = str(inference_props.get("model_id") or "").strip()
        if inferred_model_id != recommended_model_id:
            raise SystemExit(
                "Inference did not use training recommended_model_id. "
                f"training recommended_model_id={recommended_model_id} inference model_id={inferred_model_id}"
            )

        print(f"[verify] training-summary task_id={getattr(training_summary, 'id', '')} name={_task_name(training_summary)}")
        try:
            print(f"[verify] training-summary url={training_summary.get_output_log_web_page()}")
        except Exception:
            pass
        print(f"[verify] inference task_id={getattr(inference_task, 'id', '')} name={_task_name(inference_task)}")
        try:
            print(f"[verify] inference url={inference_task.get_output_log_web_page()}")
        except Exception:
            pass
        print("[verify] training recommended_model_id:")
        print(f"- {recommended_model_id}")
        print("[verify] inference USER PROPERTIES:")
        for k in sorted(inference_props.keys()):
            print(f"- {k}: {inference_props[k]}")
        print("[verify] inference artifacts:")
        for a in artifacts:
            print(f"- {a}")

        extra = sorted(set(artifacts) - set(expected) - OPTIONAL_ARTIFACTS)
        if extra:
            print("[verify] Note: extra artifacts were present (allowed):")
            print(f"- extras: {extra}")

        print("[verify] OK: inference task/artifacts are present and model_id hand-off works.")
        return

    # optimize mode: inference-summary + prediction_run children
    inference_summary = _wait_for_task_by_prefix(
        run_id=run_id,
        prefix="inference-summary",
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )
    inference_props = _user_properties(inference_summary)
    inferred_model_id = str(inference_props.get("model_id") or "").strip()
    if inferred_model_id != recommended_model_id:
        raise SystemExit(
            "Inference did not use training recommended_model_id. "
            f"training recommended_model_id={recommended_model_id} inference model_id={inferred_model_id}"
        )

    summary_artifacts = _wait_for_expected_artifacts(
        inference_summary,
        expected=expected,
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )

    # "predict ..." child tasks should exist (all trials).
    from clearml import Task  # type: ignore

    expected_children = None
    try:
        out_base = Path(str((inference_cfg or {}).get("output_dir") or "outputs/inference"))
        expected_children = _count_trials_csv_rows(out_base / run_id / "trials.csv")
    except Exception:
        expected_children = None

    tasks = Task.get_tasks(tags=[f"run:{run_id}"])
    predict_children = [t for t in (tasks or []) if _task_name(t).startswith("predict ")]
    if expected_children is not None:
        predict_children = _wait_for_predict_children_count(
            run_id=run_id,
            expected=int(expected_children),
            timeout_seconds=float(args.timeout_seconds),
            poll_seconds=float(args.poll_seconds),
        )
    if not predict_children:
        raise SystemExit("No Prediction_runs child tasks found (expected at least 1 'predict ...' task).")
    if expected_children is not None and len(predict_children) != int(expected_children):
        raise SystemExit(
            "Unexpected number of Prediction_runs child tasks. "
            f"expected={expected_children} got={len(predict_children)}"
        )

    print(f"[verify] training-summary task_id={getattr(training_summary, 'id', '')} name={_task_name(training_summary)}")
    try:
        print(f"[verify] training-summary url={training_summary.get_output_log_web_page()}")
    except Exception:
        pass
    print(f"[verify] inference-summary task_id={getattr(inference_summary, 'id', '')} name={_task_name(inference_summary)}")
    try:
        print(f"[verify] inference-summary url={inference_summary.get_output_log_web_page()}")
    except Exception:
        pass
    print("[verify] training recommended_model_id:")
    print(f"- {recommended_model_id}")
    print("[verify] inference-summary artifacts:")
    for a in summary_artifacts:
        print(f"- {a}")
    if expected_children is not None:
        print(f"[verify] predict child tasks: {len(predict_children)} (expected={expected_children})")
    else:
        print(f"[verify] predict child tasks: {len(predict_children)}")
    for t in sorted(predict_children, key=lambda x: _task_name(x))[:10]:
        print(f"- {getattr(t, 'id', '')} { _task_name(t)}")

    print("[verify] OK: optimize inference-summary + Prediction_runs children exist and model_id hand-off works.")


if __name__ == "__main__":
    main()
