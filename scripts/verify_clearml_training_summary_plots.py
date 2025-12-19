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


EXPECTED_PLOT_METRICS_EXACT = {
    "01_Recommended Model",
    "02_Leaderboard",
    "03_Leaderboard Table",
    "04_Tradeoff",
    "05_Scatter Plot of Recommended Model",
    "06_Feature Importance from Recommended Model",
    "08_SHAP values",
}
EXPECTED_PLOT_METRICS_PREFIX = "07_Interpolation space:"

EXPECTED_HYPERPARAM_SECTIONS = {
    "Training",
    "Models",
    "Ensembles",
    "CrossValidation",
    "Evaluation",
    "Optimization",
}


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
    # Prefer "last_update" if available; fall back to insertion order.
    try:
        return sorted(
            tasks,
            key=lambda t: float(getattr(getattr(t, "data", None), "last_update", 0) or 0),
        )[-1]
    except Exception:
        return tasks[-1]


def _wait_for_training_summary_task(*, run_id: str, timeout_seconds: float, poll_seconds: float) -> Any:
    from clearml import Task  # type: ignore

    deadline = time.time() + float(timeout_seconds)
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            tasks = Task.get_tasks(tags=[f"run:{run_id}"])
            candidates = [t for t in (tasks or []) if _task_name(t).startswith("training-summary")]
            picked = _pick_latest(candidates)
            if picked is not None:
                return picked
        except Exception as exc:
            last_err = exc
        time.sleep(float(poll_seconds))
    if last_err is not None:
        raise RuntimeError(f"Failed to locate training-summary task (run_id={run_id}). Last error: {last_err}")
    raise RuntimeError(f"Failed to locate training-summary task (run_id={run_id}).")


def _unique_plot_metrics(task: Any) -> List[str]:
    try:
        reported = task.get_reported_plots()
    except Exception:
        reported = []
    metrics: List[str] = []
    for rec in reported or []:
        if not isinstance(rec, dict):
            continue
        metric = rec.get("metric")
        if not metric:
            continue
        metrics.append(str(metric))
    # stable order, de-dup
    seen = set()
    uniq: List[str] = []
    for m in metrics:
        if m in seen:
            continue
        seen.add(m)
        uniq.append(m)
    return uniq


def _hyperparam_sections(task: Any) -> List[str]:
    try:
        params = task.get_parameters_as_dict(cast=True)
    except Exception:
        params = None
    if not isinstance(params, dict):
        return []
    out = []
    for k, v in params.items():
        if not isinstance(k, str):
            continue
        if not k.strip():
            continue
        if isinstance(v, dict):
            out.append(k.strip())
    # stable order
    return sorted(set(out))


def _get_training_debug_image_metrics(task_id: str) -> List[str]:
    try:
        from clearml.backend_api.session import Session  # type: ignore
        from clearml.backend_api.services.v2_9 import events  # type: ignore
    except Exception:
        return []

    try:
        session = Session()
        resp = session.send(
            events.GetTaskMetricsRequest(
                tasks=[str(task_id)],
                event_type=events.EventTypeEnum.training_debug_image,
            )
        ).wait()
        if not resp.ok():
            return []
        blocks = (resp.response_data or {}).get("metrics", []) or []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if str(b.get("task") or "") != str(task_id):
                continue
            metrics = b.get("metrics", []) or []
            return [str(m) for m in metrics if str(m).strip()]
    except Exception:
        return []
    return []


def _validate_metrics(plot_metrics: Sequence[str], debug_image_metrics: Sequence[str]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    plot_metrics = [str(m) for m in (plot_metrics or []) if str(m).strip()]
    debug_image_metrics = [str(m) for m in (debug_image_metrics or []) if str(m).strip()]

    interp = [m for m in plot_metrics if m.startswith(EXPECTED_PLOT_METRICS_PREFIX)]
    other_plots = [m for m in plot_metrics if not m.startswith(EXPECTED_PLOT_METRICS_PREFIX)]

    if len(interp) != 1:
        errors.append(
            f"Expected exactly 1 interpolation plot metric ('{EXPECTED_PLOT_METRICS_PREFIX}*'), got {len(interp)}: {interp}"
        )

    other_set = set(other_plots)
    if other_set != EXPECTED_PLOT_METRICS_EXACT:
        missing = sorted(EXPECTED_PLOT_METRICS_EXACT - other_set)
        extra = sorted(other_set - EXPECTED_PLOT_METRICS_EXACT)
        if missing:
            errors.append(f"Missing expected plot metrics: {missing}")
        if extra:
            errors.append(f"Unexpected plot metrics found: {extra}")

    if debug_image_metrics:
        errors.append(f"Unexpected debug-image metrics found (should be none): {sorted(set(debug_image_metrics))}")

    expected_total = len(EXPECTED_PLOT_METRICS_EXACT) + 1
    if len(set(plot_metrics)) != expected_total:
        errors.append(
            f"Expected exactly {expected_total} unique plot metrics, got {len(set(plot_metrics))}: {sorted(set(plot_metrics))}"
        )

    return (len(errors) == 0), errors


def _validate_hyperparams(sections: Sequence[str]) -> Tuple[bool, List[str]]:
    got = {str(s).strip() for s in (sections or []) if str(s).strip()}
    missing = sorted(EXPECTED_HYPERPARAM_SECTIONS - got)
    if missing:
        return False, [f"Missing expected HyperParameters sections: {missing}"]
    return True, []


def _wait_for_expected_training_summary_outputs(
    task: Any,
    *,
    timeout_seconds: float,
    poll_seconds: float,
) -> Tuple[List[str], List[str]]:
    deadline = time.time() + float(timeout_seconds)
    last_plot_metrics: List[str] = []
    last_hyper_sections: List[str] = []
    while time.time() < deadline:
        try:
            task.reload()
        except Exception:
            pass

        last_plot_metrics = _unique_plot_metrics(task)
        last_hyper_sections = _hyperparam_sections(task)
        ok_plots, _ = _validate_metrics(last_plot_metrics, [])
        ok_hp, _ = _validate_hyperparams(last_hyper_sections)
        if ok_plots and ok_hp:
            return last_plot_metrics, last_hyper_sections

        st = _task_status(task)
        if st in {"failed", "aborted", "stopped"}:
            raise RuntimeError(
                f"training-summary task reached terminal status '{st}' before expected plots appeared. "
                f"Last plots={sorted(set(last_plot_metrics))} hyperparams={last_hyper_sections}"
            )
        time.sleep(float(poll_seconds))

    raise RuntimeError(
        "Timed out waiting for training-summary plots/hyperparams to match expectations. "
        f"Last plots={sorted(set(last_plot_metrics))} hyperparams={last_hyper_sections}"
    )


def _run_pipeline(config_path: Path, *, run_id: str) -> Dict[str, Any]:
    from automl_lib.pipeline.controller import run_pipeline

    info = run_pipeline(config_path, mode="clearml")
    if not isinstance(info, dict):
        return {"run_id": run_id, "info": str(info)}
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a new ClearML pipeline and verify training-summary plots are exactly 01–08 only."
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to training config YAML.")
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="If set, skip running the pipeline and only verify plots for the specified run_id.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0, help="Timeout waiting for tasks/plots.")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval.")
    args = parser.parse_args()

    if os.environ.get("CLEARML_OFFLINE_MODE"):
        raise SystemExit("CLEARML_OFFLINE_MODE is set; disable it to verify against a real ClearML server.")

    try:
        from automl_lib.integrations.clearml.bootstrap import ensure_clearml_config_file
        from automl_lib.integrations.clearml.context import generate_run_id, set_run_id_env
    except Exception as exc:
        raise SystemExit(f"Failed importing automl_lib. Run from repo root. Reason: {exc}") from exc

    ensure_clearml_config_file()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    run_id = str(args.run_id).strip() if args.run_id else generate_run_id()
    if not args.run_id:
        set_run_id_env(run_id)
        # Prevent accidental reuse when invoked from within an existing ClearML task.
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""
        print(f"[verify] Starting pipeline for run_id={run_id} config={config_path}")
        info = _run_pipeline(config_path, run_id=run_id)
        print("[verify] Pipeline finished:")
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        run_id = str(info.get("run_id") or run_id)
    else:
        print(f"[verify] Verifying existing run_id={run_id}")

    summary_task = _wait_for_training_summary_task(
        run_id=run_id,
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )

    plot_metrics, hyper_sections = _wait_for_expected_training_summary_outputs(
        summary_task,
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )
    debug_image_metrics = _get_training_debug_image_metrics(str(getattr(summary_task, "id", "") or ""))
    ok, errors = _validate_metrics(plot_metrics, debug_image_metrics)
    ok_hp, hp_errors = _validate_hyperparams(hyper_sections)
    ok = bool(ok and ok_hp)
    errors.extend(hp_errors)

    print(f"[verify] training-summary task_id={getattr(summary_task, 'id', '')} name={_task_name(summary_task)}")
    try:
        print(f"[verify] url={summary_task.get_output_log_web_page()}")
    except Exception:
        pass
    print("[verify] HyperParameters sections:")
    for sec in hyper_sections:
        print(f"- {sec}")
    print("[verify] Plot metrics (tables/plotly):")
    for t in plot_metrics:
        print(f"- {t}")
    print("[verify] Debug-image metrics (reported images):")
    for t in debug_image_metrics:
        print(f"- {t}")

    if not ok:
        print("[verify] FAILED")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(2)
    print("[verify] OK: training-summary plots are restricted to 01–08 only.")


if __name__ == "__main__":
    main()
