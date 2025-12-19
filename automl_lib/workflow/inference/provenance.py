from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return ""


def collect_model_provenance(model_id: Optional[str]) -> Dict[str, Any]:
    """Collect best-effort provenance for a ClearML model_id.

    Goal: show in inference tasks (Configuration Objects) where the selected model came from:
    - training child task
    - training-summary task
    - (if available) dataset / preprocessing ids from training-summary USER PROPERTIES
    """

    out: Dict[str, Any] = {"model_id": _safe_str(model_id).strip()}
    mid = out["model_id"]
    if not mid:
        return out

    try:
        from clearml import InputModel, Task  # type: ignore
    except Exception:
        return out

    training_task_id = ""
    try:
        im = InputModel(model_id=mid)
        # ClearML returns task ids as strings for these properties.
        training_task_id = _safe_str(getattr(im, "original_task", "") or getattr(im, "task", "")).strip()
        out["training_task_id"] = training_task_id
        try:
            out["model_name"] = _safe_str(getattr(im, "name", "")).strip()
        except Exception:
            pass
        try:
            out["model_project"] = _safe_str(getattr(im, "project", "")).strip()
        except Exception:
            pass
        try:
            tags = getattr(im, "tags", None)
            if tags is not None:
                out["model_tags"] = list(tags)  # type: ignore[arg-type]
        except Exception:
            pass
    except Exception:
        training_task_id = ""

    if training_task_id:
        try:
            t = Task.get_task(task_id=training_task_id)
            out["training_task_name"] = _safe_str(getattr(t, "name", "")).strip()
            try:
                out["training_task_url"] = _safe_str(t.get_output_log_web_page())  # type: ignore[attr-defined]
            except Exception:
                pass
            parent_id = _safe_str(getattr(t, "parent", "")).strip()
            if parent_id:
                out["training_summary_task_id"] = parent_id
                p = None
                try:
                    p = Task.get_task(task_id=parent_id)
                except Exception:
                    p = None
                if p is not None:
                    out["training_summary_task_name"] = _safe_str(getattr(p, "name", "")).strip()
                    try:
                        out["training_summary_url"] = _safe_str(p.get_output_log_web_page())  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    try:
                        props = p.get_user_properties(value_only=True)  # type: ignore[attr-defined]
                        if isinstance(props, dict):
                            # Keep only a small, user-facing subset.
                            keep = {}
                            for k in [
                                "run_id",
                                "dataset_id",
                                "dataset_key",
                                "dataset_role",
                                "source_dataset_id",
                                "selected_preprocessor",
                                "recommended_model_id",
                                "recommended_model_task_id",
                            ]:
                                v = props.get(k)
                                if v is None:
                                    continue
                                keep[str(k)] = _safe_str(v)
                            # Include preproc-* hints if present (stable keys).
                            for k, v in props.items():
                                if not isinstance(k, str):
                                    continue
                                if not k.startswith("preproc_"):
                                    continue
                                keep[k] = _safe_str(v)
                            out["training_summary_user_properties"] = keep
                    except Exception:
                        pass
        except Exception:
            pass

    return out


def resolve_dataset_id_for_range(
    *,
    input_info: Optional[Dict[str, Any]],
    input_conf: Dict[str, Any],
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Pick a dataset_id to use for training-range visualization (best-effort)."""

    # Explicit inference config wins (batch mode).
    cand = _safe_str(input_conf.get("dataset_id") or "").strip()
    if cand:
        return cand

    # Pipeline hand-off (training output).
    cand = _safe_str((input_info or {}).get("dataset_id") or "").strip()
    if cand:
        return cand

    # From training-summary user properties (via model provenance).
    props = (provenance or {}).get("training_summary_user_properties")
    if isinstance(props, dict):
        cand = _safe_str(props.get("dataset_id") or "").strip()
        if cand:
            return cand
        cand = _safe_str(props.get("source_dataset_id") or "").strip()
        if cand:
            return cand

    return None
