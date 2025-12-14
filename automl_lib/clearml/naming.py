from __future__ import annotations

from typing import Iterable, List, Optional

from .context import RunContext, sanitize_token


def build_project_path(
    ctx: RunContext,
    *,
    project_mode: str = "root",
    suffix: Optional[str] = None,
) -> str:
    """
    project_mode:
      - "root": use ctx.project_root as-is
      - "user_dataset": use "<root>/<user>/<dataset_key>"
    """

    mode = str(project_mode or "root").strip().lower()
    base = ctx.project_root or "AutoML"
    if mode in {"user_dataset", "user-dataset", "user/dataset", "hierarchy"}:
        base = f"{base}/{ctx.user}/{ctx.dataset_key}"
    if suffix:
        base = f"{base}/{suffix}"
    return base


def task_name(phase: str, ctx: RunContext, *, model: Optional[str] = None, preproc: Optional[str] = None) -> str:
    p = str(phase).strip().lower()
    if p in {"data_registration", "data-registration"}:
        return f"data_registration [{ctx.dataset_key}] [{ctx.run_id}]"
    if p in {"data_editing", "data-editing"}:
        return f"data_editing [{ctx.dataset_key}] [{ctx.run_id}]"
    if p in {"preprocessing", "preprocess"}:
        return f"preprocessing [{ctx.dataset_key}] [{ctx.run_id}]"
    if p in {"training_summary", "training-summary", "training"} and not model:
        return f"training-summary [{ctx.dataset_key}] [{ctx.run_id}]"
    if p in {"training_child", "train", "training-child"} or (p in {"training", "training_summary"} and model):
        model_tok = sanitize_token(model or "model", max_len=48)
        parts = ["train"]
        if preproc:
            parts.append(f"[{sanitize_token(preproc, max_len=48)}]")
        parts.append(f"[{model_tok}]")
        parts.append(f"[{ctx.run_id}]")
        return " ".join(parts)
    if p in {"comparison", "compare_results", "compare-results"}:
        return f"comparison [{ctx.dataset_key}] [{ctx.run_id}]"
    if p in {"inference", "inference_summary", "inference-summary"} and not model:
        return f"inference-summary [{ctx.dataset_key}] [{ctx.run_id}]"
    return f"{sanitize_token(p, max_len=48)} [{ctx.dataset_key}] [{ctx.run_id}]"


def dataset_name(kind: str, ctx: RunContext, *, preproc: Optional[str] = None) -> str:
    k = str(kind).strip().lower()
    if k in {"raw", "raw-dataset", "raw_dataset"}:
        return f"raw-dataset [{ctx.dataset_key}] [{ctx.run_id}]"
    if k in {"edited", "edited-dataset", "edited_dataset"}:
        return f"edited-dataset [{ctx.dataset_key}] [{ctx.run_id}]"
    if k in {"preprocessed", "preprocessed-dataset", "preprocessed_dataset"}:
        base = f"preprocessed-dataset [{ctx.dataset_key}]"
        if preproc:
            base += f" [{sanitize_token(preproc, max_len=48)}]"
        return f"{base} [{ctx.run_id}]"
    return f"{sanitize_token(k, max_len=48)} [{ctx.dataset_key}] [{ctx.run_id}]"


def build_tags(
    ctx: RunContext,
    *,
    phase: str,
    model: Optional[str] = None,
    preproc: Optional[str] = None,
    extra: Optional[Iterable[str]] = None,
) -> List[str]:
    tags: List[str] = list(ctx.tags_base)
    phase_tok = sanitize_token(str(phase).strip().lower(), max_len=48)
    tags.append(f"phase:{phase_tok}")
    if preproc:
        tags.append(f"preprocess:{sanitize_token(preproc, max_len=64)}")
    if model:
        tags.append(f"model:{sanitize_token(model, max_len=64)}")
    if extra:
        for t in extra:
            if not t:
                continue
            tags.append(str(t))
    # keep stable order but de-duplicate
    seen = set()
    uniq: List[str] = []
    for t in tags:
        key = str(t)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq

