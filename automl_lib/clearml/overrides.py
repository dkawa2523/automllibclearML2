from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union


_PATH_SPLIT_RE = re.compile(r"\.")


def get_task_overrides(*, name: str = "overrides") -> Optional[Dict[str, Any]]:
    """
    Fetch overrides dict from the currently running ClearML task (if any).

    This is a best-effort helper for clone execution mode:
    - `run_clone` stores overrides as a ClearML configuration object named "overrides".
    - Phase runners can apply these overrides before validating configs.
    """

    task_id = None
    try:
        task_id = str((__import__("os").environ.get("CLEARML_TASK_ID") or "")).strip()
    except Exception:
        task_id = None
    if not task_id:
        return None
    try:
        from clearml import Task  # type: ignore
    except Exception:
        return None
    try:
        task = Task.current_task()
    except Exception:
        task = None
    if not task:
        try:
            task = Task.get_task(task_id=task_id)
        except Exception:
            task = None
    if not task:
        return None
    try:
        payload = task.get_configuration_object_as_dict(str(name))
    except Exception:
        return None
    if isinstance(payload, dict):
        return dict(payload)
    return None


def _tokenize_path(path: str) -> List[Union[str, int]]:
    tokens: List[Union[str, int]] = []
    for segment in _PATH_SPLIT_RE.split(str(path).strip()):
        s = segment
        if not s:
            continue
        while s:
            if "[" not in s:
                tokens.append(s)
                break
            head, rest = s.split("[", 1)
            if head:
                tokens.append(head)
            idx_str, rest2 = rest.split("]", 1)
            try:
                tokens.append(int(idx_str))
            except Exception as exc:
                raise ValueError(f"Invalid list index in override path: {path!r}") from exc
            s = rest2
            if s.startswith("."):
                s = s[1:]
    return tokens


def _ensure_list_size(value: list, idx: int) -> None:
    while len(value) <= idx:
        value.append(None)


def _set_by_tokens(root: Any, tokens: List[Union[str, int]], value: Any) -> Any:
    cur = root
    for i, tok in enumerate(tokens):
        last = i == len(tokens) - 1
        nxt = tokens[i + 1] if not last else None

        if isinstance(tok, int):
            if not isinstance(cur, list):
                raise TypeError(f"Expected list at {tokens[:i]!r}, got {type(cur).__name__}")
            _ensure_list_size(cur, tok)
            if last:
                cur[tok] = value
                return root
            if cur[tok] is None:
                cur[tok] = [] if isinstance(nxt, int) else {}
            cur = cur[tok]
            continue

        # tok is str
        if not isinstance(cur, dict):
            raise TypeError(f"Expected dict at {tokens[:i]!r}, got {type(cur).__name__}")
        if last:
            cur[tok] = value
            return root
        if tok not in cur or cur[tok] is None:
            cur[tok] = [] if isinstance(nxt, int) else {}
        cur = cur[tok]
    return root


def _deep_merge(dst: Any, src: Any) -> Any:
    if isinstance(dst, dict) and isinstance(src, dict):
        for k, v in src.items():
            if k in dst:
                dst[k] = _deep_merge(dst[k], v)
            else:
                dst[k] = deepcopy(v)
        return dst
    return deepcopy(src)


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Apply overrides to a config dict (supports dotted paths and list indices)."""

    base: Dict[str, Any] = deepcopy(config or {})
    for key, value in (overrides or {}).items():
        if not isinstance(key, str):
            continue
        k = key.strip()
        if not k:
            continue
        if "." in k or "[" in k:
            tokens = _tokenize_path(k)
            try:
                _set_by_tokens(base, tokens, value)
            except Exception:
                # Best-effort: ignore malformed paths
                continue
            continue
        if isinstance(value, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], value)
        else:
            base[k] = deepcopy(value)
    return base
