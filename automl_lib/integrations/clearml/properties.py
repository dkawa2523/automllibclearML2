from __future__ import annotations

from typing import Any, Dict, Optional


def set_user_properties(task: Any, properties: Dict[str, Any], *, prefix: Optional[str] = None) -> None:
    """Best-effort set ClearML USER PROPERTIES.

    ClearML shows these under Configuration -> USER PROPERTIES and they are editable/searchable.
    This should contain identifiers and small summaries (not large configs).
    """

    if not task or not isinstance(properties, dict):
        return

    payload: Dict[str, str] = {}
    for key, value in properties.items():
        if value is None:
            continue
        name = str(key).strip()
        if not name:
            continue
        if prefix:
            name = f"{prefix}.{name}"
        try:
            text = str(value).strip()
        except Exception:
            text = ""
        if not text:
            continue
        # Keep values reasonably small for UI readability.
        if len(text) > 500:
            text = text[:500] + "..."
        payload[name] = text

    if not payload:
        return

    try:
        # ClearML expects user properties as keyword args (not a single dict positional arg).
        task.set_user_properties(**payload)
        return
    except TypeError:
        # Best-effort backward compatibility (some versions may accept a mapping).
        try:
            task.set_user_properties(payload)
        except Exception:
            return
    except Exception:
        return
