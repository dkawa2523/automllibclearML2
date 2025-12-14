from __future__ import annotations

import getpass
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

AUTO_ML_RUN_ID_ENV = "AUTO_ML_RUN_ID"

_RUN_ID_PATTERN = re.compile(r"^\\d{8}-\\d{6}-[0-9a-f]{6}$")
_DATASET_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")


def generate_run_id(now: Optional[datetime] = None) -> str:
    """Generate a human-sortable run_id: YYYYMMDD-HHMMSS-<6hex>."""

    ts = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{secrets.token_hex(3)}"


def normalize_run_id(value: str) -> str:
    return str(value).strip()


def resolve_run_id(
    *,
    explicit: Optional[str] = None,
    from_input: Optional[str] = None,
    from_config: Optional[str] = None,
    from_env: Optional[str] = None,
    force_new: bool = False,
) -> str:
    if not force_new:
        for cand in (explicit, from_input, from_config, from_env):
            if cand:
                return normalize_run_id(cand)
    return generate_run_id()


def set_run_id_env(run_id: str) -> None:
    os.environ[AUTO_ML_RUN_ID_ENV] = str(run_id)


def get_run_id_env() -> Optional[str]:
    value = os.environ.get(AUTO_ML_RUN_ID_ENV)
    value = str(value).strip() if value else None
    return value or None


def sanitize_token(value: str, *, max_len: int = 64) -> str:
    """Make a value safe for ClearML tags/names (best-effort)."""

    text = str(value).strip().replace(" ", "_")
    text = re.sub(r"[^0-9A-Za-z._:-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "unknown"
    return text[:max_len]


def resolve_dataset_key(
    *,
    explicit: Optional[str] = None,
    dataset_id: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> str:
    if explicit:
        return sanitize_token(explicit, max_len=64)
    if dataset_id:
        cand = str(dataset_id).strip()
        if _DATASET_ID_PATTERN.fullmatch(cand):
            return cand[:8].lower()
        return sanitize_token(cand, max_len=64)
    if csv_path:
        try:
            stem = Path(str(csv_path)).stem
            if stem:
                return sanitize_token(stem, max_len=64)
        except Exception:
            pass
    return "unknown"


def default_user() -> str:
    try:
        return sanitize_token(getpass.getuser(), max_len=32)
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    user: str
    project_root: str
    dataset_project: str
    dataset_key: str
    tags_base: List[str]


def build_run_context(
    *,
    run_id: str,
    dataset_key: str,
    project_root: Optional[str],
    dataset_project: Optional[str],
    user: Optional[str] = None,
) -> RunContext:
    rid = normalize_run_id(run_id)
    if not _RUN_ID_PATTERN.fullmatch(rid):
        rid = sanitize_token(rid, max_len=64)
    key = sanitize_token(dataset_key, max_len=64)
    usr = sanitize_token(user or default_user(), max_len=32)
    proj = str(project_root or "AutoML")
    ds_proj = str(dataset_project or "datasets")
    return RunContext(
        run_id=rid,
        user=usr,
        project_root=proj,
        dataset_project=ds_proj,
        dataset_key=key,
        tags_base=[f"run:{rid}", f"dataset:{key}"],
    )

