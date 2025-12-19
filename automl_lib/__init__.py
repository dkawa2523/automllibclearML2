"""
automl_lib
-----------
新しい階層で各フェーズの処理・設定・ClearML連携を統一的に管理するためのパッケージ。
当面は既存の `auto_ml` 実装をラップしつつ、段階的にこちらへ機能を移行する。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path


def _ensure_writable_cache_dirs() -> None:
    """
    Best-effort: make matplotlib/fontconfig caches writable even in sandboxed environments.

    - MPLCONFIGDIR: matplotlib cache/config dir
    - XDG_CACHE_HOME: used by fontconfig and other libraries
    """

    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    try:
        outputs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    if not os.environ.get("MPLCONFIGDIR"):
        mpl_dir = outputs_dir / ".matplotlib"
        try:
            mpl_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(mpl_dir)
        except Exception:
            pass

    if not os.environ.get("XDG_CACHE_HOME"):
        cache_dir = outputs_dir / ".cache"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["XDG_CACHE_HOME"] = str(cache_dir)
        except Exception:
            pass


def _suppress_noisy_warnings() -> None:
    flag = str(os.environ.get("AUTO_ML_SUPPRESS_WARNINGS", "1")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return
    try:
        warnings.filterwarnings(
            "ignore",
            message=r"Please import `spmatrix` from the `scipy\.sparse` namespace.*",
            category=DeprecationWarning,
        )
    except Exception:
        pass


_ensure_writable_cache_dirs()
_suppress_noisy_warnings()

__all__ = ["config", "clearml", "pipeline", "registry", "workflow"]
