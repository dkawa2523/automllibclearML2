from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def conf_root() -> Path:
    return repo_root() / "conf"


def to_clean_dict(cfg_obj: Any) -> Dict[str, Any]:
    """Convert an OmegaConf/Hydra config object into a plain dict for execution.

    - Resolves interpolations
    - Drops Hydra-internal keys (e.g. `hydra`)
    """

    try:  # pragma: no cover - optional dependency
        from omegaconf import OmegaConf  # type: ignore

        data = OmegaConf.to_container(cfg_obj, resolve=True)
    except Exception:
        data = cfg_obj
    out: Dict[str, Any] = dict(data or {}) if isinstance(data, dict) else {}
    out.pop("hydra", None)
    return out


def write_yaml_config(
    payload: Dict[str, Any],
    *,
    prefix: str,
    out_dir: Optional[Path] = None,
) -> Path:
    """Write a YAML config file (OmegaConf when available) and return the path."""

    out_dir = Path(out_dir) if out_dir else repo_root() / "outputs" / "hydra_configs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    pid = os.getpid()
    path = out_dir / f"{prefix}_{ts}_{pid}.yaml"

    try:  # pragma: no cover - optional dependency
        from omegaconf import OmegaConf  # type: ignore

        OmegaConf.save(config=OmegaConf.create(payload), f=str(path))
        return path
    except Exception:
        import yaml  # type: ignore

        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

