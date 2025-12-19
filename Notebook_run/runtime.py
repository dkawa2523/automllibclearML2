from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "automl_lib").exists():
        return cwd
    for p in [cwd, *cwd.parents]:
        if (p / "automl_lib").exists():
            return p
    return cwd


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class NotebookContext:
    repo_root: Path
    state_path: Path
    config_out_dir: Path

    def load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def update_state(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.load_state()
        merged = _deep_update(cur, patch)
        self.save_state(merged)
        return merged

    def load_yaml(self, path: Path) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyyaml が必要です: pip install pyyaml") from exc

        obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError(f"YAML must be a dict: {path}")
        return obj

    def dump_yaml(self, data: Dict[str, Any]) -> str:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyyaml が必要です: pip install pyyaml") from exc
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    def deep_update(self, base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        return _deep_update(base, patch)

    def write_yaml(self, path: Path, data: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dump_yaml(data), encoding="utf-8")
        return path


def bootstrap(
    *,
    repo_root: Optional[Path] = None,
    state_path: Optional[Path] = None,
    config_out_dir: Optional[Path] = None,
) -> NotebookContext:
    root = Path(repo_root).resolve() if repo_root else _find_repo_root()
    os.chdir(root)

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Keep these under outputs/ (gitignored) by default.
    state = Path(state_path) if state_path else Path("outputs/notebook_state.json")
    cfg_dir = Path(config_out_dir) if config_out_dir else Path("outputs/notebook_configs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return NotebookContext(repo_root=root, state_path=state, config_out_dir=cfg_dir)

