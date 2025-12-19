"""
Common logging helpers for ClearML.
-----------------------------------
Hyperparameters登録、Artifacts/Plots/DebugSamplesの報告を一箇所にまとめる。
今後、各フェーズの clearml_integration から呼び出す想定。
"""

from pathlib import Path
from dataclasses import asdict, is_dataclass
import json
from typing import Any, Dict, Iterable, Mapping

from .bootstrap import ensure_clearml_config_file

ensure_clearml_config_file()


def report_hyperparams(task, params: Dict[str, Any]) -> None:
    """Register hyperparameters (dict) into ClearML task."""
    if not task:
        return
    try:
        task.connect(params)
    except Exception:
        pass


def _to_serialisable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass
    if isinstance(obj, Mapping):
        return dict(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def report_hyperparams_sections(task, sections: Dict[str, Dict[str, Any]]) -> None:
    """Register hyperparameters into ClearML task using multiple named sections."""

    if not task:
        return
    if not isinstance(sections, dict):
        return

    for section_name, params in sections.items():
        name = str(section_name).strip()
        if not name:
            continue
        if params is None:
            continue
        payload = _to_serialisable(params)
        if isinstance(payload, dict) and not payload:
            continue
        try:
            task.connect(payload, name=name)
        except TypeError:
            try:
                task.connect(payload)
            except Exception:
                pass
        except Exception:
            pass


def report_table(logger, title: str, df, series: str = "table", iteration: int = 0) -> None:
    if not logger:
        return
    try:
        logger.report_table(title=title, series=series, iteration=iteration, table_plot=df)
    except Exception:
        pass


def report_scalar(logger, title: str, series: str, value: float, iteration: int = 0) -> None:
    if not logger:
        return
    try:
        logger.report_scalar(title=title, series=series, value=float(value), iteration=iteration)
    except Exception:
        pass


def report_plotly(logger, title: str, series: str, figure: Any, iteration: int = 0) -> None:
    if not logger:
        return
    try:
        logger.report_plotly(title=title, series=series, iteration=iteration, figure=figure)
    except Exception:
        pass


def report_image(logger, title: str, series: str, local_path: str, iteration: int = 0) -> None:
    if not logger:
        return
    try:
        logger.report_image(title=title, series=series, local_path=local_path, iteration=iteration)
    except Exception:
        pass


def upload_artifacts(task, paths: Iterable[Path]) -> None:
    if not task:
        return
    for p in paths:
        if p.exists():
            try:
                task.upload_artifact(name=p.name, artifact_object=str(p))
            except Exception:
                try:
                    task.upload_artifact(name=p.name, artifact_object=p)
                except Exception:
                    pass
