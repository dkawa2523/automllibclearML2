"""
Common logging helpers for ClearML.
-----------------------------------
Hyperparameters登録、Artifacts/Plots/DebugSamplesの報告を一箇所にまとめる。
今後、各フェーズの clearml_integration から呼び出す想定。
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .bootstrap import ensure_clearml_config_file

ensure_clearml_config_file()

try:
    from clearml import Logger  # type: ignore
except Exception:  # pragma: no cover
    Logger = None


def report_hyperparams(task, params: Dict[str, Any]) -> None:
    """Register hyperparameters (dict) into ClearML task."""
    if not task:
        return
    try:
        task.connect(params)
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
