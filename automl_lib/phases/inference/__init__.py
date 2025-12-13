"""
inference phase package
-----------------------
処理・可視化・メタ情報・ClearML連携を分離するためのスタブ。
"""

from .processing import run_inference_processing
from .visualization import render_inference_visuals
from .meta import build_inference_metadata
from .clearml_integration import (
    create_inference_summary_task,
    create_inference_child_task,
    finalize_inference_task,
)

__all__ = [
    "run_inference_processing",
    "render_inference_visuals",
    "build_inference_metadata",
    "create_inference_summary_task",
    "create_inference_child_task",
    "finalize_inference_task",
]
