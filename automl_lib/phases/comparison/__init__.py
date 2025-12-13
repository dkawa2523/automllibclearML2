"""
comparison phase package
------------------------
処理・可視化・メタ情報・ClearML連携を分離。
"""

from .processing import run_comparison_processing
from .visualization import render_comparison_visuals
from .meta import build_comparison_metadata
from .clearml_integration import create_comparison_task, finalize_comparison_task

__all__ = [
    "run_comparison_processing",
    "render_comparison_visuals",
    "build_comparison_metadata",
    "create_comparison_task",
    "finalize_comparison_task",
    "run_comparison",
]


def run_comparison(config_path, training_info=None, parent_task_id=None):
    """
    Convenience wrapper to execute comparison processing.
    """
    return run_comparison_processing(config_path, training_info=training_info, parent_task_id=parent_task_id)
