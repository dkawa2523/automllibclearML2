"""
training phase package
----------------------
処理(analysis)、可視化(visualization)、メタ情報管理(meta)、ClearML連携(clearml_integration)
に分離した構造。現状はスタブで、順次実装を移行していく。
"""

from .processing import run_training_processing
from .visualization import render_training_visuals
from .meta import build_training_metadata
from .clearml_integration import create_training_tasks, create_model_child_task, finalize_training_task

__all__ = [
    "run_training_processing",
    "render_training_visuals",
    "build_training_metadata",
    "create_training_tasks",
    "create_model_child_task",
    "finalize_training_task",
]
