"""
workflow package
----------------
ワークフロー（フェーズ）実行の入口を提供する。

NOTE:
- import 時点で各フェーズの重い依存関係（sklearn/clearml等）を読み込まないよう、
  ここでは lazy import の薄い関数だけを公開する。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

__all__ = [
    "run_data_registration",
    "run_data_editing",
    "run_preprocessing",
    "run_training",
    "run_inference",
]


def run_data_registration(config_path, parent_task_id: Optional[str] = None) -> Dict[str, Any]:
    from .data_registration.processing import run_data_registration_processing

    return run_data_registration_processing(config_path, parent_task_id=parent_task_id)


def run_data_editing(
    config_path,
    input_info: Optional[Dict[str, Any]] = None,
    parent_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    from .data_editing.processing import run_data_editing_processing

    return run_data_editing_processing(config_path, input_info=input_info, parent_task_id=parent_task_id)


def run_preprocessing(config_path, input_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from .preprocessing.processing import run_preprocessing_processing

    return run_preprocessing_processing(config_path, input_info=input_info)


def run_training(
    config_path,
    input_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from .training.processing import run_training_processing

    return run_training_processing(config_path, input_info=input_info)


def run_inference(config_path) -> Dict[str, Any]:
    from .inference.processing import run_inference_processing

    return run_inference_processing(config_path)


#
# comparison phase removed (intentionally)
#
