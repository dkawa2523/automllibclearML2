"""
フェーズ実行のラッパ。現状は既存 auto_ml の処理を呼び出しつつ、
将来的に automl_lib 内で完結させることを目指す。

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
    "run_comparison",
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


def run_training(config_path, input_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    from .training.processing import run_training_processing

    return run_training_processing(config_path, input_info=input_info)


def run_inference(config_path) -> Dict[str, Any]:
    from .inference.processing import run_inference_processing

    return run_inference_processing(config_path)


def run_comparison(
    config_path,
    training_info: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
    parent_task_id: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    from .comparison.processing import run_comparison_processing

    return run_comparison_processing(config_path, training_info=training_info, parent_task_id=parent_task_id)
