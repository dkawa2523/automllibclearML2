"""
data_registration phase package
-------------------------------
CSV を ClearML Dataset として登録し、重複（hashタグ）を検知して再利用する。
"""

from .processing import run_data_registration_processing

__all__ = ["run_data_registration_processing", "run_data_registration"]


def run_data_registration(config_path, parent_task_id=None):
    return run_data_registration_processing(config_path, parent_task_id=parent_task_id)

