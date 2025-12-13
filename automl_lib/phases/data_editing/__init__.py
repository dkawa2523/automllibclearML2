"""
data_editing phase package
--------------------------
既存Dataset/CSVを入力として編集し、ClearML Dataset として登録する（重複はhashで再利用）。
"""

from .processing import run_data_editing_processing

__all__ = ["run_data_editing_processing", "run_data_editing"]


def run_data_editing(config_path, input_info=None, parent_task_id=None):
    return run_data_editing_processing(config_path, input_info=input_info, parent_task_id=parent_task_id)

