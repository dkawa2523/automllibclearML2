"""Compatibility wrapper for ClearML utilities used by training/inference.

Historically `automl_lib.training.clearml_integration` hosted shared ClearML helpers.
Those utilities are now centralized under `automl_lib.integrations.clearml`.

New code should import from `automl_lib.integrations.clearml` directly.
"""

from __future__ import annotations

from automl_lib.integrations.clearml.datasets import dataframe_from_dataset, ensure_local_dataset_copy, find_first_csv
from automl_lib.integrations.clearml.manager import (  # noqa: F401
    ClearMLManager,
    build_clearml_config_from_dict,
    _import_clearml,
)

__all__ = [
    "ClearMLManager",
    "build_clearml_config_from_dict",
    "_import_clearml",
    "ensure_local_dataset_copy",
    "find_first_csv",
    "dataframe_from_dataset",
]

