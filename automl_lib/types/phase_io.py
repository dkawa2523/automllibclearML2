from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("pydantic is required: pip install pydantic") from exc


class DatasetInfo(BaseModel):
    """
    Minimal, phase-agnostic handoff structure between phases/CLIs.
    """

    dataset_id: Optional[str] = None
    task_id: Optional[str] = None
    csv_path: Optional[str] = None


class TrainingInfo(BaseModel):
    """
    Training phase output structure (summary).
    """

    dataset_id: Optional[str] = None
    task_id: Optional[str] = None
    training_task_ids: List[str] = Field(default_factory=list)
    metrics: Optional[List[Dict[str, Any]]] = None


class ComparisonInfo(BaseModel):
    """
    Comparison phase output structure.
    """

    task_id: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)


class InferenceInfo(BaseModel):
    """
    Inference phase output structure.
    """

    task_id: Optional[str] = None
    child_task_ids: List[str] = Field(default_factory=list)
    output_dir: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)
    mode: Optional[str] = None
