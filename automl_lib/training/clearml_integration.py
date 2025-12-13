"""ClearML integration helpers for the AutoML pipeline.

This module keeps ClearML optional: if the ``clearml`` package is not
installed or configuration disables integration, all helpers degrade to
no‑ops so the core pipeline can run unchanged. When enabled, it provides
simple utilities to:

* initialise ClearML tasks for training/inference/data processing
* register datasets (raw or derived) and log basic metadata
* upload artifacts such as result CSV/plots
* register trained models with ClearML's model registry
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from automl_lib.config.schemas import ClearMLAgentsConfig, ClearMLSettings


def _import_clearml():
    try:  # pragma: no cover - optional dependency
        from automl_lib.clearml.bootstrap import ensure_clearml_config_file

        ensure_clearml_config_file()
        from clearml import Dataset, OutputModel, Task, TaskTypes

        return Dataset, OutputModel, Task, TaskTypes
    except Exception:
        return None, None, None, None


def _map_task_type(task_type: str, task_types_cls) -> Any:
    mapping = {
        "training": getattr(task_types_cls, "training", None),
        "inference": getattr(task_types_cls, "inference", None),
        "optimization": getattr(task_types_cls, "optimizer", None) or getattr(task_types_cls, "optimization", None),
        "data_processing": getattr(task_types_cls, "data_processing", None),
        "pipeline": getattr(task_types_cls, "controller", None),
    }
    return mapping.get(task_type.lower(), getattr(task_types_cls, "custom", None))


def build_clearml_config_from_dict(data: Optional[Dict[str, Any]]) -> Optional[ClearMLSettings]:
    """Convert a loose dict (e.g., from YAML) into ClearMLSettings."""

    if not data:
        return None
    if hasattr(data, "model_dump"):
        data = data.model_dump()  # type: ignore[assignment]
    elif hasattr(data, "dict"):
        data = data.dict()  # type: ignore[assignment]
    if not isinstance(data, dict):
        return None
    try:
        return ClearMLSettings.model_validate(data)
    except Exception:
        return None


def _to_serialisable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()  # type: ignore[attr-defined]
        except Exception:
            pass
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


class ClearMLManager:
    """Lightweight ClearML helper that no-ops when ClearML is unavailable."""

    def __init__(
        self,
        cfg: Optional[ClearMLSettings],
        task_name: str,
        task_type: str,
        default_project: str = "AutoML",
        parent: Optional[str] = None,
        existing_task: Any = None,
    ) -> None:
        if parent is None:
            parent = os.environ.get("AUTO_ML_PARENT_TASK_ID")
        self.cfg = cfg
        self.enabled = bool(cfg and cfg.enabled)
        self.task = None
        self.logger = None
        self.output_uri = cfg.base_output_uri if cfg else None
        if existing_task is not None:
            self.task = existing_task
            try:
                self.logger = existing_task.get_logger()
            except Exception:
                self.logger = None
            self.enabled = True
            return
        if not self.enabled:
            return
        # Avoid accidental reuse of a previous task id
        os.environ.pop("CLEARML_TASK_ID", None)
        os.environ["CLEARML_TASK_ID"] = ""
        Dataset, OutputModel, Task, TaskTypes = _import_clearml()
        if Task is None:
            print("ClearML is not installed; skipping ClearML logging.")
            self.enabled = False
            return
        try:
            # In a ClearML PipelineController step, the controller manages the
            # step task lifecycle. Closing/resetting it here breaks artifacts
            # hand-off between steps. Only detach previous tasks in normal runs.
            if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
                current = Task.current_task()
                if current:
                    try:
                        current.close()
                    except Exception:
                        pass
                    try:
                        Task.current_task(None)
                    except Exception:
                        pass
        except Exception:
            pass
        project = (cfg.project_name or default_project) if cfg else default_project
        tags = list(cfg.tags) if cfg and cfg.tags else []
        # Reduce overhead / avoid hangs from repo & resource scanning
        os.environ.setdefault("CLEARML_SKIP_PIP_FREEZE", "1")
        try:
            self.task = Task.init(
                project_name=project,
                task_name=task_name,
                task_type=_map_task_type(task_type, TaskTypes),
                reuse_last_task_id=False,
                auto_connect_frameworks=False,
                auto_resource_monitoring=False,
                auto_connect_arg_parser=False,
            )
            if parent:
                try:
                    self.task.add_parent(parent)
                except Exception:
                    try:
                        self.task.set_parent(parent)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            if tags:
                self.task.add_tags(tags)
            self.logger = self.task.get_logger()
            # Stash queue preference for visibility; do not force remote execution here.
            if cfg.queue and not cfg.run_tasks_locally:
                self.task.set_parameter("requested_queue", cfg.queue)
        except Exception as exc:  # pragma: no cover - optional dependency path
            print(f"ClearML task initialisation failed; continuing without ClearML. Reason: {exc}")
            self.enabled = False
            self.task = None
            self.logger = None

    def connect_configuration(self, obj: Any, name: str = "config") -> None:
        if not self.task:
            return
        try:
            payload = _to_serialisable(obj)
            self.task.connect_configuration(name=name, configuration=payload)
        except Exception:
            try:
                self.task.connect(payload)  # type: ignore[arg-type]
            except Exception:
                pass

    def connect_params(self, obj: Any) -> None:
        """Register params into ClearML Hyperparameters (preferred over connect_configuration)."""

        if not self.task:
            return
        try:
            self.task.connect(_to_serialisable(obj))
        except Exception:
            pass

    def report_table(self, title: str, df: pd.DataFrame, series: str = "table", iteration: int = 0) -> None:
        if not self.logger:
            return
        try:
            self.logger.report_table(title=title, series=series, iteration=iteration, table_plot=df)
        except Exception:
            pass

    def report_scalar(self, title: str, series: str, value: float, iteration: int = 0) -> None:
        if not self.logger:
            return
        try:
            self.logger.report_scalar(title=title, series=series, value=float(value), iteration=iteration)
        except Exception:
            pass

    def upload_artifact(self, name: str, path: Path) -> None:
        if not self.task or not path.exists():
            return
        try:
            self.task.upload_artifact(name=name, artifact_object=str(path))
        except Exception:
            try:
                self.task.upload_artifact(name=name, artifact_object=path)
            except Exception:
                pass

    def upload_artifacts(self, paths: Iterable[Path]) -> None:
        for path in paths:
            self.upload_artifact(path.name, path)

    def register_output_model(self, model_path: Path, name: str) -> Optional[str]:
        if not model_path.exists() or not self.enabled:
            return None
        Dataset, OutputModel, Task, TaskTypes = _import_clearml()
        if OutputModel is None or self.task is None:
            return None
        try:
            model = OutputModel(task=self.task, name=name, framework="joblib")
            model.update_weights(str(model_path))
            model.update_design()  # ensure metadata persisted
            return model.id
        except Exception:
            return None

    def register_dataset_from_path(
        self,
        name: str,
        path: Path,
        dataset_project: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not self.enabled or not path.exists():
            return None
        Dataset, OutputModel, Task, TaskTypes = _import_clearml()
        if Dataset is None:
            return None
        try:
            ds = Dataset.create(
                dataset_name=name,
                dataset_project=dataset_project or (self.cfg.dataset_project if self.cfg else None) or "datasets",
                use_current_task=False,
                parent_datasets=parent_ids or None,
            )
            ds.add_files(str(path))
            ds.upload(output_url=self.output_uri)
            combined_tags = []
            if self.cfg and self.cfg.tags:
                combined_tags.extend(self.cfg.tags)
            if tags:
                combined_tags.extend(tags)
            if combined_tags:
                ds.add_tags(combined_tags)
            ds.finalize()
            return ds.id
        except Exception as exc:
            print(f"Warning: ClearML dataset registration failed for {name}: {exc}")
            return None

    def register_dataframe_dataset(
        self,
        name: str,
        df: pd.DataFrame,
        output_dir: Path,
        filename: str,
        dataset_project: Optional[str] = None,
        parent_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / filename
        try:
            df.to_csv(csv_path, index=False)
        except Exception:
            return None
        return self.register_dataset_from_path(
            name=name,
            path=csv_path,
            dataset_project=dataset_project,
            parent_ids=parent_ids,
            tags=tags,
        )

    def log_dataset_overview(self, df: pd.DataFrame, name: str, source: Optional[str] = None) -> None:
        if not self.logger:
            return
        meta = {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "source": source or "",
            "preview_columns": list(df.columns[: min(10, len(df.columns))]),
        }
        try:
            self.logger.report_table(
                title=f"{name} preview",
                series="dataset_head",
                iteration=0,
                table_plot=df.head(50),
            )
        except Exception:
            pass
        for key, value in meta.items():
            try:
                self.logger.report_single_value(f"{name}/{key}", value)
            except Exception:
                try:
                    self.logger.report_scalar(title=name, series=key, value=value, iteration=0)
                except Exception:
                    pass

    def close(self) -> None:
        if self.task:
            try:
                # Force completion if still running
                status = getattr(self.task, "get_status", lambda: None)()
                if status and str(status).lower() not in {"completed", "closed"}:
                    try:
                        self.task.mark_completed()
                    except Exception:
                        pass
                self.task.close()
            except Exception:
                pass


def ensure_local_dataset_copy(dataset_id: str, target_dir: Optional[Path] = None) -> Optional[Path]:
    """Download a ClearML dataset by ID and return the local path."""

    Dataset, _, _, _ = _import_clearml()
    if Dataset is None:
        return None
    try:
        dataset = Dataset.get(dataset_id=dataset_id)
        kwargs: Dict[str, Any] = {}
        if target_dir:
            # Some ClearML versions support selecting a target folder for the local copy.
            # Others only support using the internal cache path.
            import inspect

            params = inspect.signature(dataset.get_local_copy).parameters
            if "target_dir" in params:
                kwargs["target_dir"] = str(target_dir)
            elif "target_folder" in params:
                kwargs["target_folder"] = str(target_dir)
        local = dataset.get_local_copy(**kwargs)
        return Path(local)
    except Exception:
        return None


def find_first_csv(path: Path) -> Optional[Path]:
    """Return the first CSV file under path (including path itself if file)."""

    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if path.is_dir():
        for candidate in path.rglob("*.csv"):
            return candidate
    return None


def dataframe_from_dataset(dataset_id: str, target_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Fetch a ClearML dataset and load the first CSV into a DataFrame."""

    local_path = ensure_local_dataset_copy(dataset_id, target_dir=target_dir)
    if not local_path:
        return None
    csv_path = find_first_csv(local_path)
    if not csv_path:
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None
