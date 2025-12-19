"""ClearML manager utilities (optional dependency).

This module centralizes the common ClearML Task lifecycle / logging behavior used across
training and inference. It is intentionally defensive: when ClearML is unavailable or
disabled, all methods become no-ops so local execution can proceed unchanged.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from automl_lib.config.schemas import ClearMLSettings

from .bootstrap import ensure_clearml_config_file

ensure_clearml_config_file()


def _import_clearml():
    try:  # pragma: no cover - optional dependency
        from clearml import Dataset, OutputModel, Task, TaskTypes  # type: ignore

        return Dataset, OutputModel, Task, TaskTypes
    except Exception:
        return None, None, None, None


def _map_task_type(task_type: str, task_types_cls) -> Any:
    mapping = {
        "training": getattr(task_types_cls, "training", None),
        "inference": getattr(task_types_cls, "inference", None),
        "optimization": getattr(task_types_cls, "optimizer", None) or getattr(task_types_cls, "optimization", None),
        "data_processing": getattr(task_types_cls, "data_processing", None),
        "analysis": getattr(task_types_cls, "analysis", None),
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
        project: Optional[str] = None,
        parent: Optional[str] = None,
        existing_task: Any = None,
        extra_tags: Optional[List[str]] = None,
    ) -> None:
        if parent is None:
            parent = os.environ.get("AUTO_ML_PARENT_TASK_ID")
        self.cfg = cfg
        self.enabled = bool(cfg and cfg.enabled)
        self.task = None
        self.logger = None
        self.output_uri = cfg.base_output_uri if cfg else None
        # In ClearML PipelineController function steps, the step task lifecycle is managed by the controller.
        # Closing the current step task from inside the step breaks artifacts hand-off and can confuse the UI.
        self._skip_close = False

        combined_tags: List[str] = []
        if cfg and cfg.tags:
            combined_tags.extend([str(t) for t in cfg.tags if str(t).strip()])
        if extra_tags:
            combined_tags.extend([str(t) for t in extra_tags if str(t).strip()])
        # de-duplicate while preserving order
        seen = set()
        tags: List[str] = []
        for t in combined_tags:
            if t in seen:
                continue
            seen.add(t)
            tags.append(t)

        # If executing inside an existing ClearML task (e.g., cloned/enqueued execution
        # or a PipelineController step), prefer reusing the current task to avoid
        # creating orphan tasks and breaking the step lifecycle.
        if existing_task is None and self.enabled:
            current_task_id = os.environ.get("CLEARML_TASK_ID")
            if current_task_id and str(current_task_id).strip():
                _, _, Task, _ = _import_clearml()
                if Task is not None:
                    try:
                        existing_task = Task.current_task()
                    except Exception:
                        existing_task = None

        if existing_task is not None:
            self.task = existing_task
            try:
                self.logger = existing_task.get_logger()
            except Exception:
                self.logger = None
            self.enabled = True
            try:
                if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
                    step_task_id = str(os.environ.get("CLEARML_TASK_ID") or "").strip()
                    if step_task_id:
                        self._skip_close = str(getattr(self.task, "id", "") or "") == step_task_id
            except Exception:
                self._skip_close = False
            # In PipelineController function steps, the step task is initially created under the controller's
            # project/name. Move it to the intended project to keep per-phase tasks discoverable.
            if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
                try:
                    target_project = project or ((cfg.project_name or default_project) if cfg else default_project)
                except Exception:
                    target_project = project or default_project
                try:
                    if target_project and hasattr(self.task, "move_to_project"):
                        self.task.move_to_project(new_project_name=str(target_project))
                except Exception:
                    pass
            # Tasks created with `Task.create()` start in "draft/created" status.
            # Mark them as started so they appear as active/completed rather than Draft in the UI.
            try:
                st = ""
                try:
                    st = str(getattr(self.task, "get_status", lambda: "")() or "").strip().lower()
                except Exception:
                    st = ""
                if st in {"draft", "created"}:
                    try:
                        self.task.mark_started(force=True)
                    except Exception:
                        try:
                            self.task.started(ignore_errors=True, force=True)  # type: ignore[attr-defined]
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if hasattr(self.task, "set_name") and task_name:
                    self.task.set_name(str(task_name))
            except Exception:
                pass
            if parent:
                try:
                    self.task.add_parent(parent)
                except Exception:
                    try:
                        self.task.set_parent(parent)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            if tags:
                try:
                    self.task.add_tags(tags)
                except Exception:
                    pass
            if cfg and cfg.queue and not cfg.run_tasks_locally:
                try:
                    self.task.set_parameter("requested_queue", cfg.queue)
                except Exception:
                    pass
            return

        if not self.enabled:
            return

        # Avoid accidental reuse of a previous task id (but keep the original
        # task id around so remote/cloned executions do not lose it).
        original_task_id = os.environ.get("CLEARML_TASK_ID")
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

        project_name = project or ((cfg.project_name or default_project) if cfg else default_project)
        # Reduce overhead / avoid hangs from repo & resource scanning
        os.environ.setdefault("CLEARML_SKIP_PIP_FREEZE", "1")
        try:
            self.task = Task.init(
                project_name=project_name,
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
                try:
                    self.task.add_tags(tags)
                except Exception:
                    pass
            self.logger = self.task.get_logger()
            # Stash queue preference for visibility; do not force remote execution here.
            if cfg and cfg.queue and not cfg.run_tasks_locally:
                try:
                    self.task.set_parameter("requested_queue", cfg.queue)
                except Exception:
                    pass
            if original_task_id and str(original_task_id).strip():
                os.environ["CLEARML_TASK_ID"] = str(original_task_id)
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
            # Do not fall back to `task.connect()` here, because that would pollute HyperParameters.
            return

    def connect_params(self, obj: Any) -> None:
        """Register params into ClearML Hyperparameters (preferred over connect_configuration)."""

        if not self.task:
            return
        try:
            self.task.connect(_to_serialisable(obj))
        except Exception:
            pass

    def connect_params_sections(self, sections: Dict[str, Any]) -> None:
        """Register params into ClearML Hyperparameters using multiple named sections."""

        if not self.task:
            return
        if not isinstance(sections, dict):
            return
        for section_name, payload in sections.items():
            name = str(section_name).strip()
            if not name:
                continue
            if payload is None:
                continue
            serialised = _to_serialisable(payload)
            if isinstance(serialised, dict) and not serialised:
                continue
            try:
                self.task.connect(serialised, name=name)
            except TypeError:
                try:
                    self.task.connect(serialised)
                except Exception:
                    pass
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
        _, OutputModel, _, _ = _import_clearml()
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
        Dataset, _, _, _ = _import_clearml()
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
            if bool(getattr(self, "_skip_close", False)):
                # Keep the step task open; only flush to persist artifacts/logs.
                try:
                    if hasattr(self.task, "flush"):
                        self.task.flush(wait_for_uploads=True)
                except Exception:
                    pass
                return
            try:
                status = getattr(self.task, "get_status", lambda: None)()
                st = str(status or "").strip().lower() if status is not None else ""
                if st in {"draft", "created"}:
                    try:
                        self.task.mark_started(force=True)
                    except Exception:
                        try:
                            self.task.started(ignore_errors=True, force=True)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                # Do not override terminal failure states.
                if st in {"failed", "aborted", "stopped"}:
                    pass
                elif st and st not in {"completed", "closed"}:
                    try:
                        self.task.mark_completed(ignore_errors=True, force=True)
                    except Exception:
                        pass
                try:
                    if hasattr(self.task, "flush"):
                        self.task.flush(wait_for_uploads=True)
                except Exception:
                    pass
                self.task.close()
            except Exception:
                pass
