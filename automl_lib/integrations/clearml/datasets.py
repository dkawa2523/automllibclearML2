from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .bootstrap import ensure_clearml_config_file

ensure_clearml_config_file()


def _import_dataset():
    try:  # pragma: no cover - optional dependency
        from clearml import Dataset  # type: ignore

        return Dataset
    except Exception:
        return None


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tag_for_path(path: Path) -> str:
    return f"hash:{file_md5(path)}"


def find_first_csv(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    if path.is_dir():
        # Prefer known "processed" filenames (preprocessing contract v1).
        # ClearML Dataset local copies may include parent dataset files; we want the
        # processed table when it exists to avoid picking a raw parent CSV by accident.
        preferred_names = [
            "data_processed.csv",
            # legacy / training-side generated dataset name (older versions)
            "preprocessed_features.csv",
        ]
        for name in preferred_names:
            try:
                direct = path / name
                if direct.exists():
                    return direct
            except Exception:
                pass
            try:
                matches = list(path.rglob(name))
                if matches:
                    matches.sort(key=lambda p: (len(p.parts), str(p)))
                    return matches[0]
            except Exception:
                pass

        try:
            candidates = list(path.rglob("*.csv"))
            if not candidates:
                return None
            candidates.sort(key=lambda p: (len(p.parts), str(p)))
            return candidates[0]
        except Exception:
            return None
    return None


def ensure_local_dataset_copy(dataset_id: str, target_dir: Optional[Path] = None) -> Optional[Path]:
    Dataset = _import_dataset()
    if Dataset is None:
        return None
    try:
        ds = Dataset.get(dataset_id=dataset_id)
        kwargs: dict[str, Any] = {}
        if target_dir:
            params = inspect.signature(ds.get_local_copy).parameters
            if "target_dir" in params:
                kwargs["target_dir"] = str(target_dir)
            elif "target_folder" in params:
                kwargs["target_folder"] = str(target_dir)
        local = ds.get_local_copy(**kwargs)
        return Path(local)
    except Exception:
        return None


def dataframe_from_dataset(dataset_id: str, target_dir: Optional[Path] = None):
    local = ensure_local_dataset_copy(dataset_id, target_dir=target_dir)
    if not local:
        return None
    csv_path = find_first_csv(local)
    if not csv_path:
        return None
    try:
        import pandas as pd

        return pd.read_csv(csv_path)
    except Exception:
        return None


def find_first_dataset_id_by_tag(tag: str, dataset_project: Optional[str]) -> Optional[str]:
    Dataset = _import_dataset()
    if Dataset is None:
        return None
    try:
        matches = Dataset.list_datasets(dataset_project=dataset_project, tags=[tag])
        if matches:
            return matches[0].get("id")
    except Exception:
        return None
    return None


def add_tags_to_dataset(dataset_id: str, tags: Iterable[str]) -> None:
    Dataset = _import_dataset()
    if Dataset is None:
        return
    tag_list = [str(t) for t in tags if str(t).strip()]
    if not tag_list:
        return
    try:
        ds = Dataset.get(dataset_id=str(dataset_id))
        ds.add_tags(tag_list)
    except Exception:
        return


def register_dataset_from_path(
    *,
    name: str,
    path: Path,
    dataset_project: Optional[str],
    parent_ids: Optional[list[str]] = None,
    tags: Optional[Iterable[str]] = None,
    output_uri: Optional[str] = None,
    hyperparams_sections: Optional[dict[str, Any]] = None,
    configuration_objects: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    Dataset = _import_dataset()
    if Dataset is None or not path.exists():
        return None
    try:
        try:
            ds = Dataset.create(
                dataset_name=name,
                dataset_project=dataset_project or "datasets",
                use_current_task=False,
                parent_datasets=parent_ids or None,
            )
        except TypeError:
            ds = Dataset.create(
                dataset_name=name,
                dataset_project=dataset_project or "datasets",
                parent_datasets=parent_ids or None,
            )
        task = None
        try:
            task = getattr(ds, "_task", None)
        except Exception:
            task = None

        if task is not None:
            if hyperparams_sections:
                try:
                    if isinstance(hyperparams_sections, dict):
                        for section_name, payload in hyperparams_sections.items():
                            section = str(section_name).strip()
                            if not section or payload is None:
                                continue
                            if hasattr(payload, "model_dump"):
                                try:
                                    payload = payload.model_dump()  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            try:
                                payload = json.loads(json.dumps(payload, default=str))
                            except Exception:
                                payload = str(payload)
                            try:
                                task.connect(payload, name=section)
                            except TypeError:
                                try:
                                    task.connect(payload)
                                except Exception:
                                    pass
                            except Exception:
                                pass
                except Exception:
                    pass
            if configuration_objects:
                try:
                    if isinstance(configuration_objects, dict):
                        for obj_name, payload in configuration_objects.items():
                            name = str(obj_name).strip()
                            if not name or payload is None:
                                continue
                            if hasattr(payload, "model_dump"):
                                try:
                                    payload = payload.model_dump()  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                            try:
                                payload = json.loads(json.dumps(payload, default=str))
                            except Exception:
                                payload = str(payload)
                            try:
                                task.connect_configuration(configuration=payload, name=name)
                            except Exception:
                                pass
                except Exception:
                    pass
        ds.add_files(str(path))
        try:
            if output_uri:
                ds.upload(output_url=output_uri)
            else:
                ds.upload()
        except TypeError:
            ds.upload()
        if tags:
            try:
                ds.add_tags(list(tags))
            except Exception:
                pass
        ds.finalize()
        return ds.id
    except Exception:
        return None
