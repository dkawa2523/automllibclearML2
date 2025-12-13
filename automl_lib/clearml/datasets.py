from __future__ import annotations

import hashlib
import inspect
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
        for candidate in path.rglob("*.csv"):
            return candidate
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


def register_dataset_from_path(
    *,
    name: str,
    path: Path,
    dataset_project: Optional[str],
    parent_ids: Optional[list[str]] = None,
    tags: Optional[Iterable[str]] = None,
    output_uri: Optional[str] = None,
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

