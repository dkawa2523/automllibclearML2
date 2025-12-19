from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from automl_lib.config.schemas import TrainingConfig
from automl_lib.integrations.clearml.datasets import ensure_local_dataset_copy, find_first_csv


@dataclass(frozen=True)
class TrainingDatasetSources:
    dataset_id_for_load: Optional[str]
    local_copy: Optional[Path]
    csv_override: Optional[Path]
    preproc_summary_src: Optional[Path]
    preproc_recipe_src: Optional[Path]
    preproc_schema_src: Optional[Path]
    preproc_manifest_src: Optional[Path]
    preproc_bundle_src: Optional[Path]
    preproc_bundle: Any

    @property
    def has_preproc_contract(self) -> bool:
        return bool(
            self.preproc_schema_src
            or self.preproc_manifest_src
            or self.preproc_summary_src
            or self.preproc_recipe_src
            or self.preproc_bundle_src
        )


def normalize_dataset_id(raw_id: str, cfg: TrainingConfig) -> Optional[str]:
    """If raw_id looks like a name, try resolving by name+project. Otherwise return as-is."""

    if not raw_id:
        return None
    raw_str = str(raw_id)
    try:  # pragma: no cover - optional dependency
        from clearml import Dataset  # type: ignore

        if len(raw_str) == 32:
            return raw_str
        ds_obj = Dataset.get(dataset_name=raw_str, dataset_project=cfg.clearml.dataset_project if cfg.clearml else None)
        if ds_obj:
            return ds_obj.id
    except Exception:
        pass
    return raw_str if raw_str else None


def resolve_training_dataset_sources(
    *,
    cfg: TrainingConfig,
    dataset_id_override: Optional[str],
    base_output_dir: Path,
) -> TrainingDatasetSources:
    """Resolve dataset id/local copy and preprocessing contract sources (best-effort)."""

    raw_dataset_id = cfg.clearml.raw_dataset_id if cfg.clearml else None
    preprocessed_dataset_id = cfg.clearml.preprocessed_dataset_id if cfg.clearml else None
    dataset_id_for_load = dataset_id_override or preprocessed_dataset_id or raw_dataset_id or getattr(cfg.data, "dataset_id", None)

    local_copy = None
    csv_override = None
    preproc_summary_src: Optional[Path] = None
    preproc_recipe_src: Optional[Path] = None
    preproc_schema_src: Optional[Path] = None
    preproc_manifest_src: Optional[Path] = None
    preproc_bundle_src: Optional[Path] = None
    preproc_bundle = None

    if dataset_id_for_load:
        dataset_id_for_load = normalize_dataset_id(str(dataset_id_for_load), cfg)
    if dataset_id_for_load:
        local_copy = ensure_local_dataset_copy(str(dataset_id_for_load), base_output_dir / "clearml_dataset")
        if cfg.clearml and cfg.clearml.enabled and not local_copy:
            raise ValueError(f"Failed to download ClearML Dataset (dataset_id={dataset_id_for_load})")
        csv_override = find_first_csv(local_copy) if local_copy else None
        try:
            if local_copy:
                pre_dir = Path(local_copy) / "preprocessing"
                summary_cand = pre_dir / "summary.md"
                recipe_cand = pre_dir / "recipe.json"
                bundle_cand = pre_dir / "bundle.joblib"
                schema_cand = Path(local_copy) / "schema.json"
                manifest_cand = Path(local_copy) / "manifest.json"
                if summary_cand.exists():
                    preproc_summary_src = summary_cand
                if recipe_cand.exists():
                    preproc_recipe_src = recipe_cand
                if bundle_cand.exists():
                    preproc_bundle_src = bundle_cand
                if schema_cand.exists():
                    preproc_schema_src = schema_cand
                if manifest_cand.exists():
                    preproc_manifest_src = manifest_cand
        except Exception:
            pass
        try:
            if preproc_bundle_src and preproc_bundle_src.exists():
                from joblib import load  # type: ignore

                preproc_bundle = load(preproc_bundle_src)
        except Exception:
            preproc_bundle = None

    return TrainingDatasetSources(
        dataset_id_for_load=(str(dataset_id_for_load) if dataset_id_for_load else None),
        local_copy=(Path(local_copy) if local_copy else None),
        csv_override=(Path(csv_override) if csv_override else None),
        preproc_summary_src=preproc_summary_src,
        preproc_recipe_src=preproc_recipe_src,
        preproc_schema_src=preproc_schema_src,
        preproc_manifest_src=preproc_manifest_src,
        preproc_bundle_src=preproc_bundle_src,
        preproc_bundle=preproc_bundle,
    )

