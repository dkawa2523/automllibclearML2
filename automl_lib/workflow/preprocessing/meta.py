"""
Metadata management for preprocessing phase.
前処理の結果を要約し、Artifacts/テーブル用のメタ情報を構築する。
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_or_none(items: List[Any]) -> Any:
    return items[0] if items else None


def build_preprocessing_metadata(
    *,
    output_dir: Path,
    run_id: str,
    dataset_key: str,
    parent_dataset_id: Optional[str] = None,
    preprocessing_task_id: Optional[str] = None,
    contract_version: str = "v1",
    target_col: str,
    feature_cols: Sequence[str],
    feature_types: Dict[str, List[str]],
    preproc_name: str,
    cfg_preprocessing: Optional[Dict[str, Any]] = None,
    df_raw: Optional[pd.DataFrame] = None,
    df_preprocessed: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Persist light-weight metadata artifacts for ClearML.

    Returns:
      dict with "artifacts": list[str] (file paths)
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[str] = []

    n_rows = int(df_raw.shape[0]) if isinstance(df_raw, pd.DataFrame) else None
    n_features_raw = int(len(feature_cols))
    n_features_pre = None
    if isinstance(df_preprocessed, pd.DataFrame):
        n_features_pre = int(df_preprocessed.shape[1] - (1 if target_col in df_preprocessed.columns else 0))

    schema = {
        "run_id": str(run_id),
        "dataset_key": str(dataset_key),
        "target_column": str(target_col),
        "feature_columns": list(feature_cols),
        "feature_types": {
            "numeric": list(feature_types.get("numeric") or feature_types.get("numeric_cols") or []),
            "categorical": list(feature_types.get("categorical") or feature_types.get("categorical_cols") or []),
        },
        "n_rows": n_rows,
        "n_features_raw": n_features_raw,
        "n_features_preprocessed": n_features_pre,
    }

    schema_path = output_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(schema_path))

    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        "run_id": str(run_id),
        "dataset_key": str(dataset_key),
        "parent_dataset_id": str(parent_dataset_id) if parent_dataset_id else None,
        "preprocessing_task_id": str(preprocessing_task_id) if preprocessing_task_id else None,
        "created_at": created_at,
        "contract_version": str(contract_version),
        "selected_preprocessor": str(preproc_name),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(manifest_path))

    cfg_p = cfg_preprocessing or {}
    # These config fields are candidate lists; preprocessing task picks one preprocessor.
    # We keep the raw config for reproducibility and derive a readable summary separately.
    numeric_cols_present = bool(schema["feature_types"]["numeric"])
    categorical_cols_present = bool(schema["feature_types"]["categorical"])

    numeric_imputation_candidates = [None] if not numeric_cols_present else (_as_list(cfg_p.get("numeric_imputation")) or [None])
    scaling_candidates = [None] if not numeric_cols_present else (_as_list(cfg_p.get("scaling")) or [None])

    poly_raw = cfg_p.get("polynomial_degree")
    if not numeric_cols_present:
        polynomial_degree_candidates = [int(poly_raw) if isinstance(poly_raw, int) and poly_raw > 1 else None]
    else:
        if isinstance(poly_raw, int) and poly_raw > 1:
            polynomial_degree_candidates = [None, int(poly_raw)]
        else:
            polynomial_degree_candidates = [None]

    categorical_imputation_candidates = (
        [None]
        if not categorical_cols_present
        else (_as_list(cfg_p.get("categorical_imputation")) or [None])
    )
    encoding_candidates = [None] if not categorical_cols_present else (_as_list(cfg_p.get("categorical_encoding")) or [None])

    selected_settings = {
        "numeric_imputation": _first_or_none(numeric_imputation_candidates),
        "categorical_imputation": _first_or_none(categorical_imputation_candidates),
        "scaling": _first_or_none(scaling_candidates),
        "categorical_encoding": _first_or_none(encoding_candidates),
        "polynomial_degree": _first_or_none(polynomial_degree_candidates),
        "numeric_pipeline_steps": cfg_p.get("numeric_pipeline_steps") or [],
        "categorical_pipeline_steps": cfg_p.get("categorical_pipeline_steps") or [],
    }

    target_standardize_requested = bool(cfg_p.get("target_standardize"))
    target_standardize_enabled = target_standardize_requested
    try:
        if target_standardize_requested and isinstance(df_raw, pd.DataFrame) and target_col in df_raw.columns:
            y_numeric = pd.to_numeric(df_raw[target_col], errors="coerce")
            target_standardize_enabled = not y_numeric.dropna().empty
    except Exception:
        target_standardize_enabled = target_standardize_requested
    recipe = {
        "contract_version": str(contract_version),
        "dataset_lineage": {
            "parent_dataset_id": str(parent_dataset_id) if parent_dataset_id else None,
            "preprocessing_task_id": str(preprocessing_task_id) if preprocessing_task_id else None,
            "created_at": created_at,
        },
        "selected_preprocessor": str(preproc_name),
        "selected_settings": selected_settings,
        "candidates": {
            "numeric_imputation": numeric_imputation_candidates,
            "categorical_imputation": categorical_imputation_candidates,
            "scaling": scaling_candidates,
            "categorical_encoding": encoding_candidates,
            "polynomial_degree": polynomial_degree_candidates,
        },
        "config": cfg_p,
        "features": {
            "numeric": schema["feature_types"]["numeric"],
            "categorical": schema["feature_types"]["categorical"],
        },
        "target": {
            "target_column": str(target_col),
            "transform": ("standardize" if target_standardize_enabled else "none"),
            "transformer": ("StandardScaler" if target_standardize_enabled else None),
            "inverse_transform": bool(target_standardize_enabled),
            "standardize_requested": bool(target_standardize_requested),
        },
    }

    # Missing values quick summary (top 20 by count)
    missing_top: List[Dict[str, Any]] = []
    if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
        cols = list(feature_cols)
        if target_col and target_col not in cols and target_col in df_raw.columns:
            cols.append(target_col)
        total = int(df_raw.shape[0])
        for col in cols:
            if col not in df_raw.columns:
                continue
            try:
                missing = int(df_raw[col].isna().sum())
            except Exception:
                continue
            if missing <= 0:
                continue
            pct = float(missing / total * 100.0) if total > 0 else 0.0
            missing_top.append({"column": str(col), "missing": missing, "pct": pct})
        missing_top = sorted(missing_top, key=lambda r: float(r.get("missing") or 0), reverse=True)[:20]
    recipe["missing_values"] = missing_top

    pre_dir = output_dir / "preprocessing"
    pre_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = pre_dir / "recipe.json"
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(recipe_path))

    # Human-readable summary for ClearML (category-based, avoids single-line dumping).
    md_lines: List[str] = []
    md_lines.append("# Preprocessing Summary")
    md_lines.append("")
    md_lines.append("## Dataset Lineage")
    md_lines.append(f"- parent_dataset_id: {parent_dataset_id or ''}")
    md_lines.append(f"- preprocessing_task_id: {preprocessing_task_id or ''}")
    md_lines.append(f"- created_at: {created_at}")
    md_lines.append(f"- contract_version: {contract_version}")
    md_lines.append("")
    md_lines.append("## Input Features")
    md_lines.append("### Type Inference")
    numeric_cols = list(schema["feature_types"]["numeric"])
    categorical_cols = list(schema["feature_types"]["categorical"])
    sample_n = 20

    md_lines.append(f"- numerical (n={len(numeric_cols)}):")
    for col in numeric_cols[:sample_n]:
        md_lines.append(f"  - {col}")
    if len(numeric_cols) > sample_n:
        md_lines.append(f"  - ... ({len(numeric_cols) - sample_n} more)")

    md_lines.append(f"- categorical (n={len(categorical_cols)}):")
    for col in categorical_cols[:sample_n]:
        md_lines.append(f"  - {col}")
    if len(categorical_cols) > sample_n:
        md_lines.append(f"  - ... ({len(categorical_cols) - sample_n} more)")
    md_lines.append("")
    md_lines.append("### Missing Values")
    md_lines.append(f"- numeric_imputation_selected: {selected_settings.get('numeric_imputation')}")
    md_lines.append(f"- categorical_imputation_selected: {selected_settings.get('categorical_imputation')}")
    if missing_top:
        md_lines.append("- columns_with_missing_top20:")
        for row in missing_top:
            md_lines.append(
                f"  - {row.get('column')}: {row.get('missing')} ({row.get('pct'):.2f}%)"
            )
    else:
        md_lines.append("- columns_with_missing_top20: []")
    md_lines.append("")
    md_lines.append("### Encoding")
    md_lines.append(f"- categorical_encoding_selected: {selected_settings.get('categorical_encoding')}")
    md_lines.append("")
    md_lines.append("### Scaling / Polynomial")
    md_lines.append(f"- scaling_selected: {selected_settings.get('scaling')}")
    md_lines.append(f"- polynomial_degree_selected: {selected_settings.get('polynomial_degree')}")
    md_lines.append("")
    md_lines.append("## Target")
    md_lines.append(f"- target_column: {target_col}")
    md_lines.append(f"- transform: {recipe['target']['transform']}")
    md_lines.append(f"- transformer: {recipe['target'].get('transformer') or ''}")
    md_lines.append(f"- inverse_transform: {recipe['target']['inverse_transform']}")
    md_lines.append("")
    md_lines.append("## Selected Preprocessor")
    md_lines.append(f"- name: {preproc_name}")
    md_lines.append("")

    summary_path = pre_dir / "summary.md"
    summary_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    artifacts.append(str(summary_path))

    version_path = pre_dir / "version.txt"
    version_path.write_text(str(contract_version).strip() + "\n", encoding="utf-8")
    artifacts.append(str(version_path))

    return {"artifacts": artifacts, "schema": schema, "manifest": manifest, "recipe": recipe}
