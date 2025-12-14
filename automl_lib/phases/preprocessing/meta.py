"""
Metadata management for preprocessing phase.
前処理の結果を要約し、Artifacts/テーブル用のメタ情報を構築する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def build_preprocessing_metadata(
    *,
    output_dir: Path,
    run_id: str,
    dataset_key: str,
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

    schema_path = output_dir / "schema_raw.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(schema_path))

    pipeline_meta = {
        "run_id": str(run_id),
        "dataset_key": str(dataset_key),
        "selected_preprocessor": str(preproc_name),
        "preprocessing_config": cfg_preprocessing or {},
    }
    pipeline_path = output_dir / "preprocess_pipeline.json"
    pipeline_path.write_text(json.dumps(pipeline_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append(str(pipeline_path))

    return {"artifacts": artifacts, "schema": schema, "pipeline": pipeline_meta}
