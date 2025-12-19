from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def write_model_task_records(output_dir: Path, model_task_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Persist per-model task records to disk and return as a DataFrame."""

    df_links = pd.DataFrame(model_task_records) if model_task_records else pd.DataFrame()
    if df_links is None or df_links.empty:
        return pd.DataFrame()

    try:
        df_links.to_csv(output_dir / "model_metrics.csv", index=False)
    except Exception:
        pass

    try:
        errors: List[Dict[str, Any]] = []
        for r in model_task_records:
            if not isinstance(r, dict):
                continue
            status = str(r.get("status") or "").strip().lower()
            err = str(r.get("error") or "").strip()
            if status in {"ok", ""} and not err:
                continue
            errors.append(
                {
                    "model": r.get("model"),
                    "preprocessor": r.get("preprocessor"),
                    "task_id": r.get("task_id"),
                    "model_id": r.get("model_id"),
                    "status": status or "failed",
                    "error": err,
                }
            )
        if errors:
            (output_dir / "error_models.json").write_text(
                json.dumps(errors, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
    except Exception:
        pass

    try:
        df_fail = df_links.copy()
        if "status" in df_fail.columns:
            st = df_fail["status"].fillna("").astype(str).str.strip().str.lower()
            bad_status = ~st.isin(["", "ok"])
        else:
            bad_status = pd.Series([False] * len(df_fail))
        if "error" in df_fail.columns:
            er = df_fail["error"].fillna("").astype(str).str.strip()
            bad_error = er != ""
        else:
            bad_error = pd.Series([False] * len(df_fail))
        df_fail = df_fail[bad_status | bad_error]
        if not df_fail.empty:
            df_fail.to_csv(output_dir / "model_task_failures.csv", index=False)
    except Exception:
        pass

    return df_links


def collect_training_summary_artifacts(
    *,
    output_dir: Path,
    results_path: Path,
    full_config_path: Optional[Path],
    preproc_summary_path: Optional[Path],
    preproc_recipe_path: Optional[Path],
    preproc_schema_path: Optional[Path],
    preproc_manifest_path: Optional[Path],
) -> List[Path]:
    """Collect key (small) artifacts for the training-summary task."""

    artifacts: List[Path] = []
    artifacts.append(results_path)
    artifacts.append(results_path.with_suffix(".json"))

    if full_config_path and full_config_path.exists():
        artifacts.append(full_config_path)

    for p in [preproc_summary_path, preproc_recipe_path, preproc_schema_path, preproc_manifest_path]:
        if p and p.exists():
            artifacts.append(p)

    for name in [
        "recommended_model.csv",
        "recommendation_rationale.json",
        "recommendation_rationale.md",
        "model_metrics.csv",
        "model_tasks_ranked.csv",
        "leaderboard.csv",
        "model_task_failures.csv",
        "error_models.json",
    ]:
        cand = output_dir / name
        if cand.exists():
            artifacts.append(cand)

    proj_csv = output_dir / "interpolation_space" / "feature_space_projection.csv"
    if proj_csv.exists():
        artifacts.append(proj_csv)

    return artifacts

