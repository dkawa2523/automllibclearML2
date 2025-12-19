from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from automl_lib.config.schemas import TrainingConfig
from automl_lib.registry.metrics import is_loss_metric


def _find_column_case_insensitive(df: pd.DataFrame, key: str) -> Optional[str]:
    if not key:
        return None
    if key in df.columns:
        return key
    lowered = str(key).strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == lowered:
            return str(col)
    return None


def _select_recommend_metric(
    df: pd.DataFrame,
    *,
    primary_metric_model: Optional[str],
    problem_type: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (metric_column_name, goal[min|max])."""

    if df is None or df.empty:
        return None, None

    metric_col = _find_column_case_insensitive(df, str(primary_metric_model or ""))
    if metric_col:
        try:
            goal = "min" if is_loss_metric(metric_col, problem_type=problem_type) else "max"
        except Exception:
            goal = "max"
        return metric_col, goal

    non_metric_cols = {
        "model",
        "preprocessor",
        "params",
        "status",
        "error",
        "train_seconds",
        "predict_seconds",
        "predict_train_seconds",
        "predict_test_seconds",
        "model_size_bytes",
        "num_features",
        "task_id",
        "model_id",
        "model_input",
        "exported_with_preproc_bundle",
        "url",
        "link_html",
        "metric_source",
    }
    for col in df.columns:
        if str(col) in non_metric_cols:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            try:
                goal = "min" if is_loss_metric(str(col), problem_type=problem_type) else "max"
            except Exception:
                continue
            return str(col), goal

    return None, None


def build_recommendation_and_leaderboard(
    *,
    cfg: TrainingConfig,
    output_dir: Path,
    df_links: Optional[pd.DataFrame],
    primary_metric_model: Optional[str],
    problem_type: str,
    run_id: str,
    dataset_key: Optional[str],
    summary_best_model_id: Optional[str],
) -> Dict[str, Any]:
    """
    Build training-summary recommendation artifacts.

    Comparison phase is intentionally removed: recommendation is always based on the training primary metric.

    Writes (best-effort):
    - leaderboard.csv
    - model_tasks_ranked.csv
    - recommended_model.csv
    - recommendation_rationale.json
    - recommendation_rationale.md
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df_links.copy() if isinstance(df_links, pd.DataFrame) else pd.DataFrame()

    training_primary_metric = str(primary_metric_model or "").strip().lower() or None
    recommend_metric, recommend_goal = _select_recommend_metric(df, primary_metric_model=training_primary_metric, problem_type=problem_type)

    if df.empty:
        df_ranked = pd.DataFrame()
        recommended_df = pd.DataFrame()
    else:
        df_ranked = df.copy()
        status = df_ranked.get("status", "").fillna("").astype(str).str.strip().str.lower()
        error = df_ranked.get("error", "").fillna("").astype(str).str.strip()
        ok_mask = status.isin(["", "ok"]) & (error == "")
        df_ranked["_ok"] = ok_mask

        metric_values = None
        metric_col = None
        if recommend_metric:
            metric_col = _find_column_case_insensitive(df_ranked, recommend_metric)
            if metric_col and metric_col in df_ranked.columns:
                metric_values = pd.to_numeric(df_ranked[metric_col], errors="coerce")
        df_ranked["_metric_value"] = metric_values if metric_values is not None else pd.NA

        if recommend_goal == "min":
            df_ranked = df_ranked.sort_values(by=["_ok", "_metric_value"], ascending=[False, True], na_position="last")
        else:
            df_ranked = df_ranked.sort_values(by=["_ok", "_metric_value"], ascending=[False, False], na_position="last")

        df_ranked["rank"] = range(1, len(df_ranked) + 1)
        df_ranked["is_recommended"] = False

        recommended_df = pd.DataFrame()
        if len(df_ranked) > 0:
            cand = df_ranked[df_ranked["_ok"]]
            if recommend_metric:
                cand = cand[cand["_metric_value"].notna()]
            if not cand.empty:
                idx0 = cand.index[0]
                try:
                    df_ranked.loc[idx0, "is_recommended"] = True
                except Exception:
                    pass
                recommended_df = df_ranked.loc[[idx0]].copy()

        df_ranked = df_ranked.drop(columns=[c for c in ["_ok", "_metric_value"] if c in df_ranked.columns], errors="ignore")

    if (recommended_df is None or recommended_df.empty) and summary_best_model_id:
        recommended_df = pd.DataFrame(
            [
                {
                    "model": "",
                    "preprocessor": "",
                    "task_id": "",
                    "model_id": str(summary_best_model_id),
                    "status": "ok",
                    "error": "",
                    "rank": 1,
                    "is_recommended": True,
                }
            ]
        )

    try:
        df_ranked.to_csv(output_dir / "model_tasks_ranked.csv", index=False)
    except Exception:
        pass
    try:
        df_ranked.to_csv(output_dir / "leaderboard.csv", index=False)
    except Exception:
        pass
    try:
        if recommended_df is None:
            recommended_df = pd.DataFrame()
        recommended_df.to_csv(output_dir / "recommended_model.csv", index=False)
    except Exception:
        pass

    rationale: Dict[str, Any] = {
        "run_id": run_id,
        "dataset_key": dataset_key,
        "recommendation_mode": (cfg.clearml.recommendation_mode if cfg.clearml else None),
        "training_primary_metric": training_primary_metric,
        "recommend_metric": recommend_metric,
        "recommend_goal": recommend_goal,
        "filters": {"require_status_ok": True, "require_no_error": True},
    }
    try:
        if recommended_df is not None and not recommended_df.empty:
            rationale["recommended_row"] = recommended_df.iloc[0].to_dict()
    except Exception:
        pass

    try:
        (output_dir / "recommendation_rationale.json").write_text(
            json.dumps(rationale, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass

    try:
        lines = [
            "# Recommendation Rationale",
            "",
            f"- run_id: {run_id}",
            f"- dataset_key: {dataset_key or ''}",
            f"- training_primary_metric: {training_primary_metric or ''}",
            f"- recommend_metric: {recommend_metric or ''}",
            f"- recommend_goal: {recommend_goal or ''}",
            "",
            "## Filters",
            "- require_status_ok: true",
            "- require_no_error: true",
            "",
        ]
        if recommended_df is not None and not recommended_df.empty:
            row = recommended_df.iloc[0].to_dict()
            lines.extend(
                [
                    "## Recommended",
                    f"- model: {row.get('model','')}",
                    f"- preprocessor: {row.get('preprocessor','')}",
                    f"- model_id: {row.get('model_id','')}",
                    f"- task_id: {row.get('task_id','')}",
                ]
            )
        (output_dir / "recommendation_rationale.md").write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    return {
        "recommended_df": recommended_df,
        "df_links_ranked": df_ranked,
        "training_primary_metric": training_primary_metric,
        "recommend_metric": recommend_metric,
        "recommend_goal": recommend_goal,
    }

