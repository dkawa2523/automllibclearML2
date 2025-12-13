"""
Metadata management for comparison phase.
ClearML から取得した指標を DataFrame にまとめ、Artifacts 用の保存パスを返す。
"""

import json
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import pandas as pd


_MINIMIZE_METRICS = {
    "mse",
    "rmse",
    "mae",
    "mape",
    "smape",
    "logloss",
    "loss",
    "error",
}


def _resolve_goal(metric: Optional[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        lowered = str(explicit).strip().lower()
        if lowered in {"min", "max"}:
            return lowered
    if not metric:
        return None
    return "min" if str(metric).strip().lower() in _MINIMIZE_METRICS else "max"


def build_comparison_metadata(
    rows: List[Dict[str, Any]],
    *,
    output_dir: Optional[Path] = None,
    metric_cols: Optional[Sequence[str]] = None,
    primary_metric: Optional[str] = None,
    goal: Optional[str] = None,
    group_col: Optional[str] = None,
    top_k: Optional[int] = None,
    composite_enabled: bool = True,
    composite_metrics: Optional[Sequence[str]] = None,
    composite_weights: Optional[Dict[str, float]] = None,
    composite_require_all: bool = False,
) -> Dict[str, Any]:
    """
    rows: list of dicts with metrics per task
    output_dir: optional path to save CSV
    """
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df is not None and not df.empty:
        try:
            df.columns = [str(c).strip().lower() for c in df.columns]
        except Exception:
            pass
    metric_cols = [str(m).strip().lower() for m in (metric_cols or []) if str(m).strip()]

    # If primary_metric is not provided, try the first desired metric that exists in df.
    if primary_metric:
        primary_metric = str(primary_metric).strip().lower()
    if (
        primary_metric
        and primary_metric != "composite_score"
        and df is not None
        and not df.empty
        and primary_metric not in df.columns
    ):
        primary_metric = None
    if (not primary_metric) and df is not None and not df.empty:
        for cand in metric_cols:
            if cand in df.columns:
                primary_metric = cand
                break

    # Composite score (higher is better): min-max normalize each metric and weighted-average.
    # - metrics selection priority: composite_metrics > composite_weights keys > metric_cols
    # - weights default to 1.0 (unless composite_weights provided)
    if df is not None and not df.empty and composite_enabled:
        weights_map: Dict[str, float] = {}
        if isinstance(composite_weights, dict):
            for k, v in composite_weights.items():
                try:
                    key = str(k).strip().lower()
                except Exception:
                    continue
                try:
                    weight = float(v)
                except Exception:
                    continue
                if weight <= 0:
                    continue
                weights_map[key] = weight

        include_metrics: List[str] = []
        if composite_metrics:
            include_metrics = [str(m).strip().lower() for m in composite_metrics if str(m).strip()]
        elif weights_map:
            include_metrics = list(weights_map.keys())
        else:
            include_metrics = list(metric_cols)

        include_metrics = [m for m in include_metrics if m and m != "composite_score"]
        available = [m for m in include_metrics if m in df.columns]

        wants_composite_primary = primary_metric == "composite_score"
        min_needed = 1 if wants_composite_primary else 2
        if len(available) >= min_needed:
            norm_df = pd.DataFrame(index=df.index)
            weights_used: Dict[str, float] = {}
            for metric in available:
                weight = float(weights_map.get(metric, 1.0))
                if weight <= 0:
                    continue
                values = pd.to_numeric(df[metric], errors="coerce")
                if values.notna().sum() == 0:
                    continue
                vmin = float(values.min())
                vmax = float(values.max())
                denom = vmax - vmin
                norm = pd.Series([float("nan")] * len(values), index=values.index, dtype="float64")
                mask = values.notna()
                if denom > 0:
                    norm.loc[mask] = (values.loc[mask] - vmin) / denom
                else:
                    norm.loc[mask] = 0.5
                if _resolve_goal(metric) == "min":
                    norm = 1.0 - norm
                norm_df[metric] = norm
                weights_used[metric] = weight

            if not norm_df.empty and weights_used:
                weight_vec = pd.Series(weights_used)
                weighted_vals = norm_df.mul(weight_vec, axis=1)
                weighted_present = norm_df.notna().mul(weight_vec, axis=1)
                denom = weighted_present.sum(axis=1)
                score = weighted_vals.sum(axis=1, skipna=True) / denom
                score = score.where(denom > 0)
                if composite_require_all:
                    all_present = norm_df.notna().all(axis=1)
                    score = score.where(all_present)
                df = df.assign(composite_score=score)
                if not primary_metric:
                    primary_metric = "composite_score"

    # If user explicitly requested composite_score but it could not be computed, fall back.
    if primary_metric == "composite_score" and (df is None or df.empty or "composite_score" not in df.columns):
        primary_metric = None
        if df is not None and not df.empty:
            for cand in metric_cols:
                if cand in df.columns:
                    primary_metric = cand
                    break
    goal = _resolve_goal(primary_metric, explicit=goal)

    ranked_df = df
    ranked_topk_df: Optional[pd.DataFrame] = None
    best: Optional[Dict[str, Any]] = None
    best_by_group_df: Optional[pd.DataFrame] = None
    best_by_group: Optional[Dict[str, Any]] = None
    best_by_model_df: Optional[pd.DataFrame] = None
    best_by_model: Optional[Dict[str, Any]] = None
    best_by_group_model_df: Optional[pd.DataFrame] = None
    model_summary_df: Optional[pd.DataFrame] = None
    win_summary_df: Optional[pd.DataFrame] = None
    recommended_model: Optional[Dict[str, Any]] = None
    if df is not None and not df.empty and primary_metric and primary_metric in df.columns:
        values = pd.to_numeric(df[primary_metric], errors="coerce")
        ranked_df = df.assign(**{f"__{primary_metric}_numeric": values})
        ranked_df = ranked_df.sort_values(
            by=f"__{primary_metric}_numeric",
            ascending=(goal == "min"),
            na_position="last",
        ).drop(columns=[f"__{primary_metric}_numeric"])

        if top_k is not None:
            try:
                k = int(top_k)
                if k >= 1:
                    ranked_topk_df = ranked_df.head(k)
            except Exception:
                ranked_topk_df = None
        try:
            best_row = ranked_df.iloc[0].to_dict()
            best = {
                "primary_metric": primary_metric,
                "goal": goal,
                "best_row": best_row,
            }
        except Exception:
            best = None

        if "model" in df.columns:
            try:
                tmp = df.assign(**{f"__{primary_metric}_numeric": values})
                tmp = tmp.sort_values(
                    by=["model", f"__{primary_metric}_numeric"],
                    ascending=[True, (goal == "min")],
                    na_position="last",
                )
                best_by_model_df = (
                    tmp.groupby("model", as_index=False)
                    .head(1)
                    .drop(columns=[f"__{primary_metric}_numeric"])
                )
                best_by_model = {
                    "group_col": "model",
                    "primary_metric": primary_metric,
                    "goal": goal,
                    "best_rows": best_by_model_df.to_dict(orient="records"),
                }
            except Exception:
                best_by_model_df = None
                best_by_model = None

        if group_col and group_col in df.columns:
            try:
                tmp = df.assign(**{f"__{primary_metric}_numeric": values})
                tmp = tmp.sort_values(
                    by=[group_col, f"__{primary_metric}_numeric"],
                    ascending=[True, (goal == "min")],
                    na_position="last",
                )
                best_by_group_df = (
                    tmp.groupby(group_col, as_index=False)
                    .head(1)
                    .drop(columns=[f"__{primary_metric}_numeric"])
                )
                best_by_group = {
                    "group_col": group_col,
                    "primary_metric": primary_metric,
                    "goal": goal,
                    "best_rows": best_by_group_df.to_dict(orient="records"),
                }
            except Exception:
                best_by_group_df = None
                best_by_group = None

            if best_by_group_df is not None and "model" in best_by_group_df.columns:
                try:
                    counts = best_by_group_df["model"].value_counts(dropna=True)
                    win_summary_df = counts.rename("n_wins").reset_index().rename(columns={"index": "model"})
                    denom = float(len(best_by_group_df)) if len(best_by_group_df) > 0 else 0.0
                    if denom > 0:
                        win_summary_df["win_rate"] = win_summary_df["n_wins"] / denom
                    win_summary_df = win_summary_df.sort_values(by=["n_wins", "model"], ascending=[False, True])
                except Exception:
                    win_summary_df = None

            try:
                if "model" in df.columns:
                    tmp = df.assign(**{f"__{primary_metric}_numeric": values})
                    tmp = tmp.sort_values(
                        by=[group_col, "model", f"__{primary_metric}_numeric"],
                        ascending=[True, True, (goal == "min")],
                        na_position="last",
                    )
                    best_by_group_model_df = (
                        tmp.groupby([group_col, "model"], as_index=False)
                        .head(1)
                        .drop(columns=[f"__{primary_metric}_numeric"])
                    )
                    numeric_best = pd.to_numeric(best_by_group_model_df[primary_metric], errors="coerce")
                    summary = best_by_group_model_df.assign(**{f"__{primary_metric}_numeric": numeric_best})
                    model_summary_df = (
                        summary.groupby("model", as_index=False)[f"__{primary_metric}_numeric"]
                        .agg(["count", "mean", "median", "std"])
                        .reset_index()
                        .rename(
                            columns={
                                "count": "n_runs",
                                "mean": f"{primary_metric}_mean",
                                "median": f"{primary_metric}_median",
                                "std": f"{primary_metric}_std",
                            }
                        )
                    )
                    model_summary_df = model_summary_df.sort_values(
                        by=f"{primary_metric}_mean",
                        ascending=(goal == "min"),
                        na_position="last",
                    )

                    try:
                        top = model_summary_df.iloc[0].to_dict()
                        recommended_model = {
                            "strategy": "best_mean",
                            "group_col": group_col,
                            "primary_metric": primary_metric,
                            "goal": goal,
                            "selected_model": top.get("model"),
                            "stats": top,
                        }
                    except Exception:
                        recommended_model = None
            except Exception:
                best_by_group_model_df = None
                model_summary_df = None

    artifacts: List[str] = []
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = output_dir / "comparison_metrics.csv"
        df.to_csv(raw_csv, index=False)
        artifacts.append(str(raw_csv))

        ranked_csv = output_dir / "comparison_ranked.csv"
        ranked_df.to_csv(ranked_csv, index=False)
        artifacts.append(str(ranked_csv))

        if ranked_topk_df is not None:
            ranked_topk_csv = output_dir / "comparison_ranked_topk.csv"
            ranked_topk_df.to_csv(ranked_topk_csv, index=False)
            artifacts.append(str(ranked_topk_csv))

        if best is not None:
            best_json = output_dir / "best_result.json"
            with best_json.open("w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)
            artifacts.append(str(best_json))

        if best_by_model_df is not None:
            best_by_model_csv = output_dir / "best_by_model.csv"
            best_by_model_df.to_csv(best_by_model_csv, index=False)
            artifacts.append(str(best_by_model_csv))
        if best_by_model is not None:
            best_by_model_json = output_dir / "best_by_model.json"
            with best_by_model_json.open("w", encoding="utf-8") as f:
                json.dump(best_by_model, f, ensure_ascii=False, indent=2)
            artifacts.append(str(best_by_model_json))

        if best_by_group_df is not None:
            best_by_run_csv = output_dir / "best_by_run.csv"
            best_by_group_df.to_csv(best_by_run_csv, index=False)
            artifacts.append(str(best_by_run_csv))
        if best_by_group is not None:
            best_by_run_json = output_dir / "best_by_run.json"
            with best_by_run_json.open("w", encoding="utf-8") as f:
                json.dump(best_by_group, f, ensure_ascii=False, indent=2)
            artifacts.append(str(best_by_run_json))
        if best_by_group_model_df is not None:
            best_by_run_model_csv = output_dir / "best_by_run_model.csv"
            best_by_group_model_df.to_csv(best_by_run_model_csv, index=False)
            artifacts.append(str(best_by_run_model_csv))
        if model_summary_df is not None:
            model_summary_csv = output_dir / "model_summary.csv"
            model_summary_df.to_csv(model_summary_csv, index=False)
            artifacts.append(str(model_summary_csv))
        if win_summary_df is not None:
            win_summary_csv = output_dir / "win_summary.csv"
            win_summary_df.to_csv(win_summary_csv, index=False)
            artifacts.append(str(win_summary_csv))
        if recommended_model is not None:
            recommended_json = output_dir / "recommended_model.json"
            with recommended_json.open("w", encoding="utf-8") as f:
                json.dump(recommended_model, f, ensure_ascii=False, indent=2)
            artifacts.append(str(recommended_json))

    return {
        "df": df,
        "ranked_df": ranked_df,
        "ranked_topk_df": ranked_topk_df,
        "best": best,
        "best_by_group_df": best_by_group_df,
        "best_by_group": best_by_group,
        "best_by_model_df": best_by_model_df,
        "best_by_model": best_by_model,
        "best_by_group_model_df": best_by_group_model_df,
        "model_summary_df": model_summary_df,
        "win_summary_df": win_summary_df,
        "recommended_model": recommended_model,
        "artifacts": artifacts,
    }
