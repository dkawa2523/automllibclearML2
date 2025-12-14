"""
Visualization utilities for preprocessing phase.
分布確認や欠損、特徴量数などの診断可視化を担当する。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import pandas as pd

try:
    import plotly.express as px  # type: ignore
except Exception:  # pragma: no cover
    px = None

from automl_lib.clearml.logging import report_plotly, report_table


def _is_classification_like(y: pd.Series) -> bool:
    try:
        if pd.api.types.is_bool_dtype(y):
            return True
        if not pd.api.types.is_numeric_dtype(y):
            return True
        nunique = int(y.dropna().nunique())
        return nunique <= 20
    except Exception:
        return True


def render_preprocessing_visuals(
    logger,
    *,
    df_raw: pd.DataFrame,
    df_preprocessed: Optional[pd.DataFrame],
    target_col: str,
    feature_cols: Sequence[str],
    feature_types: Dict[str, List[str]],
    max_plot_rows: int = 10000,
    top_n: int = 20,
) -> None:
    if not logger:
        return
    if df_raw is None or df_raw.empty:
        return

    # Target distribution
    if target_col in df_raw.columns:
        try:
            y = df_raw[target_col]
            y_nonnull = y.dropna()
            if len(y_nonnull) > max_plot_rows:
                y_nonnull = y_nonnull.sample(n=max_plot_rows, random_state=0)
            if _is_classification_like(y_nonnull):
                vc = y_nonnull.astype(str).value_counts()
                if len(vc) > top_n:
                    head = vc.iloc[:top_n]
                    rest = vc.iloc[top_n:].sum()
                    vc = pd.concat([head, pd.Series({"Others": rest})])
                df_counts = vc.reset_index()
                df_counts.columns = ["class", "count"]
                report_table(logger, title="target_distribution_table", df=df_counts, series="target")
                if px is not None:
                    fig = px.bar(df_counts, x="class", y="count", title="Target distribution (classification-like)")
                    report_plotly(logger, title="target_distribution", series="target", figure=fig)
            else:
                df_y = pd.DataFrame({"target": y_nonnull})
                if px is not None:
                    fig = px.histogram(df_y, x="target", nbins=30, title="Target distribution (regression-like)")
                    report_plotly(logger, title="target_distribution", series="target", figure=fig)
        except Exception:
            pass

    # Missing summary (top-N columns)
    try:
        missing = df_raw.isna().sum()
        if int(missing.sum()) > 0:
            miss_df = pd.DataFrame(
                {
                    "column": missing.index.astype(str),
                    "missing_count": missing.values.astype(int),
                }
            )
            miss_df["missing_rate"] = miss_df["missing_count"] / float(max(1, len(df_raw)))
            miss_df = miss_df[miss_df["missing_count"] > 0].sort_values("missing_count", ascending=False)
            if hasattr(miss_df, "head"):
                miss_df_top = miss_df.head(top_n)
            else:
                miss_df_top = miss_df
            report_table(logger, title="missing_summary", df=miss_df_top, series="quality")
            if px is not None and not miss_df_top.empty:
                fig = px.bar(
                    miss_df_top[::-1],
                    x="missing_rate",
                    y="column",
                    orientation="h",
                    title="Missing rate by column (top)",
                )
                report_plotly(logger, title="missing_rate_top", series="quality", figure=fig)
    except Exception:
        pass

    # Feature count before/after
    try:
        n_raw = int(len(feature_cols))
        n_pre = None
        if isinstance(df_preprocessed, pd.DataFrame):
            n_pre = int(df_preprocessed.shape[1] - (1 if target_col in df_preprocessed.columns else 0))
        rows = [{"stage": "raw", "n_features": n_raw}]
        if n_pre is not None:
            rows.append({"stage": "preprocessed", "n_features": n_pre})
        feat_df = pd.DataFrame(rows)
        report_table(logger, title="feature_count", df=feat_df, series="summary")
        if px is not None and not feat_df.empty:
            fig = px.bar(feat_df, x="stage", y="n_features", title="Number of features (raw vs preprocessed)")
            report_plotly(logger, title="feature_count", series="summary", figure=fig)
    except Exception:
        pass

    # Numeric columns quick stats (representative subset)
    try:
        numeric_cols = list(feature_types.get("numeric") or [])
        if numeric_cols:
            cols = numeric_cols[: min(top_n, len(numeric_cols))]
            desc = df_raw[cols].describe().T.reset_index().rename(columns={"index": "column"})
            report_table(logger, title="numeric_describe", df=desc, series="summary")
    except Exception:
        pass
