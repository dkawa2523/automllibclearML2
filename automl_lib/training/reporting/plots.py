from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationPlotPaths:
    confusion_csv: Optional[Path] = None
    confusion_png: Optional[Path] = None
    confusion_normalized_csv: Optional[Path] = None
    confusion_normalized_png: Optional[Path] = None
    roc_png: Optional[Path] = None
    roc_auc_csv: Optional[Path] = None
    pr_png: Optional[Path] = None
    pr_ap_csv: Optional[Path] = None


def _as_labels(labels: Optional[Sequence[object]], y_true: np.ndarray, y_pred: np.ndarray) -> Optional[List[object]]:
    if labels:
        out = [l for l in labels]
        return out if out else None
    try:
        vals = list(pd.unique(pd.Series(list(y_true) + list(y_pred))))
        return vals if vals else None
    except Exception:
        return None


def _safe_mkdir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return


def _plot_matrix(
    matrix: np.ndarray,
    labels: Sequence[object],
    *,
    title: str,
    path_png: Path,
    fmt: str,
    cmap: str = "Blues",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    try:
        size = max(4.0, 0.6 * max(2, len(labels)))
        fig, ax = plt.subplots(figsize=(size, size))
        ax.imshow(matrix, cmap=cmap)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([str(l) for l in labels], rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([str(l) for l in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(str(title))
        try:
            vmax = float(np.nanmax(matrix)) if matrix.size else 0.0
            thresh = vmax / 2.0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    if val is None or (isinstance(val, float) and val != val):  # NaN
                        continue
                    fval = float(val)
                    ax.text(
                        j,
                        i,
                        format(fval, fmt),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=("white" if fval > thresh else "black"),
                    )
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(path_png, dpi=150)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass


def save_confusion_matrices(
    *,
    y_true: Union[Sequence[object], np.ndarray],
    y_pred: Union[Sequence[object], np.ndarray],
    labels: Optional[Sequence[object]],
    out_dir: Path,
    base_name: str,
    title_prefix: str = "",
) -> ClassificationPlotPaths:
    """Save confusion matrix (counts + normalized-by-true) as CSV/PNG."""

    try:
        from sklearn.metrics import confusion_matrix  # type: ignore
    except Exception:
        return ClassificationPlotPaths()

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labs = _as_labels(labels, y_true_arr, y_pred_arr)
    if not labs:
        return ClassificationPlotPaths()

    _safe_mkdir(out_dir)

    cm = None
    try:
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labs)
    except Exception:
        cm = None
    if cm is None:
        return ClassificationPlotPaths()

    cm_df = pd.DataFrame(cm, index=[str(l) for l in labs], columns=[str(l) for l in labs])
    csv_path = out_dir / f"{base_name}.csv"
    png_path = out_dir / f"{base_name}.png"
    try:
        cm_df.to_csv(csv_path, index=True)
    except Exception:
        csv_path = None  # type: ignore[assignment]
    try:
        _plot_matrix(
            cm.astype(float),
            labs,
            title=f"{title_prefix}Confusion Matrix".strip(),
            path_png=png_path,
            fmt=".0f",
            cmap="Blues",
        )
    except Exception:
        png_path = None  # type: ignore[assignment]

    # normalized (by true row sum)
    cm_norm = cm.astype("float64")
    try:
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.divide(cm_norm, row_sums, where=(row_sums != 0))
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    except Exception:
        cm_norm = None

    norm_csv_path = out_dir / f"{base_name}_normalized.csv"
    norm_png_path = out_dir / f"{base_name}_normalized.png"
    if cm_norm is not None:
        cm_norm_df = pd.DataFrame(cm_norm, index=[str(l) for l in labs], columns=[str(l) for l in labs])
        try:
            cm_norm_df.to_csv(norm_csv_path, index=True)
        except Exception:
            norm_csv_path = None  # type: ignore[assignment]
        try:
            _plot_matrix(
                cm_norm.astype(float),
                labs,
                title=f"{title_prefix}Confusion Matrix (Normalized)".strip(),
                path_png=norm_png_path,
                fmt=".2f",
                cmap="Blues",
            )
        except Exception:
            norm_png_path = None  # type: ignore[assignment]
    else:
        norm_csv_path = None  # type: ignore[assignment]
        norm_png_path = None  # type: ignore[assignment]

    return ClassificationPlotPaths(
        confusion_csv=(csv_path if isinstance(csv_path, Path) and csv_path.exists() else None),
        confusion_png=(png_path if isinstance(png_path, Path) and png_path.exists() else None),
        confusion_normalized_csv=(norm_csv_path if isinstance(norm_csv_path, Path) and norm_csv_path.exists() else None),
        confusion_normalized_png=(norm_png_path if isinstance(norm_png_path, Path) and norm_png_path.exists() else None),
    )


def _resolve_scores_for_binary(
    scores: np.ndarray, classes: Optional[Sequence[object]]
) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """Return (pos_scores, pos_label) for binary classification when possible."""
    if scores.ndim == 2 and scores.shape[1] == 2:
        pos_scores = scores[:, 1]
        pos_label = None
        if classes and len(classes) >= 2:
            pos_label = classes[1]
        return pos_scores, pos_label
    if scores.ndim == 1:
        pos_label = None
        if classes and len(classes) >= 2:
            pos_label = classes[1]
        return scores, pos_label
    return None, None


def save_roc_pr_curves(
    *,
    y_true: Union[Sequence[object], np.ndarray],
    scores: Union[Sequence[object], np.ndarray],
    classes: Optional[Sequence[object]],
    out_roc_dir: Path,
    out_pr_dir: Path,
    base_name: str,
    title_prefix: str = "",
    max_classes: int = 10,
) -> ClassificationPlotPaths:
    """Save ROC/PR curves as PNG (+ summary CSV).

    - binary: single curve + AUC/AP
    - multiclass: one-vs-rest curves when classes <= max_classes
    """

    try:
        from sklearn.metrics import auc, precision_recall_curve, roc_curve  # type: ignore
        from sklearn.metrics import average_precision_score  # type: ignore
        from sklearn.preprocessing import label_binarize  # type: ignore
    except Exception:
        return ClassificationPlotPaths()

    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)
    if y_true_arr.size == 0:
        return ClassificationPlotPaths()

    # Resolve classes
    cls = list(classes) if classes else None
    if not cls:
        try:
            cls = list(pd.unique(pd.Series(list(y_true_arr))))
        except Exception:
            cls = None
    if not cls or len(cls) < 2:
        return ClassificationPlotPaths()

    roc_png = out_roc_dir / f"{base_name}.png"
    roc_auc_csv = out_roc_dir / f"{base_name}.csv"
    pr_png = out_pr_dir / f"{base_name}.png"
    pr_ap_csv = out_pr_dir / f"{base_name}.csv"

    _safe_mkdir(out_roc_dir)
    _safe_mkdir(out_pr_dir)

    # Matplotlib plotters
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return ClassificationPlotPaths()

    # Binary
    pos_scores, pos_label = _resolve_scores_for_binary(scores_arr, cls)
    if pos_scores is not None:
        try:
            if pos_label is None:
                pos_label = cls[1]
            y_bin = (y_true_arr == pos_label).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, pos_scores)
            roc_auc = float(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_bin, pos_scores)
            ap = float(average_precision_score(y_bin, pos_scores))

            # ROC
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{title_prefix}ROC Curve".strip())
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(roc_png, dpi=150)
            plt.close(fig)

            # PR
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec, prec, label=f"AP={ap:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{title_prefix}Precision-Recall Curve".strip())
            ax.legend(loc="lower left")
            fig.tight_layout()
            fig.savefig(pr_png, dpi=150)
            plt.close(fig)

            # Summary CSVs
            pd.DataFrame([{"class": str(pos_label), "auc": roc_auc}]).to_csv(roc_auc_csv, index=False)
            pd.DataFrame([{"class": str(pos_label), "average_precision": ap}]).to_csv(pr_ap_csv, index=False)

            return ClassificationPlotPaths(
                roc_png=(roc_png if roc_png.exists() else None),
                roc_auc_csv=(roc_auc_csv if roc_auc_csv.exists() else None),
                pr_png=(pr_png if pr_png.exists() else None),
                pr_ap_csv=(pr_ap_csv if pr_ap_csv.exists() else None),
            )
        except Exception:
            return ClassificationPlotPaths()

    # Multiclass (OVR)
    if scores_arr.ndim != 2:
        return ClassificationPlotPaths()
    if scores_arr.shape[1] != len(cls):
        return ClassificationPlotPaths()
    if len(cls) > max_classes:
        return ClassificationPlotPaths()

    try:
        y_bin = label_binarize(y_true_arr, classes=cls)
    except Exception:
        return ClassificationPlotPaths()

    auc_rows: List[Dict[str, object]] = []
    ap_rows: List[Dict[str, object]] = []

    try:
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        for i, c in enumerate(cls):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], scores_arr[:, i])
                val_auc = float(auc(fpr, tpr))
                auc_rows.append({"class": str(c), "auc": val_auc})
                ax_roc.plot(fpr, tpr, label=f"{c} (AUC={val_auc:.3f})")
            except Exception:
                continue
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{title_prefix}ROC Curve (OVR)".strip())
        ax_roc.legend(loc="lower right", fontsize=7)
        fig_roc.tight_layout()
        fig_roc.savefig(roc_png, dpi=150)
        plt.close(fig_roc)

        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        for i, c in enumerate(cls):
            try:
                prec, rec, _ = precision_recall_curve(y_bin[:, i], scores_arr[:, i])
                val_ap = float(average_precision_score(y_bin[:, i], scores_arr[:, i]))
                ap_rows.append({"class": str(c), "average_precision": val_ap})
                ax_pr.plot(rec, prec, label=f"{c} (AP={val_ap:.3f})")
            except Exception:
                continue
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{title_prefix}Precision-Recall Curve (OVR)".strip())
        ax_pr.legend(loc="lower left", fontsize=7)
        fig_pr.tight_layout()
        fig_pr.savefig(pr_png, dpi=150)
        plt.close(fig_pr)

        if auc_rows:
            pd.DataFrame(auc_rows).to_csv(roc_auc_csv, index=False)
        if ap_rows:
            pd.DataFrame(ap_rows).to_csv(pr_ap_csv, index=False)
        return ClassificationPlotPaths(
            roc_png=(roc_png if roc_png.exists() else None),
            roc_auc_csv=(roc_auc_csv if roc_auc_csv.exists() else None),
            pr_png=(pr_png if pr_png.exists() else None),
            pr_ap_csv=(pr_ap_csv if pr_ap_csv.exists() else None),
        )
    except Exception:
        return ClassificationPlotPaths()


def iter_existing_paths(paths: Iterable[Optional[Path]]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if isinstance(p, Path) and p.exists():
            out.append(p)
    return out

