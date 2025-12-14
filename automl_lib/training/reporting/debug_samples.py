from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


def collect_plot_paths(output_dir: Path) -> List[Path]:
    """Collect known plot/artifact paths under output_dir (best-effort)."""

    paths: List[Path] = []
    candidates: Sequence[Iterable[Path]] = [
        output_dir.glob("*.png"),
        (output_dir / "scatter_plots").glob("*.png"),
        (output_dir / "residual_scatter").glob("*.png"),
        (output_dir / "residual_hist").glob("*.png"),
        (output_dir / "interpolation_space").glob("*.png"),
        (output_dir / "confusion_matrices").glob("*.png"),
        (output_dir / "roc_curves").glob("*.png"),
        (output_dir / "pr_curves").glob("*.png"),
    ]
    for it in candidates:
        try:
            for p in it:
                if p.exists():
                    paths.append(p)
        except Exception:
            continue

    # de-duplicate
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def build_plot_artifacts_table(output_dir: Path) -> pd.DataFrame:
    rows = []
    for p in collect_plot_paths(output_dir):
        rows.append({"file": p.name, "path": str(p)})
    return pd.DataFrame(rows) if rows else pd.DataFrame()

