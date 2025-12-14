from __future__ import annotations

from typing import Dict, Optional


def report_metric_scalars(
    task_mgr,
    *,
    train_metrics: Dict[str, Optional[float]],
    test_metrics: Optional[Dict[str, Optional[float]]] = None,
    iteration: int = 0,
    train_prefix: str = "metric",
    test_prefix: str = "metric_test",
) -> None:
    """Report standardized metric scalars (title-based) to ClearML.

    Example titles:
      - metric/r2 (series=value)
      - metric_test/r2 (series=value)
    """

    if not task_mgr:
        return

    def _emit(prefix: str, metrics: Dict[str, Optional[float]]) -> None:
        for name, val in metrics.items():
            key = str(name).strip().lower()
            if not key:
                continue
            if val is None:
                continue
            try:
                task_mgr.report_scalar(f"{prefix}/{key}", "value", float(val), iteration=iteration)
            except Exception:
                pass

    if isinstance(train_metrics, dict):
        _emit(str(train_prefix), train_metrics)
    if isinstance(test_metrics, dict) and test_metrics:
        _emit(str(test_prefix), test_metrics)

