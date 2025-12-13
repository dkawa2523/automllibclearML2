"""Example metrics plugin.

To enable:
  evaluation:
    plugins: ["automl_lib.plugins.example_metrics"]
    regression_metrics: ["mae", "rmse", "r2", "mape", "smape"]
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import make_scorer

from automl_lib.registry.metrics import register_metric


register_metric(
    "mape",
    sklearn_scoring="neg_mean_absolute_percentage_error",
    kind="regression",
    is_loss=True,
    aliases=["mean_absolute_percentage_error"],
)


def _smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, 1e-12)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


register_metric(
    "smape",
    sklearn_scoring=make_scorer(_smape, greater_is_better=False),
    kind="regression",
    is_loss=True,
    aliases=["symmetric_mape"],
)

register_metric(
    "balanced_accuracy",
    sklearn_scoring="balanced_accuracy",
    kind="classification",
    is_loss=False,
    aliases=["bacc", "bal_acc"],
)
