from __future__ import annotations

import sys
from importlib import reload as _reload

from .runtime import NotebookContext, bootstrap
from .steps import (
    data_registration,
    inference_optimize,
    inference_single,
    pipeline_training,
)

__all__ = [
    "NotebookContext",
    "bootstrap",
    "data_registration",
    "pipeline_training",
    "inference_single",
    "inference_optimize",
    "reload",
]


def reload():
    """Reload Notebook_run modules in a running notebook kernel (dev convenience)."""

    mod_runtime = sys.modules.get("Notebook_run.runtime")
    if mod_runtime is not None:
        _reload(mod_runtime)
    mod_steps = sys.modules.get("Notebook_run.steps")
    if mod_steps is not None:
        _reload(mod_steps)

    # Re-bind exports to the reloaded modules.
    runtime = sys.modules.get("Notebook_run.runtime")
    steps = sys.modules.get("Notebook_run.steps")
    if runtime is not None:
        globals()["NotebookContext"] = getattr(runtime, "NotebookContext", NotebookContext)
        globals()["bootstrap"] = getattr(runtime, "bootstrap", bootstrap)
    if steps is not None:
        globals()["data_registration"] = getattr(steps, "data_registration", data_registration)
        globals()["pipeline_training"] = getattr(steps, "pipeline_training", pipeline_training)
        globals()["inference_single"] = getattr(steps, "inference_single", inference_single)
        globals()["inference_optimize"] = getattr(steps, "inference_optimize", inference_optimize)
    return sys.modules.get(__name__)
