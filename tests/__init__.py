"""Unit tests for automl_lib.

Run:
  ./.venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os

# Unit tests must never require ClearML server connectivity (network is sandboxed).
os.environ.setdefault("CLEARML_OFFLINE_MODE", "1")
os.environ.setdefault("CLEARML_SKIP_PIP_FREEZE", "1")
os.environ.setdefault("AUTO_ML_SKIP_CLEARML_PING", "1")
os.environ.setdefault("AUTO_ML_SKIP_CLEARML_QUEUE_CHECK", "1")
