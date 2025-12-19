"""Compatibility wrapper (deprecated): `automl_lib.clearml` -> `automl_lib.integrations.clearml`."""

from __future__ import annotations

import os
import warnings

flag = str(os.environ.get("AUTO_ML_SUPPRESS_WARNINGS", "1")).strip().lower()
if flag not in {"0", "false", "no", "off"}:
    warnings.warn(
        "`automl_lib.clearml` is deprecated; use `automl_lib.integrations.clearml` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

from automl_lib.integrations.clearml import *  # noqa: F401,F403
