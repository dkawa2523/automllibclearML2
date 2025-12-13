"""Example preprocessor plugin.

To enable:
  preprocessing:
    plugins: ["automl_lib.plugins.example_preprocessors"]
    numeric_pipeline_steps:
      - name: log1p
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from automl_lib.registry.preprocessors import register_preprocessor


register_preprocessor(
    "log1p",
    lambda **kwargs: FunctionTransformer(np.log1p, validate=False, **kwargs),
    aliases=["log_1p", "log1p_transform"],
)

register_preprocessor(
    "power",
    lambda **kwargs: PowerTransformer(**kwargs),
    aliases=["powertransformer", "power_transformer"],
)

