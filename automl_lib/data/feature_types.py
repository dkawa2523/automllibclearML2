from __future__ import annotations

from typing import Dict, List

import pandas as pd


def get_feature_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identify numeric and categorical feature names in the DataFrame.
    (Copied from legacy implementation; kept stable for phase migration.)
    """

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            uniques = X[col].dropna().unique()
            if len(uniques) <= 2 and set(uniques).issubset({0, 1}):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return {"numeric": numeric_cols, "categorical": categorical_cols}

