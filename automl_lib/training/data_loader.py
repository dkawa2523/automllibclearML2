"""Data loading utilities for the AutoML library.

This module contains helper functions to read CSV files into pandas DataFrames,
infer the problem type (classification or regression) based on the target
column, and split the data into training and test sets if requested.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

from automl_lib.config.schemas import DataSettings


def load_dataset(config: DataSettings) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset from a CSV file.

    Parameters
    ----------
    config : DataConfig
        The data configuration specifying the CSV path and columns to use.

    Returns
    -------
    X : pd.DataFrame
        DataFrame containing the input features.
    y : pd.Series
        Series containing the target variable.
    """
    if not config.csv_path:
        raise ValueError("data.csv_path is required (or ensure the ClearML dataset resolves to a local CSV)")

    csv_path = Path(config.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' does not exist")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset at '{csv_path}' is empty")

    # Determine target column
    target_col = config.target_column or df.columns[-1]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset")

    # Determine feature columns
    feature_cols: List[str]
    if config.feature_columns:
        # Validate provided features
        missing = [c for c in config.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Specified feature columns not found: {missing}")
        feature_cols = config.feature_columns
    else:
        feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns specified or left after excluding the target")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def infer_problem_type(y: pd.Series, explicit_type: Optional[str] = None) -> str:
    """Infer the problem type based on the target series.

    Parameters
    ----------
    y : pd.Series
        Target variable values.
    explicit_type : str, optional
        User-provided override. Must be either 'regression' or 'classification'.

    Returns
    -------
    str
        'regression' or 'classification'.
    """
    if explicit_type:
        explicit_type = explicit_type.lower()
        if explicit_type not in {"classification", "regression"}:
            raise ValueError("problem_type must be either 'classification' or 'regression'")
        return explicit_type

    # Determine automatically
    if pd.api.types.is_numeric_dtype(y):
        # If the number of unique values is small relative to sample size, treat as classification.
        # Otherwise regression.
        unique = y.nunique(dropna=False)
        if unique <= max(2, int(len(y) * 0.1)):
            return "classification"
        return "regression"
    else:
        return "classification"


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, random_seed: int, shuffle: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and test sets.

    If test_size is 0 or None, the function returns the original X and y as both
    train and test sets.

    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Target.
    test_size : float
        Proportion of the data to use for testing.
    random_seed : int
        Random seed for reproducibility.
    shuffle : bool, default True
        Whether to shuffle before splitting. If the data has a temporal order,
        set to False.

    Returns
    -------
    X_train, X_test, y_train, y_test
        The split data. When test_size is 0, X_test and y_test are copies of
        X_train and y_train.
    """
    if test_size and test_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, shuffle=shuffle
        )
    else:
        # Use all data as training; for evaluation we rely on cross‑validation
        X_train, X_test, y_train, y_test = X.copy(), X.copy(), y.copy(), y.copy()
    return X_train, X_test, y_train, y_test


def get_feature_types(X: pd.DataFrame) -> Dict[str, List[str]]:
    """Identify numeric and categorical feature names in the DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Input features.

    Returns
    -------
    dict
        A dictionary with keys 'numeric' and 'categorical', each containing a
        list of column names of that type.
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # Check for binary indicator: treat as categorical if only two unique values
            uniques = X[col].dropna().unique()
            if len(uniques) <= 2 and set(uniques).issubset({0, 1}):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return {"numeric": numeric_cols, "categorical": categorical_cols}
