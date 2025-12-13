"""Ensemble model utilities for the AutoML library.

This module contains helper functions to build stacking and voting
ensembles. An ensemble combines multiple base estimators to often
achieve better predictive performance than any single model. The
functions here wrap scikit‑learn's StackingRegressor/Classifier and
VotingRegressor/Classifier and accept a preprocessor so that all
estimators receive identical feature transformations.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor, StackingClassifier, VotingRegressor, VotingClassifier


def build_stacking(
    preprocessor: object,
    estimators: List[Tuple[str, object]],
    final_estimator: object,
    problem_type: str,
) -> Pipeline:
    """Construct a stacking ensemble wrapped in a pipeline.

    Parameters
    ----------
    preprocessor : transformer
        Preprocessing transformer (e.g., ColumnTransformer).
    estimators : list of (str, estimator)
        Base estimators included in the stack.
    final_estimator : estimator
        Model used to aggregate base estimators' predictions.
    problem_type : str
        'regression' or 'classification'. Determines which stacking class to use.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with preprocessor and stacking estimator.
    """
    if problem_type.lower() == "regression":
        stack = StackingRegressor(estimators=estimators, final_estimator=final_estimator, passthrough=False)
    else:
        stack = StackingClassifier(estimators=estimators, final_estimator=final_estimator, passthrough=False)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", stack),
    ])


def build_voting(
    preprocessor: object,
    estimators: List[Tuple[str, object]],
    voting: str,
    problem_type: str,
) -> Pipeline:
    """Construct a voting ensemble wrapped in a pipeline.

    Parameters
    ----------
    preprocessor : transformer
        Preprocessing transformer.
    estimators : list of (str, estimator)
        Estimators included in the vote.
    voting : str
        Voting scheme: 'hard' or 'soft' for classifiers; 'mean' for regressors.
    problem_type : str
        'regression' or 'classification'.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with voting ensemble.
    """
    if problem_type.lower() == "regression":
        voter = VotingRegressor(estimators=estimators)
    else:
        voter = VotingClassifier(estimators=estimators, voting=voting)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", voter),
    ])