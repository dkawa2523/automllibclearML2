"""Hyperparameter search strategies for the AutoML library.

This module provides functions to produce hyperparameter combinations for
each model specification using different search strategies: grid search,
random search and Bayesian optimization via Optuna. These functions
operate on an abstract parameter grid defined in the configuration and
return a list of dictionaries that can be fed into the model factory to
instantiate models.

Random search samples a subset of the full parameter space to reduce
combinatorial explosion, while Bayesian optimization uses Optuna to
iteratively explore promising regions. Bayesian search evaluates a
cross‑validated score for each trial on a simple passthrough
preprocessor; the best hyperparameters are returned for downstream
evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from automl_lib.config.schemas import CVConfig, ModelSpec, OptimizationConfig
from automl_lib.registry.metrics import base_metric_key, build_sklearn_scoring
from .model_factory import prepare_tabpfn_params
from .evaluation import _get_cv_splitter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor


def _flatten_param_grid(params: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Normalize parameter values to lists.

    If a value is a list, it is returned as‑is; otherwise wrapped in a list.
    """
    grid: Dict[str, List[Any]] = {}
    for k, v in params.items():
        if isinstance(v, list):
            grid[k] = v
        else:
            grid[k] = [v]
    return grid


def random_search_combos(param_grid: Dict[str, List[Any]], n_iter: int, random_state: int) -> List[Dict[str, Any]]:
    """Sample random hyperparameter combinations from the full grid.

    Parameters
    ----------
    param_grid : dict
        Mapping of parameter names to lists of possible values.
    n_iter : int
        Number of parameter combinations to sample.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Sampled parameter combinations.
    """
    sampler = ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state)
    return list(sampler)


def _unique_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        ordered.append(val)
    return ordered


def _expand_booster_grid(name: str, param_grid: Dict[str, List[Any]]) -> None:
    """Inject sensible defaults for boosting models when grids are too narrow."""

    lname = name.lower()

    def ensure(key: str, values: List[Any]) -> None:
        current = param_grid.get(key)
        if current is None or len(current) == 0:
            param_grid[key] = _unique_preserve_order(values)
            return
        if len(current) == 1:
            merged = _unique_preserve_order(list(current) + list(values))
            param_grid[key] = merged

    if lname == "lightgbm":
        if "num_leaves" in param_grid and param_grid["num_leaves"]:
            base = max(4, int(param_grid["num_leaves"][0]))
            candidates = [max(4, base // 2), base]
            if base <= 63:
                candidates.append(min(255, base * 2))
            ensure("num_leaves", candidates)
        else:
            ensure("num_leaves", [31, 63])
        if "max_depth" in param_grid and param_grid["max_depth"]:
            bd = param_grid["max_depth"][0]
            if bd in {None, -1}:
                ensure("max_depth", [-1, 7])
            else:
                bd = int(bd)
                ensure("max_depth", [bd, bd + 3])
        else:
            ensure("max_depth", [-1, 7])
        if "learning_rate" in param_grid and param_grid["learning_rate"]:
            base_lr = float(param_grid["learning_rate"][0])
            ensure("learning_rate", [max(1e-3, base_lr * 0.5), base_lr])
        else:
            ensure("learning_rate", [0.05, 0.1])
        ensure("n_estimators", [200, 400])
        ensure("subsample", [0.8, 1.0])
        ensure("colsample_bytree", [1.0])

    elif lname == "xgboost":
        ensure("n_estimators", [200, 400])
        if "max_depth" in param_grid and param_grid["max_depth"]:
            base = int(param_grid["max_depth"][0])
            ensure("max_depth", [max(2, base), base + 2])
        else:
            ensure("max_depth", [3, 5])
        if "learning_rate" in param_grid and param_grid["learning_rate"]:
            base_lr = float(param_grid["learning_rate"][0])
            ensure("learning_rate", [max(1e-3, base_lr * 0.5), base_lr])
        else:
            ensure("learning_rate", [0.03, 0.1])
        ensure("subsample", [0.8, 1.0])
        ensure("colsample_bytree", [1.0])
        ensure("reg_lambda", [1.0, 5.0])
        ensure("min_child_weight", [1, 5])

    elif lname == "catboost":
        if "depth" in param_grid and param_grid["depth"]:
            base = int(param_grid["depth"][0])
            ensure("depth", [max(3, base - 2), base])
        else:
            ensure("depth", [4, 6, 8])
        if "iterations" in param_grid and param_grid["iterations"]:
            base_iter = max(100, int(param_grid["iterations"][0]))
            ensure("iterations", [base_iter, base_iter * 2])
        else:
            ensure("iterations", [300, 600, 900])
        if "learning_rate" in param_grid and param_grid["learning_rate"]:
            base_lr = float(param_grid["learning_rate"][0])
            ensure("learning_rate", [max(1e-3, base_lr * 0.5), base_lr])
        else:
            ensure("learning_rate", [0.03, 0.1])
        ensure("l2_leaf_reg", [1, 3, 7])

def bayesian_optimization_best_params(
    spec: ModelSpec,
    problem_type: str,
    param_grid: Dict[str, List[Any]],
    preprocessors: List[tuple],
    X,
    y,
    cv_config: CVConfig,
    metrics: Iterable[str],
    n_iter: int,
    random_state: int,
    target_standardize: bool,
) -> Dict[str, Any]:
    """Run Optuna Bayesian optimization to find hyperparameters.

    The objective trains a pipeline with a passthrough preprocessor (first in
    the list) using cross‑validation and optimizes the primary metric. The
    best parameters after ``n_iter`` trials are returned.

    If Optuna is not installed, a ValueError is raised.
    """
    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise ImportError("Optuna is required for Bayesian optimization") from exc
    from sklearn.model_selection import cross_val_score
    from .model_factory import _get_model_class

    metrics_list = list(metrics)
    if not metrics_list:
        raise ValueError("At least one metric is required for optimization")
    primary_metric = metrics_list[0]
    scoring_dict = build_sklearn_scoring(problem_type, metrics_list)
    primary_key = base_metric_key(primary_metric, problem_type=problem_type)
    scoring_name = scoring_dict[primary_key]
    # Use the first preprocessor in the list as a baseline for optimization
    if not preprocessors:
        raise ValueError("At least one preprocessor is required for optimization")
    _, preprocessor = preprocessors[0]
    cv = _get_cv_splitter(problem_type, len(y), cv_config, y)
    # Parameter bounds for continuous or integer hyperparameters could be
    # defined here if available. We treat discrete grids and sample from them.

    grid = param_grid
    # Flatten search space keys for Optuna categorical sampling
    def objective(trial: optuna.trial.Trial) -> float:
        params = {}
        for name, values in grid.items():
            params[name] = trial.suggest_categorical(name, values)
        # Instantiate estimator
        try:
            cls = _get_model_class(spec.name, problem_type)
            init_params: Dict[str, Any] = dict(params)
            name_lower = spec.name.lower()
            if name_lower in {"gaussianprocess", "gaussianprocessregressor", "gaussianprocessclassifier"}:
                if "kernel" in init_params and isinstance(init_params["kernel"], str):
                    try:
                        from sklearn.gaussian_process import kernels as gpkernels  # type: ignore

                        init_params["kernel"] = getattr(gpkernels, init_params["kernel"])()
                    except Exception:
                        init_params.pop("kernel", None)
                if problem_type == "regression":
                    if "kernel" not in init_params:
                        from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

                        init_params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                            length_scale=1.0, length_scale_bounds=(1e-2, 1e3)
                        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
                    init_params.setdefault("alpha", 1e-2)
                    init_params.setdefault("normalize_y", True)
                    init_params.setdefault("n_restarts_optimizer", 10)
            if name_lower == "tabpfn":
                prepared = prepare_tabpfn_params(problem_type, init_params)
                if prepared is None:
                    return float("inf")
                init_params = prepared
            module_lower = cls.__module__.lower()
            if "lightgbm" in module_lower:
                init_params.setdefault("force_row_wise", True)
                if problem_type == "regression":
                    init_params.setdefault("objective", "regression_l2")
                if "min_child_samples" not in init_params and "min_data_in_leaf" not in init_params:
                    init_params["min_child_samples"] = max(1, len(y) // 5)
                if "verbose" not in init_params and "verbosity" not in init_params:
                    init_params["verbose"] = -1
            base_params = dict(init_params)
            fallback = base_params.pop("use_fallback_tabpfn", False)
            if fallback:
                from .tabpfn_utils import OfflineTabPFNRegressor

                estimator = OfflineTabPFNRegressor(**base_params)
            else:
                estimator = cls(**base_params)
            if name_lower in {"gaussianprocess", "gaussianprocessregressor"} and problem_type == "regression":
                estimator = TransformedTargetRegressor(
                    regressor=estimator,
                    transformer=StandardScaler(),
                    check_inverse=False,
                )
            if (
                problem_type == "regression"
                and target_standardize
                and not isinstance(estimator, TransformedTargetRegressor)
            ):
                estimator = TransformedTargetRegressor(
                    regressor=estimator,
                    transformer=StandardScaler(),
                    check_inverse=False,
                )
        except Exception:
            return float("inf")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        # Use cross_val_score for efficiency; negative metrics are returned as negative
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring_name)
        mean_score = np.mean(scores)
        # sklearn returns neg_* for losses and positive for scores; minimizing -mean_score works for both.
        return -float(mean_score)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_iter)
    return study.best_params


def generate_param_combinations(
    spec: ModelSpec,
    problem_type: str,
    optimization_config: OptimizationConfig,
    preprocessors: List[tuple],
    X,
    y,
    cv_config: CVConfig,
    metrics: Iterable[str],
    target_standardize: bool = False,
) -> List[Dict[str, Any]]:
    """Generate hyperparameter combinations according to the search strategy.

    Parameters
    ----------
    spec : ModelSpec
        Model specification including name and parameter grid.
    problem_type : str
        'regression' or 'classification'.
    optimization_config : OptimizationConfig
        Specifies the search method and number of iterations.
    preprocessors : list of tuples
        Preprocessors available (for Bayesian search baseline).
    X, y : array‑like
        Training data for evaluating candidate hyperparameters during
        Bayesian optimization.
    cv_config : CVConfig
        Cross‑validation configuration for evaluation.
    metrics : iterable of str
        Metrics to optimize. The first element is treated as the primary metric.

    Returns
    -------
    list of dict
        Hyperparameter dictionaries to instantiate models.
    """
    params = spec.params or {}
    # Flatten values
    param_grid = _flatten_param_grid(params)
    _expand_booster_grid(spec.name, param_grid)
    if not param_grid:
        return [{}]
    method = optimization_config.method.lower()
    n_iter = optimization_config.n_iter
    if method == "grid":
        combos = list(ParameterGrid(param_grid))
    elif method == "random":
        combos = random_search_combos(param_grid, n_iter, random_state=cv_config.random_seed)
    elif method == "bayesian":
        best = bayesian_optimization_best_params(
            spec=spec,
            problem_type=problem_type,
            param_grid=param_grid,
            preprocessors=preprocessors,
            X=X,
            y=y,
            cv_config=cv_config,
            metrics=metrics,
            n_iter=n_iter,
            random_state=cv_config.random_seed,
            target_standardize=target_standardize,
        )
        combos = [best]
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    return combos
