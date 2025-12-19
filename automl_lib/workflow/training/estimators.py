from __future__ import annotations

from typing import Any, Dict, Tuple

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

from automl_lib.config.schemas import TrainingConfig
from automl_lib.training.model_factory import prepare_tabpfn_params
from automl_lib.training.tabpfn_utils import OfflineTabPFNRegressor


def tune_lightgbm_params(params: Dict[str, Any], train_size: int, problem_type: str) -> Dict[str, Any]:
    """Ensure LightGBM receives sensible defaults for small datasets."""

    tuned = dict(params)
    tuned.setdefault("force_row_wise", True)
    if problem_type == "regression":
        tuned.setdefault("objective", "regression_l2")
    if "min_child_samples" not in tuned and "min_data_in_leaf" not in tuned:
        if train_size > 0:
            candidate = max(1, train_size // 5)
            tuned["min_child_samples"] = candidate
    return tuned


def instantiate_estimator(
    model_name: str,
    estimator_cls,
    init_params: Dict[str, Any],
):
    """Instantiate an estimator, handling TabPFN fallbacks when necessary."""

    params_for_init = dict(init_params)
    fallback = params_for_init.pop("use_fallback_tabpfn", False)
    if fallback:
        return OfflineTabPFNRegressor(**params_for_init)
    return estimator_cls(**params_for_init)


def maybe_wrap_with_target_scaler(
    estimator: Any,
    cfg: TrainingConfig,
    problem_type: str,
):
    """Optionally wrap estimator with target standardization for regression."""

    if problem_type.lower() != "regression":
        return estimator
    if not getattr(cfg.preprocessing, "target_standardize", False):
        return estimator
    if isinstance(estimator, TransformedTargetRegressor):
        return estimator
    return TransformedTargetRegressor(
        regressor=estimator,
        transformer=StandardScaler(),
        check_inverse=False,
    )


def build_estimator_with_defaults(
    model_name: str,
    estimator_cls,
    init_params: Dict[str, Any] | None,
    problem_type: str,
    cfg: TrainingConfig,
    train_size: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Instantiate an estimator while applying evaluation-time defaults."""

    params: Dict[str, Any] = dict(init_params or {})
    name_lower = model_name.lower()
    module_name_lower = estimator_cls.__module__.lower()

    if name_lower in {"gaussianprocess", "gaussianprocessregressor", "gaussianprocessclassifier"}:
        if "kernel" in params and isinstance(params["kernel"], str):
            kernel_str = params["kernel"]
            try:
                from sklearn.gaussian_process import kernels as gpkernels  # type: ignore

                kernel_cls = getattr(gpkernels, kernel_str)
                params["kernel"] = kernel_cls()
            except Exception:
                pass
        if problem_type == "regression":
            if "kernel" not in params:
                from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

                params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=1.0,
                    length_scale_bounds=(1e-2, 1e3),
                ) + WhiteKernel(
                    noise_level=1.0,
                    noise_level_bounds=(1e-5, 1e5),
                )
            params.setdefault("alpha", 1e-2)
            params.setdefault("normalize_y", True)
            params.setdefault("n_restarts_optimizer", 10)

    if "catboost" in module_name_lower:
        params.setdefault("verbose", 0)
        params.setdefault("random_seed", cfg.data.random_seed)
        params.setdefault("allow_writing_files", False)

    if "lightgbm" in module_name_lower:
        params = tune_lightgbm_params(params, train_size, problem_type)
        if "verbose" not in params and "verbosity" not in params:
            params["verbose"] = -1
        params.setdefault("random_state", cfg.data.random_seed)

    if "xgboost" in module_name_lower:
        params.setdefault("random_state", cfg.data.random_seed)
        params.setdefault("n_jobs", -1)

    if "pytorch_tabnet" in module_name_lower:
        params.setdefault("device_name", "cpu")
        params.setdefault("verbose", 0)

    if name_lower == "mlp":
        params.setdefault("random_state", cfg.data.random_seed)
        if problem_type == "regression":
            params.setdefault("max_iter", 2000)
            params.setdefault("early_stopping", True)
            params.setdefault("n_iter_no_change", 20)
            params.setdefault("validation_fraction", 0.1)

    if name_lower == "tabpfn":
        tabpfn_params = prepare_tabpfn_params(problem_type, params)
        if tabpfn_params is None:
            raise ValueError("TabPFN weights are unavailable")
        params = tabpfn_params

    if "gaussian_process" in module_name_lower and problem_type == "regression":
        if "kernel" not in params:
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel  # type: ignore

            params["kernel"] = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0,
                length_scale_bounds=(1e-2, 1e3),
            ) + WhiteKernel(
                noise_level=1.0,
                noise_level_bounds=(1e-5, 1e5),
            )
        params.setdefault("alpha", 1e-2)
        params.setdefault("normalize_y", True)
        params.setdefault("n_restarts_optimizer", 10)

    estimator = instantiate_estimator(model_name, estimator_cls, params)

    if name_lower in {"gaussianprocess", "gaussianprocessregressor"} and problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            check_inverse=False,
        )

    if name_lower in {"tabnet", "mlp"} and problem_type == "regression":
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            check_inverse=False,
        )

    estimator = maybe_wrap_with_target_scaler(estimator, cfg, problem_type)

    return estimator, params

