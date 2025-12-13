from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple, Literal

try:
    from pydantic import (  # type: ignore
        AliasChoices,
        BaseModel,
        ConfigDict,
        Field,
        field_validator,
        model_validator,
    )
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("pydantic is required: pip install pydantic") from exc


class _StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClearMLAgentsConfig(_StrictBaseModel):
    data_registration: Optional[str] = None
    data_editing: Optional[str] = None
    preprocessing: Optional[str] = None
    training: Optional[str] = None
    inference: Optional[str] = None
    optimization: Optional[str] = None
    pipeline: Optional[str] = None
    comparison: Optional[str] = None


class ClearMLSettings(_StrictBaseModel):
    enabled: bool = False
    project_name: Optional[str] = None
    dataset_project: Optional[str] = None
    base_output_uri: Optional[str] = None
    task_name: Optional[str] = None
    queue: Optional[str] = None
    services_queue: Optional[str] = None
    run_tasks_locally: bool = Field(default=True, validation_alias=AliasChoices("run_tasks_locally", "run_locally"))
    run_pipeline_locally: bool = True
    tags: List[str] = Field(default_factory=list)

    raw_dataset_id: Optional[str] = None
    edited_dataset_id: Optional[str] = None
    preprocessed_dataset_id: Optional[str] = None
    register_raw_dataset: bool = False

    enable_data_editing: bool = False
    enable_preprocessing: bool = False
    enable_training: bool = True
    enable_inference: bool = False
    enable_optimization: bool = False
    enable_pipeline: bool = False
    enable_comparison: bool = False

    comparison_agent: Optional[str] = None
    comparison_metrics: List[str] = Field(default_factory=list)
    comparison_task_name: Optional[str] = None

    agents: ClearMLAgentsConfig = Field(default_factory=ClearMLAgentsConfig)


class DataSettings(_StrictBaseModel):
    dataset_id: Optional[str] = None
    csv_path: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: Optional[List[str]] = None
    problem_type: Optional[str] = None
    test_size: float = 0.0
    random_seed: int = 42

    @field_validator("test_size")
    @classmethod
    def _check_test_size(cls, value: float) -> float:
        if value < 0.0 or value >= 1.0:
            raise ValueError("data.test_size must be in [0,1)")
        return value

    @field_validator("problem_type")
    @classmethod
    def _normalize_problem_type(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        lowered = str(value).strip().lower()
        if lowered in {"", "none", "null"}:
            return None
        if lowered not in {"regression", "classification"}:
            raise ValueError("data.problem_type must be 'regression' or 'classification' (or null)")
        return lowered


class EditingSettings(_StrictBaseModel):
    enable: bool = True
    output_path: Optional[str] = None
    drop_columns: List[str] = Field(default_factory=list)
    rename_columns: Dict[str, str] = Field(default_factory=dict)
    fillna: Optional[Any] = None
    fillna_columns: Dict[str, Any] = Field(default_factory=dict)
    query: Optional[str] = None
    clip_values: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


def _normalize_none_like(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return lowered


class PreprocessSettings(_StrictBaseModel):
    class StepSpec(_StrictBaseModel):
        name: str
        params: Dict[str, Any] = Field(default_factory=dict)

        @field_validator("name")
        @classmethod
        def _check_name(cls, value: str) -> str:
            if not str(value).strip():
                raise ValueError("preprocessing step name must be non-empty")
            return str(value).strip()

    # Optional plugin modules to import (modules should call register_preprocessor()).
    plugins: List[str] = Field(default_factory=list)

    numeric_imputation: List[Optional[str]] = Field(default_factory=lambda: ["mean"])
    categorical_imputation: List[Optional[str]] = Field(default_factory=lambda: ["most_frequent"])
    scaling: List[Optional[str]] = Field(default_factory=lambda: ["standard"])
    categorical_encoding: List[Optional[str]] = Field(default_factory=lambda: ["onehot"])
    polynomial_degree: int | bool | None = False
    target_standardize: bool = False
    numeric_pipeline_steps: List[StepSpec] = Field(default_factory=list)
    categorical_pipeline_steps: List[StepSpec] = Field(default_factory=list)

    @field_validator("numeric_imputation", mode="before")
    @classmethod
    def _validate_numeric_imputation(cls, value: Any) -> List[Optional[str]]:
        if value is None:
            return [None]
        items = value if isinstance(value, list) else [value]
        allowed = {"mean", "median", "most_frequent"}
        normalized: List[Optional[str]] = []
        for item in items:
            cand = _normalize_none_like(item)
            if cand is None:
                normalized.append(None)
                continue
            if cand not in allowed:
                raise ValueError(f"preprocessing.numeric_imputation must be one of {sorted(allowed)} or null")
            normalized.append(cand)
        return normalized

    @field_validator("categorical_imputation", mode="before")
    @classmethod
    def _validate_categorical_imputation(cls, value: Any) -> List[Optional[str]]:
        if value is None:
            return [None]
        items = value if isinstance(value, list) else [value]
        allowed = {"most_frequent"}
        normalized: List[Optional[str]] = []
        for item in items:
            cand = _normalize_none_like(item)
            if cand is None:
                normalized.append(None)
                continue
            if cand not in allowed:
                raise ValueError(f"preprocessing.categorical_imputation must be one of {sorted(allowed)} or null")
            normalized.append(cand)
        return normalized

    @field_validator("scaling", mode="before")
    @classmethod
    def _validate_scaling(cls, value: Any) -> List[Optional[str]]:
        if value is None:
            return [None]
        items = value if isinstance(value, list) else [value]
        normalized: List[Optional[str]] = []
        for item in items:
            cand = _normalize_none_like(item)
            if cand is None:
                normalized.append(None)
                continue
            normalized.append(cand)
        return normalized

    @field_validator("categorical_encoding", mode="before")
    @classmethod
    def _validate_categorical_encoding(cls, value: Any) -> List[Optional[str]]:
        if value is None:
            return [None]
        items = value if isinstance(value, list) else [value]
        normalized: List[Optional[str]] = []
        for item in items:
            cand = _normalize_none_like(item)
            if cand is None:
                normalized.append(None)
                continue
            normalized.append(cand)
        return normalized

    @field_validator("polynomial_degree", mode="before")
    @classmethod
    def _normalize_polynomial_degree(cls, value: Any) -> int | bool | None:
        if value is None:
            return False
        if isinstance(value, bool):
            if value is True:
                raise ValueError("preprocessing.polynomial_degree must be false or an integer >= 2")
            return False
        if isinstance(value, int):
            return value if value >= 2 else False
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"false", "none", "null", ""}:
                return False
            try:
                parsed = int(lowered)
            except ValueError as exc:
                raise ValueError("preprocessing.polynomial_degree must be false or an integer >= 2") from exc
            return parsed if parsed >= 2 else False
        raise ValueError("preprocessing.polynomial_degree must be false or an integer >= 2")


class ModelSpec(_StrictBaseModel):
    name: str
    enable: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_params(self) -> "ModelSpec":
        if "hidden_layer_sizes" in self.params:
            self.params["hidden_layer_sizes"] = _normalize_hidden_layer_sizes(self.params["hidden_layer_sizes"])
        return self


class EnsembleConfig(_StrictBaseModel):
    enable: bool = False
    estimators: List[str] = Field(default_factory=list)
    final_estimator: Optional[str] = None
    voting: Optional[str] = None


class EnsembleGroup(_StrictBaseModel):
    stacking: EnsembleConfig = Field(default_factory=EnsembleConfig)
    voting: EnsembleConfig = Field(default_factory=EnsembleConfig)


class CVConfig(_StrictBaseModel):
    n_folds: Optional[int] = None
    shuffle: bool = True
    random_seed: int = 42

    @field_validator("n_folds")
    @classmethod
    def _check_n_folds(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 2:
            raise ValueError("cross_validation.n_folds must be >= 2 (or null)")
        return value


class OutputSettings(_StrictBaseModel):
    output_dir: str = "outputs/train"
    save_models: bool = True
    generate_plots: bool = True
    results_csv: str = "results_summary.csv"


class PreprocessingOutputSettings(_StrictBaseModel):
    output_dir: str = "outputs/preprocessing"


class ComparisonOutputSettings(_StrictBaseModel):
    output_dir: str = "outputs/comparison"


class ComparisonCompositeScoreConfig(_StrictBaseModel):
    enabled: bool = True
    metrics: List[str] = Field(default_factory=list)
    weights: Dict[str, float] = Field(default_factory=dict)
    require_all_metrics: bool = False

    @field_validator("metrics", mode="before")
    @classmethod
    def _normalize_metrics(cls, value: Any) -> List[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized: List[str] = []
        for item in items:
            cand = _normalize_none_like(item)
            if not cand:
                continue
            normalized.append(str(cand).strip().lower())
        return normalized

    @field_validator("weights", mode="before")
    @classmethod
    def _normalize_weights(cls, value: Any) -> Dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("ranking.composite.weights must be a mapping")
        normalized: Dict[str, float] = {}
        for k, v in value.items():
            key = _normalize_none_like(k)
            if not key:
                continue
            try:
                weight = float(v)
            except Exception as exc:
                raise ValueError(f"ranking.composite.weights[{k!r}] must be a number") from exc
            if weight < 0:
                raise ValueError(f"ranking.composite.weights[{k!r}] must be >= 0")
            if weight == 0:
                continue
            normalized[str(key).strip().lower()] = weight
        return normalized


class ComparisonRankingSettings(_StrictBaseModel):
    metrics: List[str] = Field(default_factory=list)
    primary_metric: Optional[str] = None
    goal: Optional[Literal["min", "max"]] = None
    top_k: Optional[int] = None
    composite: ComparisonCompositeScoreConfig = Field(default_factory=ComparisonCompositeScoreConfig)

    @field_validator("metrics", mode="before")
    @classmethod
    def _normalize_metrics(cls, value: Any) -> List[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized: List[str] = []
        for item in items:
            cand = _normalize_none_like(item)
            if not cand:
                continue
            normalized.append(str(cand).strip().lower())
        return normalized

    @field_validator("primary_metric")
    @classmethod
    def _normalize_primary_metric(cls, value: Optional[str]) -> Optional[str]:
        cand = _normalize_none_like(value)
        if not cand:
            return None
        return str(cand).strip().lower()

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        if value < 1:
            raise ValueError("ranking.top_k must be >= 1 (or null)")
        return value


def _validate_preprocessing_registry(settings: PreprocessSettings) -> None:
    """Validate preprocessing plugin modules / registered preprocessors."""

    import importlib

    from automl_lib.registry.preprocessors import ensure_default_preprocessors_registered, get_preprocessor

    ensure_default_preprocessors_registered()
    for module_name in getattr(settings, "plugins", []) or []:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise ValueError(f"Failed to import preprocessing plugin module '{module_name}': {exc}") from exc

    for scaling_name in getattr(settings, "scaling", []) or []:
        if scaling_name is None:
            continue
        try:
            get_preprocessor(str(scaling_name))
        except Exception as exc:
            raise ValueError(f"Unknown preprocessing.scaling: {scaling_name!r}") from exc

    for enc_name in getattr(settings, "categorical_encoding", []) or []:
        if enc_name is None:
            continue
        try:
            get_preprocessor(str(enc_name))
        except Exception as exc:
            raise ValueError(f"Unknown preprocessing.categorical_encoding: {enc_name!r}") from exc

    for step in getattr(settings, "numeric_pipeline_steps", []) or []:
        try:
            get_preprocessor(str(step.name))
        except Exception as exc:
            raise ValueError(f"Unknown preprocessing.numeric_pipeline_steps[].name: {step.name!r}") from exc

    for step in getattr(settings, "categorical_pipeline_steps", []) or []:
        try:
            get_preprocessor(str(step.name))
        except Exception as exc:
            raise ValueError(f"Unknown preprocessing.categorical_pipeline_steps[].name: {step.name!r}") from exc


class EvaluationConfig(_StrictBaseModel):
    plugins: List[str] = Field(default_factory=list)
    regression_metrics: List[str] = Field(default_factory=lambda: ["mae", "rmse", "r2"])
    classification_metrics: List[str] = Field(default_factory=lambda: ["accuracy", "f1_macro", "roc_auc_ovr"])
    primary_metric: Optional[str] = None

    @field_validator("regression_metrics", mode="before")
    @classmethod
    def _validate_regression_metrics(cls, value: Any) -> List[str]:
        if value is None:
            return ["mae", "rmse", "r2"]
        items = value if isinstance(value, list) else [value]
        normalized: List[str] = []
        for item in items:
            cand = _normalize_none_like(item)
            if cand is None:
                continue
            normalized.append(cand)
        return normalized

    @field_validator("classification_metrics", mode="before")
    @classmethod
    def _validate_classification_metrics(cls, value: Any) -> List[str]:
        if value is None:
            return ["accuracy", "f1_macro", "roc_auc_ovr"]
        items = value if isinstance(value, list) else [value]
        normalized: List[str] = []
        for item in items:
            cand = _normalize_none_like(item)
            if cand is None:
                continue
            normalized.append(cand)
        return normalized

    @field_validator("primary_metric")
    @classmethod
    def _validate_primary_metric(cls, value: Optional[str], info) -> Optional[str]:
        if value is None:
            return None
        lowered = str(value).strip().lower()
        if lowered in {"", "none", "null"}:
            return None
        return lowered


class OptimizationConfig(_StrictBaseModel):
    method: Literal["grid", "random", "bayesian"] = "grid"
    n_iter: int = 10

    @field_validator("n_iter")
    @classmethod
    def _check_n_iter(cls, value: int) -> int:
        if value < 1:
            raise ValueError("optimization.n_iter must be >= 1")
        return value


class InterpretationConfig(_StrictBaseModel):
    compute_feature_importance: bool = False
    compute_shap: bool = False


class VisualizationConfig(_StrictBaseModel):
    predicted_vs_actual: bool = True
    residual_scatter: bool = False
    residual_hist: bool = False
    feature_importance: bool = False
    shap_summary: bool = False
    comparative_heatmap: bool = False


class TrainingConfig(_StrictBaseModel):
    data: DataSettings
    preprocessing: PreprocessSettings = Field(default_factory=PreprocessSettings)
    models: List[ModelSpec] = Field(default_factory=list)
    ensembles: EnsembleGroup = Field(default_factory=EnsembleGroup)
    cross_validation: CVConfig = Field(default_factory=CVConfig)
    output: OutputSettings = Field(default_factory=OutputSettings)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    interpretation: InterpretationConfig = Field(default_factory=InterpretationConfig)
    visualizations: VisualizationConfig = Field(default_factory=VisualizationConfig)
    clearml: Optional[ClearMLSettings] = None

    @model_validator(mode="after")
    def _validate_training_config(self) -> "TrainingConfig":
        # Validate preprocessing plugins/registry names early (typo detection).
        import importlib

        from automl_lib.registry.metrics import ensure_default_metrics_registered, get_metric_spec
        from automl_lib.registry.models import ensure_default_models_registered, resolve_model_class

        _validate_preprocessing_registry(self.preprocessing)

        ensure_default_metrics_registered()
        for module_name in getattr(self.evaluation, "plugins", []) or []:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                raise ValueError(f"Failed to import evaluation plugin module '{module_name}': {exc}") from exc

        for metric_name in getattr(self.evaluation, "regression_metrics", []) or []:
            try:
                get_metric_spec(str(metric_name), problem_type="regression")
            except Exception as exc:
                raise ValueError(f"Unknown evaluation.regression_metrics: {metric_name!r}") from exc

        for metric_name in getattr(self.evaluation, "classification_metrics", []) or []:
            try:
                get_metric_spec(str(metric_name), problem_type="classification")
            except Exception as exc:
                raise ValueError(f"Unknown evaluation.classification_metrics: {metric_name!r}") from exc

        if self.evaluation.primary_metric:
            metric = str(self.evaluation.primary_metric)
            ok = False
            for ptype in ["regression", "classification"]:
                try:
                    get_metric_spec(metric, problem_type=ptype)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                raise ValueError(f"Unknown evaluation.primary_metric: {self.evaluation.primary_metric!r}")

        if not (self.data.dataset_id or self.data.csv_path):
            raise ValueError("Either data.dataset_id or data.csv_path must be provided")
        if not self.models:
            raise ValueError("models must be a non-empty list")
        if not any(m.enable for m in self.models):
            raise ValueError("At least one models[].enable must be true")

        # Validate model names early (typo detection / missing optional deps).
        ensure_default_models_registered()
        desired_problem_type = self.data.problem_type
        unknown: List[str] = []
        for model in self.models:
            if not getattr(model, "enable", True):
                continue
            name = str(model.name)
            ok = False
            if desired_problem_type:
                try:
                    resolve_model_class(name, str(desired_problem_type))
                    ok = True
                except Exception:
                    ok = False
            else:
                for ptype in ["regression", "classification"]:
                    try:
                        resolve_model_class(name, ptype)
                        ok = True
                        break
                    except Exception:
                        continue
            if not ok:
                unknown.append(name)
        if unknown:
            raise ValueError(
                "Unknown/unsupported models in config.models (check typos, install optional libraries, "
                "or set models[].enable=false): " + ", ".join(repr(x) for x in unknown)
            )

        return self


class PreprocessingConfig(_StrictBaseModel):
    data: DataSettings
    preprocessing: PreprocessSettings = Field(default_factory=PreprocessSettings)
    output: PreprocessingOutputSettings = Field(default_factory=PreprocessingOutputSettings)
    clearml: Optional[ClearMLSettings] = None

    @model_validator(mode="after")
    def _validate_preprocessing_config(self) -> "PreprocessingConfig":
        _validate_preprocessing_registry(self.preprocessing)
        return self


class ComparisonConfig(_StrictBaseModel):
    output: ComparisonOutputSettings = Field(default_factory=ComparisonOutputSettings)
    clearml: Optional[ClearMLSettings] = None
    ranking: ComparisonRankingSettings = Field(default_factory=ComparisonRankingSettings)


class InferenceModelSpec(_StrictBaseModel):
    name: str
    enable: bool = True
    model_id: Optional[str] = None


class InferenceVariableSpec(_StrictBaseModel):
    name: str
    type: Literal["int", "float", "categorical", "bool", "str"]
    method: Literal["range", "values", "fixed"]
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    values: Optional[List[Any]] = None
    value: Optional[Any] = None


class InferenceInputConfig(_StrictBaseModel):
    mode: Literal["csv", "params"] = "csv"
    variables: Optional[List[InferenceVariableSpec]] = None
    params_path: Optional[str] = None
    csv_path: Optional[str] = None

    @model_validator(mode="after")
    def _validate_input(self) -> "InferenceInputConfig":
        if self.mode == "csv":
            if not self.csv_path:
                raise ValueError("input.csv_path is required when input.mode=csv")
        if self.mode == "params":
            if not self.variables and not self.params_path:
                raise ValueError("input.variables or input.params_path is required when input.mode=params")
        return self


class InferenceSearchConfig(_StrictBaseModel):
    method: Literal["grid", "random", "tpe", "cmaes"] = "grid"
    n_trials: int = 20
    goal: Literal["max", "min"] = "max"

    @field_validator("n_trials")
    @classmethod
    def _check_trials(cls, value: int) -> int:
        if value < 1:
            raise ValueError("search.n_trials must be >= 1")
        return value


class InferenceConfig(_StrictBaseModel):
    model_dir: str
    models: List[InferenceModelSpec] = Field(default_factory=list)
    clearml: Optional[ClearMLSettings] = None
    input: InferenceInputConfig
    search: InferenceSearchConfig = Field(default_factory=InferenceSearchConfig)
    output_dir: str = "outputs/inference"

    @model_validator(mode="after")
    def _validate_inference(self) -> "InferenceConfig":
        if not self.models:
            raise ValueError("models must be a non-empty list")
        if not any(m.enable for m in self.models):
            raise ValueError("At least one models[].enable must be true")
        return self


class DataRegistrationConfig(_StrictBaseModel):
    data: DataSettings
    clearml: Optional[ClearMLSettings] = None

    @model_validator(mode="after")
    def _validate_data_registration(self) -> "DataRegistrationConfig":
        if not self.data.csv_path:
            raise ValueError("data.csv_path is required for data_registration")
        return self


class DataEditingConfig(_StrictBaseModel):
    data: DataSettings
    editing: EditingSettings
    clearml: Optional[ClearMLSettings] = None

    @model_validator(mode="after")
    def _validate_data_editing(self) -> "DataEditingConfig":
        if not self.data.csv_path and not self.data.dataset_id:
            raise ValueError("data.csv_path or data.dataset_id is required for data_editing")
        return self


def _coerce_hidden_layer_tuple(value: Any) -> Tuple[int, ...]:
    """Convert a raw hidden_layer_sizes entry into a tuple of ints."""

    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        return (int(value),)
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"hidden_layer_sizes entries must be integers, got {value!r}")
        return (int(value),)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("hidden_layer_sizes string entries cannot be empty")
        try:
            parsed = ast.literal_eval(stripped)
        except Exception:
            digits = re.findall(r"-?\\d+", stripped)
            if not digits:
                raise ValueError(f"Could not parse hidden_layer_sizes value '{value}'")
            return _coerce_hidden_layer_tuple(tuple(int(d) for d in digits))
        return _coerce_hidden_layer_tuple(parsed)
    else:
        raise ValueError(f"Unsupported hidden_layer_sizes value type: {type(value)}")

    if not items:
        raise ValueError("hidden_layer_sizes tuples cannot be empty")

    coerced: List[int] = []
    for item in items:
        if isinstance(item, float):
            if not item.is_integer():
                raise ValueError(f"hidden_layer_sizes entries must be integers, got {item!r}")
            coerced.append(int(item))
        elif isinstance(item, int) and not isinstance(item, bool):
            coerced.append(int(item))
        elif isinstance(item, str):
            if not item.strip():
                raise ValueError("hidden_layer_sizes string entries cannot be empty")
            try:
                coerced.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Could not parse hidden_layer_sizes value '{item}'") from exc
        else:
            raise ValueError(f"Unsupported hidden_layer_sizes element type: {type(item)}")
    return tuple(coerced)


def _merge_tokenized_hidden_layer_values(tokens: List[str]) -> List[Any]:
    """Reconstruct tuple strings from YAML-tokenized hidden_layer_sizes entries."""

    pattern = re.compile(r"\\([^()]*\\)|-?\\d+")
    merged: List[Any] = []
    text = " ".join(tokens)
    for segment in pattern.findall(text):
        if segment.startswith("(") and ")" in segment:
            digits = re.findall(r"-?\\d+", segment)
            if not digits:
                continue
            if len(digits) == 1:
                merged.append(f"({digits[0]},)")
            else:
                merged.append("(" + ", ".join(digits) + ")")
        else:
            merged.append(segment)
    return merged


def _normalize_hidden_layer_sizes(raw: Any) -> List[Tuple[int, ...]]:
    """Normalize hidden_layer_sizes config values into tuples of ints."""

    if raw is None:
        return []

    candidates: List[Tuple[int, ...]] = []

    if isinstance(raw, list):
        values: List[Any]
        if raw and all(isinstance(elem, str) for elem in raw) and any("(" in elem or ")" in elem for elem in raw):
            merged = _merge_tokenized_hidden_layer_values(raw)
            values = merged if merged else list(raw)
        else:
            values = list(raw)
    else:
        values = [raw]

    for value in values:
        try:
            tuple_value = _coerce_hidden_layer_tuple(value)
        except ValueError:
            continue
        if tuple_value not in candidates:
            candidates.append(tuple_value)

    if not candidates:
        raise ValueError("No valid hidden_layer_sizes values could be parsed from configuration")

    return candidates
