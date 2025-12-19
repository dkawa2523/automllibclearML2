from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from automl_lib.config.schemas import TrainingConfig
from automl_lib.registry.metrics import add_derived_metrics, is_loss_metric
from automl_lib.training.evaluation import _get_cv_splitter, _get_scoring, evaluate_model_combinations
from automl_lib.training.ensemble import build_stacking, build_voting
from automl_lib.training.model_factory import ModelInstance, _get_model_class
from automl_lib.training.search import generate_param_combinations
from automl_lib.workflow.training.estimators import build_estimator_with_defaults, maybe_wrap_with_target_scaler


def build_model_instances(
    *,
    cfg: TrainingConfig,
    preprocessors: List[Tuple[str, object]],
    X_train: Any,
    y_train: Any,
    problem_type: str,
    metrics: List[str],
) -> List[ModelInstance]:
    model_instances: List[ModelInstance] = []
    for spec in cfg.models:
        if hasattr(spec, "enable") and not spec.enable:
            continue
        combos = generate_param_combinations(
            spec=spec,
            problem_type=problem_type,
            optimization_config=cfg.optimization,
            preprocessors=preprocessors,
            X=X_train,
            y=y_train,
            cv_config=cfg.cross_validation,
            metrics=metrics,
            target_standardize=cfg.preprocessing.target_standardize if problem_type == "regression" else False,
        )
        for params in combos:
            try:
                cls = _get_model_class(spec.name, problem_type)
            except Exception as exc:
                print(f"Warning: {exc}")
                continue

            init_params: Dict[str, Any] = {}
            if params:
                for key, value in params.items():
                    val = value
                    if isinstance(val, str):
                        try:
                            val = ast.literal_eval(val)
                        except Exception:
                            pass
                    if key.lower() == "hidden_layer_sizes":
                        if isinstance(val, list):
                            if len(val) == 1:
                                inner = val[0]
                                if isinstance(inner, (list, tuple)):
                                    val = tuple(inner)
                                else:
                                    val = tuple(val)
                            else:
                                val = tuple(val)
                        elif isinstance(val, tuple):
                            val = val
                    init_params[key] = val

            try:
                estimator, applied_params = build_estimator_with_defaults(
                    spec.name,
                    cls,
                    init_params,
                    problem_type,
                    cfg,
                    len(y_train),
                )
            except ValueError as exc:
                print(f"Warning: could not instantiate {spec.name}: {exc}")
                continue
            except Exception as exc:
                print(f"Warning: could not instantiate {spec.name} with {init_params}: {exc}")
                continue
            model_instances.append(ModelInstance(name=spec.name, params=applied_params, estimator=estimator))
    return model_instances


def evaluate_models_with_ensembles(
    *,
    cfg: TrainingConfig,
    preprocessors: List[Tuple[str, object]],
    model_instances: List[ModelInstance],
    X_train: Any,
    y_train: Any,
    problem_type: str,
    metrics: List[str],
) -> pd.DataFrame:
    results_df = evaluate_model_combinations(
        preprocessors,
        model_instances,
        X_train,
        y_train,
        cfg.cross_validation,
        problem_type,
        metrics,
    )

    ensemble_records: List[Dict[str, Any]] = []
    if cfg.ensembles.stacking.enable:
        base_names = cfg.ensembles.stacking.estimators
        final_name = cfg.ensembles.stacking.final_estimator
        for preproc_name, transformer in preprocessors:
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            if not ests:
                continue
            final_est = None
            if final_name:
                try:
                    final_cls = _get_model_class(final_name, problem_type)
                    final_est = final_cls()
                except Exception:
                    final_est = None
            if final_est is None:
                try:
                    if problem_type == "regression":
                        from sklearn.linear_model import LinearRegression

                        final_est = LinearRegression()
                    else:
                        from sklearn.linear_model import LogisticRegression

                        final_est = LogisticRegression(max_iter=1000)
                except Exception:
                    continue
            stack_pipeline = build_stacking(transformer, ests, final_est, problem_type)
            stack_for_eval = (
                maybe_wrap_with_target_scaler(stack_pipeline, cfg, problem_type)
                if problem_type == "regression"
                else stack_pipeline
            )
            try:
                from sklearn.model_selection import cross_validate

                cv = _get_cv_splitter(problem_type, len(y_train), cfg.cross_validation, y_train)
                scoring = _get_scoring(problem_type, metrics)
                cv_res = cross_validate(
                    stack_for_eval,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                )
                rec: Dict[str, Any] = {
                    "preprocessor": preproc_name,
                    "model": f"Stacking({' + '.join(base_names)})",
                    "params": {},
                }
                for metric_name, scores in cv_res.items():
                    if not metric_name.startswith("test_"):
                        continue
                    simple = metric_name.replace("test_", "")
                    mean_score = float(np.mean(scores))
                    if is_loss_metric(simple, problem_type=problem_type):
                        mean_score = -mean_score
                    rec[simple] = mean_score
                add_derived_metrics(rec, problem_type=problem_type, requested_metrics=metrics)
                ensemble_records.append(rec)
            except Exception as exc:
                print(f"Warning: stacking ensemble failed for {preproc_name}: {exc}")

    if cfg.ensembles.voting.enable:
        base_names = cfg.ensembles.voting.estimators
        voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
        for preproc_name, transformer in preprocessors:
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            if not ests:
                continue
            vote_pipeline = build_voting(transformer, ests, voting_scheme, problem_type)
            vote_for_eval = (
                maybe_wrap_with_target_scaler(vote_pipeline, cfg, problem_type)
                if problem_type == "regression"
                else vote_pipeline
            )
            try:
                from sklearn.model_selection import cross_validate

                cv = _get_cv_splitter(problem_type, len(y_train), cfg.cross_validation, y_train)
                scoring = _get_scoring(problem_type, metrics)
                cv_res = cross_validate(
                    vote_for_eval,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    error_score="raise",
                )
                rec = {"preprocessor": preproc_name, "model": f"Voting({' + '.join(base_names)})", "params": {}}
                for metric_name, scores in cv_res.items():
                    if not metric_name.startswith("test_"):
                        continue
                    simple = metric_name.replace("test_", "")
                    mean_score = float(np.mean(scores))
                    if is_loss_metric(simple, problem_type=problem_type):
                        mean_score = -mean_score
                    rec[simple] = mean_score
                add_derived_metrics(rec, problem_type=problem_type, requested_metrics=metrics)
                ensemble_records.append(rec)
            except Exception as exc:
                print(f"Warning: voting ensemble failed for {preproc_name}: {exc}")

    if ensemble_records:
        results_df = pd.concat([results_df, pd.DataFrame(ensemble_records)], ignore_index=True)

    return results_df


def resolve_primary_metric(
    *,
    cfg: TrainingConfig,
    results_df: pd.DataFrame,
    problem_type: str,
) -> Optional[str]:
    primary_metric = cfg.evaluation.primary_metric or ("r2" if problem_type == "regression" else "accuracy")
    if primary_metric not in results_df.columns:
        available_cols = [c for c in results_df.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric = available_cols[0] if available_cols else None
    return primary_metric


def select_best_rows_per_model(
    *,
    results_df: pd.DataFrame,
    primary_metric: Optional[str],
    problem_type: str,
) -> pd.DataFrame:
    best_rows: Dict[str, pd.Series] = {}
    if primary_metric:
        goal_primary = "min" if is_loss_metric(primary_metric, problem_type=problem_type) else "max"
        for _, row in results_df.iterrows():
            if pd.isna(row.get(primary_metric)):
                continue
            model_label = row["model"]
            score = row[primary_metric]
            if model_label not in best_rows:
                best_rows[model_label] = row
                continue
            current = best_rows[model_label].get(primary_metric)
            if pd.isna(current):
                best_rows[model_label] = row
                continue
            better = score < current if goal_primary == "min" else score > current
            if better:
                best_rows[model_label] = row
    else:
        for _, row in results_df.iterrows():
            best_rows.setdefault(row["model"], row)

    return pd.DataFrame(list(best_rows.values())) if best_rows else results_df.copy()


def pick_best_row_global(
    *,
    results_df_best: pd.DataFrame,
    primary_metric: Optional[str],
    problem_type: str,
) -> pd.Series:
    primary_metric_global = primary_metric
    if not primary_metric_global or primary_metric_global not in results_df_best.columns:
        cols_available = [c for c in results_df_best.columns if c not in {"preprocessor", "model", "params", "error"}]
        primary_metric_global = cols_available[0] if cols_available else None

    if primary_metric_global:
        goal_global = "min" if is_loss_metric(primary_metric_global, problem_type=problem_type) else "max"
        return (
            results_df_best.loc[results_df_best[primary_metric_global].idxmin()]
            if goal_global == "min"
            else results_df_best.loc[results_df_best[primary_metric_global].idxmax()]
        )
    return results_df_best.iloc[0]


def fit_pipeline_for_row(
    *,
    cfg: TrainingConfig,
    preprocessors: List[Tuple[str, object]],
    best_row: pd.Series,
    problem_type: str,
    X_train: Any,
    y_train: Any,
) -> tuple[Any, str, str, Dict[str, Any]]:
    best_preproc_name = best_row["preprocessor"]
    best_model_name = best_row["model"]
    best_params = best_row["params"] if isinstance(best_row.get("params"), dict) else {}

    transformer = None
    for name, ct in preprocessors:
        if name == best_preproc_name:
            transformer = ct
            break
    if transformer is None:
        transformer = preprocessors[0][1]

    estimator = None
    if str(best_model_name).startswith("Stacking") or str(best_model_name).startswith("Voting"):
        if str(best_model_name).startswith("Stacking"):
            base_names = cfg.ensembles.stacking.estimators
            final_name = cfg.ensembles.stacking.final_estimator
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            final_est = None
            if final_name:
                try:
                    final_cls = _get_model_class(final_name, problem_type)
                    final_est = final_cls()
                except Exception:
                    final_est = None
            if final_est is None:
                from sklearn.linear_model import LinearRegression, LogisticRegression

                final_est = LinearRegression() if problem_type == "regression" else LogisticRegression(max_iter=1000)
            estimator = build_stacking(transformer, ests, final_est, problem_type).named_steps["model"]
            if problem_type == "regression":
                estimator = maybe_wrap_with_target_scaler(estimator, cfg, problem_type)
        else:
            base_names = cfg.ensembles.voting.estimators
            voting_scheme = cfg.ensembles.voting.voting or ("hard" if problem_type == "classification" else "soft")
            ests: List[Tuple[str, object]] = []
            for bn in base_names:
                try:
                    cls_bn = _get_model_class(bn, problem_type)
                    ests.append((bn, cls_bn()))
                except Exception:
                    pass
            estimator = build_voting(transformer, ests, voting_scheme, problem_type).named_steps["model"]
            if problem_type == "regression":
                estimator = maybe_wrap_with_target_scaler(estimator, cfg, problem_type)
    else:
        try:
            cls_best = _get_model_class(str(best_model_name), problem_type)
            estimator, _ = build_estimator_with_defaults(
                str(best_model_name),
                cls_best,
                best_params,
                problem_type,
                cfg,
                len(y_train),
            )
        except Exception:
            estimator = None

    from sklearn.pipeline import Pipeline as SKPipeline

    full_pipeline = SKPipeline([("preprocessor", transformer), ("model", estimator)])
    full_pipeline.fit(X_train, y_train)
    return full_pipeline, str(best_model_name), str(best_preproc_name), dict(best_params or {})

