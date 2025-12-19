"""
Core processing for preprocessing phase.
特徴量型判定、前処理パイプラインの構築・適用を担当する。
"""

import os
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from automl_lib.integrations.clearml import (
    dataframe_from_dataset,
    register_dataset_from_path,
    disable_resource_monitoring,
    upload_artifacts,
    report_scalar,
    set_user_properties,
)
from automl_lib.integrations.clearml.context import (
    build_run_context,
    get_run_id_env,
    resolve_dataset_key,
    resolve_run_id,
    run_scoped_output_dir,
    set_run_id_env,
)
from automl_lib.integrations.clearml.naming import build_tags, dataset_name
from automl_lib.config.loaders import load_preprocessing_config
from automl_lib.data import get_feature_types
from automl_lib.preprocessing import PreprocessingBundle, generate_preprocessors
from automl_lib.types import DatasetInfo

from automl_lib.workflow.preprocessing.clearml_integration import create_preprocessing_task
from automl_lib.workflow.preprocessing.meta import build_preprocessing_metadata
from automl_lib.workflow.preprocessing.visualization import render_preprocessing_visuals


def run_preprocessing_processing(
    config_path: Path,
    input_info: Optional[Dict[str, Any]] = None,
    *,
    run_id: Optional[str] = None,
    config_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config_path = Path(config_path)
    if isinstance(config_data, dict):
        from automl_lib.config.schemas import PreprocessingConfig

        cfg = PreprocessingConfig.model_validate(config_data)
    else:
        cfg = load_preprocessing_config(config_path)
    try:
        from automl_lib.integrations.clearml.overrides import apply_overrides, get_task_overrides

        overrides = get_task_overrides()
        if overrides:
            cfg = type(cfg).model_validate(apply_overrides(cfg.model_dump(), overrides))
    except Exception:
        pass
    # Allow ClearML HyperParameters edits (clone -> edit -> run) to override YAML config.
    try:
        if cfg.clearml and bool(getattr(cfg.clearml, "enabled", False)):
            from automl_lib.integrations.clearml.hyperparams import (
                apply_preprocessing_hyperparams,
                get_current_task_hyperparams,
            )

            params = get_current_task_hyperparams(cast=True)
            if isinstance(params, dict):
                cfg_dump = cfg.model_dump()
                applied = apply_preprocessing_hyperparams(cfg_dump, params)
                if applied != cfg_dump:
                    cfg = type(cfg).model_validate(applied)
    except Exception as exc:
        raise ValueError(f"Invalid ClearML HyperParameters override for preprocessing: {exc}") from exc
    clearml_cfg = cfg.clearml
    run_id = resolve_run_id(
        explicit=run_id,
        from_input=(input_info or {}).get("run_id"),
        from_config=getattr(cfg.run, "id", None),
        from_env=get_run_id_env(),
    )
    set_run_id_env(run_id)

    # PipelineController local step tasks default to auto_resource_monitoring=True (noisy on non-GPU envs).
    if os.environ.get("AUTO_ML_PIPELINE_ACTIVE") == "1":
        try:
            from clearml import Task  # type: ignore

            disable_resource_monitoring(Task.current_task())
        except Exception:
            pass

    dataset_id_source = (
        (input_info or {}).get("dataset_id")
        or (clearml_cfg.edited_dataset_id if clearml_cfg else None)
        or (clearml_cfg.raw_dataset_id if clearml_cfg else None)
        or cfg.data.dataset_id
    )
    csv_src = (input_info or {}).get("csv_path") or cfg.data.csv_path
    dataset_key = resolve_dataset_key(
        explicit=getattr(cfg.run, "dataset_key", None),
        dataset_id=str(dataset_id_source) if dataset_id_source else None,
        csv_path=csv_src,
    )
    ctx = build_run_context(
        run_id=run_id,
        dataset_key=dataset_key,
        project_root=(clearml_cfg.project_name if clearml_cfg else None),
        dataset_project=(clearml_cfg.dataset_project if clearml_cfg else None),
        user=getattr(cfg.run, "user", None),
    )

    if clearml_cfg and clearml_cfg.preprocessed_dataset_id:
        return DatasetInfo(
            dataset_id=clearml_cfg.preprocessed_dataset_id,
            task_id=(input_info or {}).get("task_id"),
            csv_path=csv_src,
            run_id=run_id,
        ).model_dump()

    if not (clearml_cfg and clearml_cfg.enabled and clearml_cfg.enable_preprocessing):
        return DatasetInfo(
            dataset_id=str(dataset_id_source) if dataset_id_source else None,
            task_id=(input_info or {}).get("task_id"),
            csv_path=csv_src,
            run_id=run_id,
        ).model_dump()

    if not dataset_id_source:
        raise ValueError("preprocessing requires data.dataset_id (existing ClearML Dataset ID)")

    t_total0 = time.perf_counter()
    t0 = time.perf_counter()
    df_raw = dataframe_from_dataset(str(dataset_id_source))
    load_dataset_seconds = float(time.perf_counter() - t0)
    if df_raw is None:
        raise ValueError(f"Failed to load ClearML Dataset (dataset_id={dataset_id_source})")

    target_col = cfg.data.target_column or df_raw.columns[-1]
    feature_cols = cfg.data.feature_columns or [c for c in df_raw.columns if c != target_col]
    X_df = df_raw[feature_cols].copy()

    feature_types = get_feature_types(X_df)
    preprocessors = generate_preprocessors(cfg.preprocessing, feature_types)
    if not preprocessors:
        return DatasetInfo(
            dataset_id=str(dataset_id_source),
            task_id=(input_info or {}).get("task_id"),
            csv_path=csv_src,
            run_id=run_id,
        ).model_dump()

    preproc_name, transformer = preprocessors[0]
    t1 = time.perf_counter()
    transformed = transformer.fit_transform(X_df)
    fit_transform_seconds = float(time.perf_counter() - t1)
    try:
        feature_names = transformer.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(transformed.shape[1])]
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(transformed):  # type: ignore[attr-defined]
            transformed = transformed.toarray()
    except Exception:
        pass

    df_preproc = pd.DataFrame(transformed, columns=feature_names)
    df_preproc[target_col] = df_raw[target_col].reset_index(drop=True)

    cfg_for_task = cfg.model_dump()
    try:
        cfg_for_task.setdefault("run", {})
        cfg_for_task["run"]["id"] = run_id
        cfg_for_task["run"]["dataset_key"] = dataset_key
        if getattr(cfg.run, "user", None):
            cfg_for_task["run"]["user"] = cfg.run.user
    except Exception:
        pass
    try:
        cfg_for_task.setdefault("data", {})
        if dataset_id_source:
            cfg_for_task["data"]["dataset_id"] = str(dataset_id_source)
        if csv_src:
            cfg_for_task["data"]["csv_path"] = str(csv_src)
    except Exception:
        pass

    task_info = create_preprocessing_task(cfg_for_task, parent_task_id=(input_info or {}).get("task_id"))
    task = task_info.get("task")
    logger = task_info.get("logger")
    if task:
        try:
            task.add_tags(build_tags(ctx, phase="preprocessing", preproc=preproc_name, extra=(clearml_cfg.tags or [])))
        except Exception:
            pass

    base_output_dir = Path(cfg.output.output_dir) if cfg.output else Path("outputs/train")
    output_dir = run_scoped_output_dir(base_output_dir, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved full config for reproducibility (avoid HyperParameters pollution).
    try:
        import yaml  # type: ignore

        cfg_path = output_dir / "preprocessing_config.yaml"
        cfg_path.write_text(
            yaml.safe_dump(cfg_for_task, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        if task:
            upload_artifacts(task, [cfg_path])
    except Exception:
        pass
    csv_out = dataset_dir / "data_processed.csv"
    df_preproc.to_csv(csv_out, index=False)

    timing_path = output_dir / "preprocessing_timing.json"
    timing = {
        "run_id": run_id,
        "dataset_key": dataset_key,
        "dataset_id_source": str(dataset_id_source),
        "selected_preprocessor": str(preproc_name),
        "load_dataset_seconds": load_dataset_seconds,
        "fit_transform_seconds": fit_transform_seconds,
    }
    try:
        timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
        if task:
            upload_artifacts(task, [timing_path])
    except Exception:
        pass
    try:
        report_scalar(logger, "time/load_dataset_seconds", "value", load_dataset_seconds, iteration=0)
        report_scalar(logger, "time/fit_transform_seconds", "value", fit_transform_seconds, iteration=0)
    except Exception:
        pass

    # Diagnostics (plots/tables) + metadata artifacts
    meta = None
    try:
        render_preprocessing_visuals(
            logger,
            df_raw=df_raw,
            df_preprocessed=df_preproc,
            target_col=target_col,
            feature_cols=feature_cols,
            feature_types=feature_types,
        )
    except Exception:
        pass
    try:
        meta = build_preprocessing_metadata(
            output_dir=dataset_dir,
            run_id=run_id,
            dataset_key=dataset_key,
            parent_dataset_id=str(dataset_id_source) if dataset_id_source else None,
            preprocessing_task_id=(task.id if task else None),
            contract_version="v1",
            target_col=target_col,
            feature_cols=feature_cols,
            feature_types=feature_types,
            preproc_name=preproc_name,
            cfg_preprocessing=(cfg.preprocessing.model_dump() if hasattr(cfg.preprocessing, "model_dump") else {}),
            df_raw=df_raw,
            df_preprocessed=df_preproc,
        )
        if task and isinstance(meta.get("artifacts"), list):
            paths = [Path(p) for p in meta["artifacts"] if isinstance(p, str)]
            upload_artifacts(task, paths)
    except Exception:
        pass

    # Save fitted preprocessing bundle into the dataset contract directory.
    try:
        target_transformer = None
        try:
            if bool(getattr(cfg.preprocessing, "target_standardize", False)):
                from sklearn.preprocessing import StandardScaler  # type: ignore

                # Fit the target transformer here so inference can reliably inverse-transform
                # even when the exported model does not wrap TransformedTargetRegressor.
                try:
                    y_series = df_raw[target_col]
                    y_numeric = pd.to_numeric(y_series, errors="coerce")
                    y_nonnull = y_numeric.dropna()
                    if not y_nonnull.empty:
                        scaler = StandardScaler()
                        scaler.fit(y_nonnull.to_numpy().reshape(-1, 1))
                        target_transformer = scaler
                except Exception:
                    target_transformer = None
        except Exception:
            target_transformer = None

        from joblib import dump  # type: ignore

        bundle_path = dataset_dir / "preprocessing" / "bundle.joblib"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = PreprocessingBundle(
            feature_transformer=transformer,
            target_transformer=target_transformer,
            contract_version="v1",
        )
        dump(bundle, bundle_path)
        if task:
            upload_artifacts(task, [bundle_path])
    except Exception:
        pass

    t2 = time.perf_counter()
    dataset_id = register_dataset_from_path(
        name=dataset_name("preprocessed", ctx, preproc=preproc_name),
        path=dataset_dir,
        dataset_project=clearml_cfg.dataset_project if clearml_cfg else None,
        parent_ids=[str(dataset_id_source)],
        tags=build_tags(ctx, phase="preprocessing", preproc=preproc_name, extra=[*(clearml_cfg.tags or []), "preprocessed"]),
        output_uri=clearml_cfg.base_output_uri if clearml_cfg else None,
        hyperparams_sections={
            "Preprocessing": {
                "numeric_imputation": list(getattr(cfg.preprocessing, "numeric_imputation", []) or []),
                "categorical_imputation": list(getattr(cfg.preprocessing, "categorical_imputation", []) or []),
                "scaling": list(getattr(cfg.preprocessing, "scaling", []) or []),
                "categorical_encoding": list(getattr(cfg.preprocessing, "categorical_encoding", []) or []),
                "polynomial_degree": getattr(cfg.preprocessing, "polynomial_degree", False),
                "target_standardize": bool(getattr(cfg.preprocessing, "target_standardize", False)),
            }
        },
        configuration_objects={
            "Preprocessing": {
                "numeric_imputation": list(getattr(cfg.preprocessing, "numeric_imputation", []) or []),
                "categorical_imputation": list(getattr(cfg.preprocessing, "categorical_imputation", []) or []),
                "scaling": list(getattr(cfg.preprocessing, "scaling", []) or []),
                "categorical_encoding": list(getattr(cfg.preprocessing, "categorical_encoding", []) or []),
                "polynomial_degree": getattr(cfg.preprocessing, "polynomial_degree", False),
                "target_standardize": bool(getattr(cfg.preprocessing, "target_standardize", False)),
            }
        },
    )
    register_dataset_seconds = float(time.perf_counter() - t2)
    if not dataset_id:
        raise RuntimeError("Failed to register preprocessed dataset to ClearML")

    try:
        if task:
            schema = (meta or {}).get("schema") if isinstance(meta, dict) else None
            n_rows = schema.get("n_rows") if isinstance(schema, dict) else None
            n_features_pre = schema.get("n_features_preprocessed") if isinstance(schema, dict) else None
            set_user_properties(
                task,
                {
                    "run_id": run_id,
                    "dataset_key": dataset_key,
                    "dataset_role": "preprocessed",
                    "source_dataset_id": str(dataset_id_source),
                    "preprocessed_dataset_id": str(dataset_id),
                    "target_column": str(target_col),
                    "selected_preprocessor": str(preproc_name),
                    "n_rows": n_rows if n_rows is not None else "",
                    "n_features_preprocessed": n_features_pre if n_features_pre is not None else "",
                },
            )
    except Exception:
        pass
    # Keep reference information in ClearML "Configuration Objects" (not HyperParameters)
    try:
        if task and isinstance(meta, dict):
            schema = meta.get("schema") if isinstance(meta.get("schema"), dict) else {}
            manifest = meta.get("manifest") if isinstance(meta.get("manifest"), dict) else {}
            recipe = meta.get("recipe") if isinstance(meta.get("recipe"), dict) else {}
            summary_md = ""
            try:
                summary_path = Path(dataset_dir) / "preprocessing" / "summary.md"
                if summary_path.exists():
                    summary_md = summary_path.read_text(encoding="utf-8")
            except Exception:
                summary_md = ""

            dataset_conf = {
                "run_id": str(run_id),
                "dataset_key": str(dataset_key),
                "source_dataset_id": str(dataset_id_source),
                "preprocessed_dataset_id": str(dataset_id),
                "csv_path_processed": str(csv_out),
                "schema": schema,
                "manifest": manifest,
            }
            preproc_conf = {
                "selected_preprocessor": str(preproc_name),
                "recipe": recipe,
                "summary_md": summary_md,
                "contract_version": str(recipe.get("contract_version") or "v1"),
            }
            try:
                task.connect_configuration(name="Dataset", configuration=dataset_conf)
            except Exception:
                pass
            try:
                task.connect_configuration(name="Preprocessing", configuration=preproc_conf)
            except Exception:
                pass
    except Exception:
        pass
    try:
        timing["register_dataset_seconds"] = register_dataset_seconds
        timing["dataset_id_preprocessed"] = str(dataset_id)
        timing["total_seconds"] = float(time.perf_counter() - t_total0)
        timing_path.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
        if task:
            upload_artifacts(task, [timing_path])
    except Exception:
        pass
    try:
        report_scalar(logger, "time/register_dataset_seconds", "value", register_dataset_seconds, iteration=0)
        report_scalar(logger, "time/total_seconds", "value", float(time.perf_counter() - t_total0), iteration=0)
    except Exception:
        pass

    try:
        if task:
            task.upload_artifact("preprocessed_dataset_id", artifact_object=str(dataset_id), wait_on_upload=True)
    except Exception:
        pass
    try:
        pre_id_path = output_dir / "preprocessed_dataset_id.txt"
        pre_id_path.write_text(str(dataset_id), encoding="utf-8")
        if task:
            upload_artifacts(task, [pre_id_path])
    except Exception:
        pass
    try:
        if task:
            task.flush(wait_for_uploads=True)
    except Exception:
        pass
    try:
        if task and os.environ.get("AUTO_ML_PIPELINE_ACTIVE") != "1":
            task.close()
    except Exception:
        pass

    return DatasetInfo(
        dataset_id=dataset_id,
        task_id=(task.id if task else None),
        csv_path=str(csv_out),
        run_id=run_id,
    ).model_dump()
