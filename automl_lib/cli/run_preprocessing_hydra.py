from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from automl_lib.cli.common import clearml_avoid_task_reuse, maybe_clone_from_config
from automl_lib.cli.hydra_support import conf_root, to_clean_dict, write_yaml_config
from automl_lib.config.schemas import PreprocessingConfig
from automl_lib.workflow import run_preprocessing


@hydra.main(version_base=None, config_path=str(conf_root()), config_name="app/preprocessing")
def main(cfg: DictConfig) -> None:
    cfg_dict = to_clean_dict(cfg)
    validated = PreprocessingConfig.model_validate(cfg_dict)
    if maybe_clone_from_config(validated, phase="preprocessing", output_info=None):
        return
    clearml_avoid_task_reuse()

    cfg_path = write_yaml_config(cfg_dict, prefix="preprocessing")
    result = run_preprocessing(cfg_path, input_info=None)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

