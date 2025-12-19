from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig

from automl_lib.cli.common import clearml_avoid_task_reuse, maybe_clone_from_config
from automl_lib.cli.hydra_support import conf_root, to_clean_dict, write_yaml_config
from automl_lib.config.schemas import InferenceConfig
from automl_lib.workflow import run_inference


@hydra.main(version_base=None, config_path=str(conf_root()), config_name="app/inference")
def main(cfg: DictConfig) -> None:
    cfg_dict = to_clean_dict(cfg)
    validated = InferenceConfig.model_validate(cfg_dict)
    if maybe_clone_from_config(validated, phase="inference", output_info=None):
        return
    clearml_avoid_task_reuse()

    cfg_path = write_yaml_config(cfg_dict, prefix="inference")
    result = run_inference(cfg_path)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

