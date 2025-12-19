# automl_lib（ClearML連動 AutoML / Tabular）

本リポジトリは、表形式データ向けの AutoML ワークフローを `automl_lib/` に集約したものです。  
YAML設定ファイル駆動で、各フェーズの単独実行とパイプライン実行（preprocessing → training）を提供します。

- フェーズ: `data_registration` / `data_editing` / `preprocessing` / `training` / `inference`
- 入力データ: 原則 **ClearML Dataset ID**（`data.dataset_id`）
- 実行: `python -m automl_lib.cli.run_<phase> --config <yaml>`

詳細:
- 利用者向け: `README_user.md`
- 開発者向け: `README_dev.md`

ClearML上での「見る場所」を固定した最短導線は `README_user.md` の「ClearMLで迷わない見方（最短導線）」を参照してください。

---

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 任意（Optuna/SHAP/LightGBM/TabPFN など）
pip install -r requirements-optional.txt
```

### ClearML 設定（任意）

本リポジトリは、リポジトリルートの `clearml.conf` を優先して利用します。

```bash
cp clearml.conf.example clearml.conf
# clearml.conf に URL / access_key / secret_key を設定
```

---

## 実行例

### Hydra/OmegaConf（推奨）

Hydra 設定は `conf/` 配下にあり、CLI で上書きして実行できます（例: `data.dataset_id=...`）。

```bash
./.venv/bin/python -m automl_lib.cli.run_pipeline_hydra training.data.dataset_id=<CLEARML_DATASET_ID>
./.venv/bin/python -m automl_lib.cli.run_training_hydra  data.dataset_id=<CLEARML_DATASET_ID>
./.venv/bin/python -m automl_lib.cli.run_preprocessing_hydra data.dataset_id=<CLEARML_DATASET_ID>
./.venv/bin/python -m automl_lib.cli.run_inference_hydra model_id=<CLEARML_MODEL_ID>
```

### パイプライン（推奨）

```bash
./.venv/bin/python -m automl_lib.cli.run_pipeline --config config.yaml
```

### フェーズ単独実行

```bash
./.venv/bin/python -m automl_lib.cli.run_preprocessing --config config_preprocessing.yaml
./.venv/bin/python -m automl_lib.cli.run_training       --config config_training.yaml
```

### Clone実行（テンプレTask複製）

```bash
./.venv/bin/python -m automl_lib.cli.run_clone --task-id <TEMPLATE_TASK_ID> --queue default --overrides overrides.yaml
```

### Clone実行（configで有効化）

各 `run_<phase>` / `run_pipeline` は、config の `clearml.execution_mode: clone` が指定されている場合に
テンプレTaskを複製して（必要ならenqueueして）終了します。

```yaml
clearml:
  enabled: true
  execution_mode: clone
  template_task_id: "<TEMPLATE_TASK_ID>"
  queue: "default"
  run_tasks_locally: false   # false のとき queue を使って enqueue
```

---

## 設定ファイル（同梱テンプレ）

- `config.yaml`（pipeline 用の統合設定。`data.dataset_id` を入力に想定）
- `config_preprocessing.yaml` / `config_training.yaml`（フェーズ別設定）
- `config_dataregit.yaml` / `config_editing.yaml`（データ登録/編集の個別実行）
- `inference_config.yaml`（推論フェーズ）

---

## メンテナンス（不要コード削除）

不要コードの削除は「主観」ではなく vulture/coverage で機械的に確定してから行います。  
手順は `docs/CLEANUP.md` を参照してください。
