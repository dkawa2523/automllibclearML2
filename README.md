# automl_lib（ClearML連動 AutoML / Tabular）

本リポジトリは、表形式データ向けの AutoML ワークフローを `automl_lib/` に集約したものです。  
YAML設定ファイル駆動で、各フェーズの単独実行とパイプライン実行（preprocessing → training → compare_results）を提供します。

- フェーズ: `data_registration` / `data_editing` / `preprocessing` / `training` / `comparison` / `inference`
- 入力データ: 原則 **ClearML Dataset ID**（`data.dataset_id`）
- 実行: `python -m automl_lib.cli.run_<phase> --config <yaml>`

詳細:
- 利用者向け: `README_user.md`
- 開発者向け: `README_dev.md`

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

### パイプライン（推奨）

```bash
./.venv/bin/python -m automl_lib.cli.run_pipeline --config config.yaml
```

### フェーズ単独実行

```bash
./.venv/bin/python -m automl_lib.cli.run_preprocessing --config config_preprocessing.yaml
./.venv/bin/python -m automl_lib.cli.run_training       --config config_training.yaml
./.venv/bin/python -m automl_lib.cli.run_comparison     --config config_comparison.yaml --training-info outputs/training_info.json
```

---

## 設定ファイル（同梱テンプレ）

- `config.yaml`（pipeline 用の統合設定。`data.dataset_id` を入力に想定）
- `config_preprocessing.yaml` / `config_training.yaml` / `config_comparison.yaml`（フェーズ別設定）
- `config_dataregit.yaml` / `config_editing.yaml`（データ登録/編集の個別実行）
- `inference_config.yaml`（推論フェーズ）
