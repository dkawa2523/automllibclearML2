# Developer Guide (automl_lib_clearML)

このプロジェクトは「各Phaseの処理」と「ClearMLへの出力」を分離し、ClearML UI上で run_id を軸に追えることを目的にしています。

## 目的（最重要）

- run_id・命名・tags を統一し、ClearML上で迷子にならない
- Hyperparameters をカテゴリ分割し、「何を変えたか」を追える
- 前処理/学習で、非専門でも理解できる可視化・サマリを残す

## 主要ディレクトリ

- `automl_lib/cli/`: `run_<phase>.py` が入口（設定を読み、phase処理を呼ぶ）
- `automl_lib/workflow/<phase>/processing.py`: 各phaseの主処理
- `automl_lib/workflow/<phase>/clearml_integration.py` / `runner.py`: ClearML task生成、親子リンク、出力登録（phaseごとに分離）
- `automl_lib/integrations/clearml/`: RunContext / naming / clone / logging / datasets 等の共通基盤（`automl_lib/clearml/` は互換ラッパ）
- `automl_lib/training/`: 学習の実行（指標/plots/artifacts）

## run_id / RunContext

- `automl_lib/integrations/clearml/context.py` が run_id と dataset_key を解決します。
  - 優先度：明示引数 → 入力info → config → env（`AUTO_ML_RUN_ID`）
- phase間で run_id を渡すため、処理の入口で `AUTO_ML_RUN_ID` を設定します。
- 出力は原則 run 単位で分離します（上書き防止）。
  - `automl_lib/integrations/clearml/context.py::run_scoped_output_dir()` を利用し、`outputs/<phase>/<run_id>/...` に保存します。

## 命名・tags 規約

- `automl_lib/integrations/clearml/naming.py` に命名/タグ付け規約を集約しています。
- 基本 tags（最低限）
  - `run:<run_id>`, `phase:<phase>`, `dataset:<dataset_key>`
  - 学習子タスクは `preprocess:<preproc_id>`, `model:<model_name>` を付与

## Hyperparameters（カテゴリ分割）

- `automl_lib/integrations/clearml/logging.py::report_hyperparams_sections()` を利用します。
- Phase側は「見せたい単位」に合わせて辞書を整形して渡します。
  - 例：training-summary は `ModelCandidates`、学習子タスクは `Model` を1モデルに絞る
- 注意: ClearML PipelineController の function step では内部メタとして `kwargs` / `kwargs_artifacts` / `properties` が HyperParameters に自動付与されます（step再実行のための仕様）。本プロジェクトではこれを許容し、ユーザーが触るべき section（`Training` / `Models` / ...）は別途安定して出す方針です。

## USER PROPERTIES（短い要約・検索キー）

- ClearML の `Configuration > USER PROPERTIES` は「結果の要約/識別情報」を置く場所として使います。
- 追加は `automl_lib/integrations/clearml/properties.py::set_user_properties()` を利用します（長い値は自動で短縮）。
- training-summary では `recommended_model_id` / `recommended_model_task_id` / `dataset_id` / `run_id` などを保存します。

## 可視化/メタ情報（追加ポイント）

### Preprocessing

- `automl_lib/workflow/preprocessing/visualization.py`: ターゲット分布/欠損/特徴量数など
- `automl_lib/workflow/preprocessing/meta.py`: schema/pipeline 等の artifacts を生成
- 前処理Dataset契約（v1）を ClearML Dataset に同梱します:
  - `schema.json` / `manifest.json`
  - `preprocessing/summary.md`（カテゴリ別、1行ダンプ禁止）
  - `preprocessing/recipe.json` / `preprocessing/version.txt` / `preprocessing/bundle.joblib`
- `outputs/preprocessing/<run_id>/preprocessing_timing.json` に処理時間を保存し、`time/*` の scalar も送信します

### Training

- `automl_lib/workflow/training/runner.py` が実行本体です（段階的に core / workflow の責務分離を進行中）
- `automl_lib/training/reporting/scalars.py`: `metric/<name>` など比較しやすいscalar命名を集約
- 学習子タスクは、`metric/*`（比較用） + `time/*` + `model/*` を揃える方針です。
- training-summary は前処理Dataset契約を検出した場合、二重前処理を避けるため identity 前処理として扱います。参照用の前処理情報は `Configuration Objects > Preprocessing` と `Artifacts` に保存します（PLOTSに長文を出しません）。
- 推奨モデルは training の primary metric（`evaluation.primary_metric`）で選定します（`clearml.recommendation_mode: auto|training`）。

## Clone実行（テンプレ複製）

- `automl_lib/integrations/clearml/clone.py` と `automl_lib/cli/run_clone.py` を利用します。
- cloneには `run:<new_run_id>` と `cloned_from:<template_task_id>` を付与します。
- config の `clearml.execution_mode: clone` / `clearml.template_task_id` を指定すると、各CLIが clone → (必要なら) enqueue して終了します。

## テスト

- `.venv\Scripts\python -m unittest discover -s tests -q`

## P0受け入れ確認（ClearML UI / training-summary）

ClearML 上で「training-summary の Plots が常に 01–08 のみ」になっていることを機械的に確認するには、
以下を実行します（ClearML サーバが必要 / `CLEARML_OFFLINE_MODE` は無効にする）。

```bat
.venv\Scripts\python scripts\verify_clearml_training_summary_plots.py --config config.yaml
```

このスクリプトは:
- `training-summary` タスクが見つかるまで待機
- Plots の metric が 01–08 以外出ていないこと
- debug-image（report_image）メトリクスが混入していないこと
- HyperParameters に必要セクション（Training/Models/...）が存在すること
を検証します。

## P0受け入れ確認（ClearML UI / inference）

ClearML 上で「pipeline 実行（enable_inference=true）時に inference の親子タスク / artifacts が仕様通り」であることは、
以下を実行して機械的に確認できます（ClearML サーバが必要 / `CLEARML_OFFLINE_MODE` は無効にする）。

```bat
.venv\Scripts\python scripts\verify_clearml_inference_pipeline.py --config config.yaml --inference-config inference_config.yaml
```

このスクリプトは:
- pipeline を新規 run_id で実行（内部で `clearml.enable_inference=true` を上書き）
- mode に応じてタスク構造が正しいこと
  - single/batch: `infer <mode> ...` の単独タスク（inference-summary は作らない）
  - optimize: `inference-summary` + `Prediction_runs/*` の `predict ...` 子タスク（全trial）
- mode に応じた artifact が存在すること（single: `input.json`/`output.json`, optimize: `trials.csv`/`best_solution.json` など）
- training-summary の `recommended_model_id` が inference 側の `model_id` と一致すること
を検証します。

## 新しい拡張を入れる時の推奨フロー

1. run_id/tags/naming が付くことを確認（ClearML検索で追えること）
2. Hyperparameters を section 化して UI の見通しを確保
3. artifacts / plots / scalars は「比較しやすい規約（title/series）」に寄せる
4. Phaseの出力（`automl_lib/types/phase_io.py`）に必要なら optional で追加
