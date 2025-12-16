# Developer Guide (automl_lib_clearML)

このプロジェクトは「各Phaseの処理」と「ClearMLへの出力」を分離し、ClearML UI上で run_id を軸に追えることを目的にしています。

## 目的（最重要）

- run_id・命名・tags を統一し、ClearML上で迷子にならない
- Hyperparameters をカテゴリ分割し、「何を変えたか」を追える
- 前処理/学習/（必要なら比較）で、非専門でも理解できる可視化・サマリを残す

## 主要ディレクトリ

- `automl_lib/cli/`: `run_<phase>.py` が入口（設定を読み、phase処理を呼ぶ）
- `automl_lib/phases/<phase>/processing.py`: 各phaseの主処理
- `automl_lib/phases/<phase>/clearml_integration.py`: ClearML task生成、親子リンク、出力登録
- `automl_lib/clearml/`: RunContext / naming / clone / logging / datasets 等の共通基盤
- `automl_lib/training/`: 学習の実行（指標/plots/artifacts）

## run_id / RunContext

- `automl_lib/clearml/context.py` が run_id と dataset_key を解決します。
  - 優先度：明示引数 → 入力info → config → env（`AUTO_ML_RUN_ID`）
- phase間で run_id を渡すため、処理の入口で `AUTO_ML_RUN_ID` を設定します。
- 出力は原則 run 単位で分離します（上書き防止）。
  - `automl_lib/clearml/context.py::run_scoped_output_dir()` を利用し、`outputs/<phase>/<run_id>/...` に保存します。

## 命名・tags 規約

- `automl_lib/clearml/naming.py` に命名/タグ付け規約を集約しています。
- 基本 tags（最低限）
  - `run:<run_id>`, `phase:<phase>`, `dataset:<dataset_key>`
  - 学習子タスクは `preprocess:<preproc_id>`, `model:<model_name>` を付与

## Hyperparameters（カテゴリ分割）

- `automl_lib/clearml/logging.py::report_hyperparams_sections()` を利用します。
- Phase側は「見せたい単位」に合わせて辞書を整形して渡します。
  - 例：training-summary は `ModelCandidates`、学習子タスクは `Model` を1モデルに絞る

## 可視化/メタ情報（追加ポイント）

### Preprocessing

- `automl_lib/phases/preprocessing/visualization.py`: ターゲット分布/欠損/特徴量数など
- `automl_lib/phases/preprocessing/meta.py`: schema/pipeline 等の artifacts を生成
- `outputs/preprocessing/<run_id>/preprocessing_timing.json` に処理時間を保存し、`time/*` の scalar も送信します

### Training

- `automl_lib/training/run.py` が実行本体です（段階的に `automl_lib/training/reporting/` に分離中）
- `automl_lib/training/reporting/scalars.py`: `metric/<name>` など比較しやすいscalar命名を集約
- 学習子タスクは、`metric/*`（比較用） + `time/*` + `model/*` を揃える方針です。
- `clearml.comparison_mode: embedded` の場合、training-summary の推奨モデル選定は comparison の `ranking.*`（例: `composite_score`）に合わせます。
  - `clearml.recommendation_mode` で切替可能（`auto`/`training`/`comparison`）。

### Comparison（任意/手動）

- pipeline は comparison タスクを作りません（`clearml.comparison_mode: embedded` で training-summary に集約します）。
- 複数 run を横断して比較したい場合のみ `automl_lib/cli/run_comparison.py` を使います（`TrainingInfo` を複数渡せます）。

## Clone実行（テンプレ複製）

- `automl_lib/clearml/clone.py` と `automl_lib/cli/run_clone.py` を利用します。
- cloneには `run:<new_run_id>` と `cloned_from:<template_task_id>` を付与します。
- config の `clearml.execution_mode: clone` / `clearml.template_task_id` を指定すると、各CLIが clone → (必要なら) enqueue して終了します。

## テスト

- `./.venv/bin/python -m unittest discover -s tests -q`

## 新しい拡張を入れる時の推奨フロー

1. run_id/tags/naming が付くことを確認（ClearML検索で追えること）
2. Hyperparameters を section 化して UI の見通しを確保
3. artifacts / plots / scalars は「比較しやすい規約（title/series）」に寄せる
4. Phaseの出力（`automl_lib/types/phase_io.py`）に必要なら optional で追加
