# docs/REFORM_PLAN.md

# automllibclearML2 リファクタリング提案計画（ClearML × Table AutoML）

> **目的（ユーザー / 非ML）**: dataset_id を入力して Pipeline 実行 → ClearML上で結果を見て「どのモデルを推論に使うか」を迷わず決められる。
> **目的（開発者）**: ClearML/各ライブラリ仕様を深掘りしなくても、定めた拡張ポイント（前処理/モデル/評価/可視化/情報付与）を守れば安全に追加できる。

---

## 0. このドキュメントの前提（重要）

* ここでは **「実装できる粒度の計画」**を提示します（設計方針 + フォルダ構成 + タスク責務 + ClearML UI設計 + 実装手順 + 削除手順）。
* **実ファイルの“削除対象リスト”は、必ずリポジトリ内で静的解析 + 実行トレースで確定**させます（手順は後述）。

  * ※本チャット環境から `automllibclearML2` の実コードを直接参照できないため、「調査して削除」部分は **再現可能な自動抽出プロセス**として提示します。

---

## 1. ゴール体験（ClearML上のユーザージャーニー）

### 1.1 ユーザー（非ML）の最短導線

1. **Pipelineタスク**（Controller）を開く
2. `dataset_id` を入力して実行
3. 生成された **Training Summary（親タスク）**を開く
4. 見る場所は固定：

   * `Configuration` → 「Training」だけ確認（余計なパラメータは出ない）
   * `Plots` → 上部に **Leaderboard（モデル比較）**と **Recommended Model** がある（詳細は出ない）
   * `Artifacts` → `leaderboard.csv` / `results_summary.csv` をダウンロードできる
   * `Models`（または Artifact内の model_id） → 推論に使う **model_id** をコピー
5. **Inferenceタスク**に model_id を入力して実行
6. 推論モード（single / batch / optimize）ごとに、必要な出力が `Artifacts / Plots` に保存されている

### 1.2 開発者の最短導線

* 追加したい対象ごとに編集場所が **一意**である：

  * 前処理を追加 → `core/preprocessing/` か `registry/preprocessors.py`
  * モデル追加 → `registry/models.py`
  * 指標追加 → `registry/metrics.py`
  * 可視化追加 → `workflow/<phase>/visualization.py`
  * ClearMLログ変更 → `integrations/clearml/*` と `workflow/<phase>/clearml_integration.py`

---

## 2. “タスクは独立実行できる”を満たす全体設計

### 2.1 タスク一覧（責務分離）

| Task                                | 役割                              | 入力                               | 出力（次工程へ渡すもの）                                   |
| ----------------------------------- | ------------------------------- | -------------------------------- | ---------------------------------------------- |
| dataset_register                    | 生データを ClearML Dataset として登録（任意） | ローカルCSV等                         | dataset_id                                     |
| preprocess                          | dataset_id を受けて前処理 → 新Dataset作成 | dataset_id                       | preprocessed_dataset_id + preprocessing_bundle |
| train_parent                        | モデル候補を列挙し、子タスクで学習・評価を実行、結果集計    | dataset_id（raw or preprocessed）  | recommended_model_id / leaderboard.csv         |
| train_model (child)                 | 特定モデル1つを学習・評価（単独運用も可能）          | dataset_id + model_name + params | model_id + 評価指標 + 詳細PLOTS                      |
| infer_parent                        | 推論を束ねる（モード別子タスクを起動）             | model_id + inference_config      | mode別出力                                        |
| infer_single/batch/optimize (child) | 推論モードごとの実処理                     | model_id + mode_params           | 予測結果、最適化結果など                                   |

> **Pipeline実行** = 上記の独立タスクを順に繋ぐだけ（Pipelineのためにタスクが“別物”にならない）

---

## 3. ClearML UI設計（HyperParameters / Properties / Artifacts / Plots の置き場所ルール）

### 3.1 原則（重要）

* **HyperParameters**: “そのタスクの入力設定”だけ（ユーザーが編集して再実行したい項目だけ）
* **User Properties**: タスクの説明/検索/誤認防止のための “結果の要約・識別情報”

  * ClearMLでは User Properties はタスクの `Configuration > USER PROPERTIES` に表示され、後から編集可能（検索・列表示にも使える） ([ClearML][1])
* **Artifacts**: 長文/一覧/表/再現に必要なファイル（config全体、前処理レシピ、leaderboard、推論結果）
* **Plots**: 意思決定に必要な可視化のみ。

  * 親タスクは “比較・要約”
  * 子タスクは “詳細分析（散布図、SHAP、残差、重要度…）”

### 3.2 HyperParameters 汚染（関係ないパラメータが出る）を止める

**原因の典型**:

* `Task.init()` の自動収集（argparse 等）で “全CLI引数” が吸い上げられている
* 1つの巨大configを全タスクで `task.connect()` している（＝各タスクに不要な設定まで出る）

**対策（必須）**:

* `Task.init(..., auto_connect_arg_parser=False)` で自動吸い上げを止める

  * ClearMLは `auto_connect_arg_parser` で argparse由来の自動ロギングを制御可能 ([ClearML][2])
* 各タスクで `task.connect()` する辞書は **そのタスクの入力設定に限定**

  * それ以外（参照用情報、派生情報、全config）は Artifact（YAML/JSON）へ

---

## 4. データセット設計（前処理の再現性 + 逆変換 + 確認性）

### 4.1 Dataset “契約”（preprocessed dataset の中身）

前処理データセットは「学習・推論が必要な情報を必ず持つ」。推奨ファイル構成：

```
dataset/
  data_raw.parquet               # (任意) 元データ保持。容量が大きければ省略可
  data_processed.parquet         # 学習に使う前処理済み特徴量+目的変数
  schema.json                    # 入力列/目的列/型/ID列/分割キー等
  preprocessing/
    bundle.joblib                # fitted transformer一式（特徴・目的変換 + 逆変換）
    summary.md                   # 人間が読む要約（カテゴリ別）
    recipe.json                  # 変換手順（カテゴリ別、短文・構造化）
    version.txt                  # 前処理契約バージョン
  manifest.json                  # lineage（親dataset_id / task_id / run_id 等）
```

### 4.2 前処理の “カテゴリ別に確認できる” 表示（要件対応）

`preprocessing/summary.md` のテンプレ（例）：

```md
# Preprocessing Summary

## Dataset Lineage
- parent_dataset_id: <...>
- preprocessing_task_id: <...>
- created_at: <ISO8601>
- contract_version: v1

## Input Features
### Type Inference
- numerical: [...]
- categorical: [...]
- datetime: [...]
- dropped: [...]

### Missing Values
- strategy_numeric: median
- strategy_categorical: most_frequent
- columns_with_missing_top20:
  - col_a: 123 (4.1%)
  - col_b: 98  (3.2%)

### Encoding
- onehot: [col_x, col_y]
- ordinal: []
- handle_unknown: ignore

### Scaling
- method: standard
- columns: [...]

## Target
- target_column: y
- transform: standardize
- inverse_transform: enabled

## Split / Leakage Control
- split_key: <...|none>
- leakage_guard: <description>
```

> **学習タスク側**は dataset_id からこの `summary.md` / `recipe.json` を読み、
>
> * “1行の長文”ではなく、**カテゴリ別に整形した同じ構造**で `Artifacts` に再掲
> * さらに重要項目だけを **User Properties** に抜粋（例：target_transform, scaling, encoding）

---

## 5. Training設計（親タスクはシンプル、子タスクで詳細）

### 5.1 親タスク（Training Summary）の責務

* 全モデルの結果を集計し、**意思決定**に必要な最小情報だけを提示する
* **PLOTSは厳選**（多すぎる問題を根絶）
* 「推論に使う model_id がどれか」を一瞬で分かるようにする

**親タスクに残す Plots（推奨）**
（ClearMLの表示順を安定させるため **title prefix** を必須化）

* `01_Overview/RecommendedModel`（表 or テキスト）
* `02_Leaderboard/MetricBar`（主要指標でモデル比較）
* `03_Leaderboard/MetricTable`（上位Nの表）
* `04_Tradeoff/Metric_vs_Latency_or_ModelSize`（任意）
* それ以外（散布図、SHAP、残差…）は **一切置かない**

**親タスクの User Properties（例）**

* `run_id`
* `dataset_id`
* `dataset_role` = `raw` / `preprocessed`
* `target_column`
* `primary_metric`
* `recommended_model_name`
* `recommended_model_id`（推論用）
* `leaderboard_top1_score`

**親タスクの Artifacts（例）**

* `leaderboard.csv`（全モデル比較）
* `results_summary.csv`（評価詳細）
* `training_config.yaml`（再現用：ただし HyperParams には出さない）
* `preprocessing_summary.md`（再掲）
* `error_models.json`（一部モデル失敗時の理由）

### 5.2 子タスク（train_model）の責務

* 1モデルの学習・評価・可視化・モデル登録まで完結
* SHAP/散布図/残差/重要度など **重いPLOTSは子タスクに集約**
* 単独運用（このタスクだけ回す）でも成立する

**子タスクの HyperParameters（例）**

* `Input`: dataset_id, target_column
* `Model`: model_name, model_params（そのモデルに関係するものだけ）
* `Training`: cv, seed, early_stopping, etc（共通）
* `Evaluation`: metric list（共通）

**子タスクの Plots（例）**

* `01_Performance/Pred_vs_True`
* `02_Performance/Residuals`
* `03_Explain/FeatureImportance`
* `04_Explain/SHAP_Summary`
* `05_Diagnostics/...`

**子タスクの Model登録**

* ClearML Model Registry に `model_id` として登録し、親タスクで一覧化

---

## 6. Inference設計（モード別タスク + “用途別Artifacts/PLOTS”）

### 6.1 推論タスク構成

* `infer_parent`: どのモードを回したかの要約とリンク
* `infer_single`: 単発入力 → 予測値
* `infer_batch`: 複数入力 → CSV等で返す
* `infer_optimize`: 条件最適化 → 最適入力 + 目的値 + 探索ログ

### 6.2 モード別の出力ルール（例）

**infer_single**

* User Properties: `model_id`, `prediction`
* Artifacts: `input.json`, `output.json`
* Plots: 必要なら感度（1D sweep）など

**infer_batch**

* Artifacts: `predictions.csv`（必須）, `inputs.csv`（任意）
* Plots: 上位/分布、ヒストグラムなど（任意）

**infer_optimize**

* Artifacts: `best_solution.json`, `trials.csv`
* Plots: 目的値推移、パラレル座標（任意）

> 重要：推論結果は **目的変数の逆変換（標準化→元スケール）**を必ず適用して出力する
> （前処理bundleに target_transformer を含める）

---

## 7. 新ディレクトリ構成（提案：core / workflow / integrations / registry）

> “どこを触れば何が変わるか”を明確にし、拡張しやすくするための抜本変更

```text
automl_lib/
  __init__.py

  cli/
    __init__.py
    main.py
    commands/
      dataset_register.py
      preprocess.py
      train_parent.py
      train_model.py
      infer_parent.py
      infer_single.py
      infer_batch.py
      infer_optimize.py
      pipeline.py

  config/
    __init__.py
    models.py                # pydantic等で型定義（推奨）
    defaults/
      pipeline.yaml
      preprocess.yaml
      training.yaml
      inference.yaml

  core/
    data/
      io.py                  # Datasetロード・保存、parquet/csv対応
      schema.py              # 列型推論、target/feature定義
      validation.py          # 入力チェック
    preprocessing/
      bundle.py              # PreprocessingBundle（transform + inverse）
      build.py               # transformer構築
      apply.py               # fit/transform, transform-only
      report.py              # summary.md / recipe.json生成
    training/
      trainer.py             # 学習ループ（ClearML非依存）
      evaluation.py          # metric算出
      leaderboard.py         # 集計・並び替え
      model_io.py            # モデル永続化
    inference/
      predictor.py
      modes/
        single.py
        batch.py
        optimize.py

  integrations/
    clearml/
      task_factory.py        # Task.init/Createの統一窓口
      hyperparams.py         # connectする辞書を“最小化”する関数群
      properties.py          # set_user_properties統一
      artifacts.py           # upload_artifact統一
      datasets.py            # ClearML Datasetのcreate/get/lineage
      models.py              # Model Registry登録/参照
      links.py               # 親子/関連タスクのリンク情報を付与

  workflow/
    dataset_register/
      processing.py
      clearml_integration.py
      meta.py
    preprocess/
      processing.py
      clearml_integration.py
      visualization.py
      meta.py
    training/
      parent_processing.py
      model_processing.py
      clearml_integration.py
      visualization.py
      meta.py
    inference/
      parent_processing.py
      mode_processing.py
      clearml_integration.py
      visualization.py
      meta.py
    pipeline/
      controller.py          # pipeline orchestration only
      steps.py               # steps定義

  registry/
    models.py                # 追加はここ
    metrics.py               # 追加はここ
    preprocessors.py         # 追加はここ

tests/
  test_smoke_pipeline.py
  test_preprocess_contract.py
  test_training_leaderboard.py
  test_inference_inverse_transform.py

docs/
  REFORM_PLAN.md
  README_user.md
  README_dev.md
```

---

## 8. Config設計（“タスク別に必要な設定だけ”を扱う）

### 8.1 pipeline.yaml（例：ユーザーが触る最小）

```yaml
run:
  project: "AutoML"
  experiment_name: "AutoML Pipeline"
  tags: ["team:xxx"]
  clearml:
    enabled: true
    queue: "default"
    task_reuse: false

input:
  dataset_id: "CLEARML_DATASET_ID"
  target_column: "y"

preprocess:
  enabled: true
  missing_values:
    numeric: "median"
    categorical: "most_frequent"
  encoding:
    categorical: "onehot"
  scaling:
    numeric: "standard"
  target_transform:
    method: "standardize"

training:
  models: ["lgbm", "xgboost", "rf"]
  cv:
    folds: 5
    seed: 42
  metric:
    primary: "rmse"
    additional: ["mae", "r2"]
  save_models: true
  leaderboard:
    top_k: 5

inference:
  enabled: false
  model_id: ""
  mode: "single"
  single:
    input_json: "examples/single_input.json"
```

### 8.2 “HyperParametersに出す/出さない”のルール（実装側）

* `task.connect()` する辞書は、上記 config から **phaseに必要な部分だけ抽出**
* config全体は `task.connect_configuration(...)` か Artifact（`full_config.yaml`）で保存

  * UI上は “Configuration Objects / Artifacts” として参照用に置く（誤認防止）

---

## 9. ClearML安定化（Fail多発/タスク見えない/summary出ない問題への対処）

### 9.1 Task再利用で “上書き/見えない” を防ぐ

ClearMLは条件次第で Task を再利用するため、デバッグ中に「タスクが増えない/期待と違う」状態が起きやすい。
→ すべての本番/検証実行で `reuse_last_task_id=False` を明示する（推奨） ([ClearML][2])

### 9.2 “関係ないHyperParameters”が混入するのを防ぐ

* `auto_connect_arg_parser=False`（argparse由来の自動収集を無効化） ([ClearML][2])
* `Task.connect()` は **必要最小**の辞書のみ
* それ以外は User Properties / Artifactsへ

### 9.3 pipelineタスクが表示されない/Failする典型原因と対策

* **Agentがいない queue に enqueue**している

  * 対策: queue名の検証（起動時に “queue exists & has agent” をチェック）
  * 対策: Pipelineをローカル実行モードも選べる設計にして、Agentなしでも回せる
* **子タスク作成のやり方が不安定**（Task.init多重、close漏れ）

  * 対策: 1プロセスにつき “main taskは1つ” を守る（Taskの注意） ([ClearML][3])
  * 対策: 子タスクは `Task.create()` を基本にし、logger/ artifact をその task object へ明示的に流す

### 9.4 summaryが作成されない問題（設計で潰す）

* ある子タスクが失敗すると、親タスクが集計前に落ちる → summaryが出ない

  * 対策: 親タスクは “部分成功” を許容し、失敗モデルは `error_models.json` に記録して集計は継続
  * 対策: `leaderboard.csv` は **最後に必ず artifact upload + flush** する（finallyブロック）

---

## 10. 不要ファイル/処理の洗い出し → 削除の手順（確定プロセス）

> 「comparison など大量に不要があるはず」を **主観ではなく機械的に確定**する

### 10.1 “入口（エントリポイント）”を確定

* CLIコマンド一覧を固定（= public API）
* Pipelineが呼ぶステップ一覧を固定（= public API）

例（チェックコマンド）:

```bash
# CLIの入口（例）
python -m automl_lib.cli.main --help

# エントリポイントの参照箇所を抽出
rg -n "cli|commands|run_pipeline|Pipeline|controller" automl_lib
```

### 10.2 未使用コード候補抽出（静的解析）

```bash
python -m pip install vulture ruff

# 未使用関数/クラス候補（confidence高め）
vulture automl_lib --min-confidence 80

# import未使用/到達不能の粗取り
ruff check automl_lib
```

### 10.3 実行トレース（動的解析）で“到達しない”を確定

```bash
python -m pip install coverage

coverage run -m automl_lib.cli.commands.pipeline --config config/pipeline.yaml
coverage html

# coverage reportで 0% のファイル群を削除候補へ
coverage report -m
```

### 10.4 削除のルール（破壊的変更を安全に）

* **2段階削除**

  1. Deprecated化（CLIから外す、importを切る、docsから消す）
  2. 次のリリースで削除（テスト通過 + coverage 0% + 参照なし）
* `docs/DEPRECATIONS.md` を作り、削除の理由・代替手段を残す

---

## 11. 実装ステップ（優先順位つき）

### P0: “動く/見える/誤認しない”を最優先で修正（最短で効果が出る）

* [ ] 全タスクで `reuse_last_task_id=False` を統一
* [ ] 全タスクで `auto_connect_arg_parser=False` を統一（HyperParams汚染を止める）
* [ ] HyperParametersは phase別の最小辞書のみ `connect`
* [ ] 参照用の full config は Artifact化（`full_config.yaml`）
* [ ] Training Summaryが **必ず** `leaderboard.csv` を出す（finallyでupload）

### P1: “タスク独立性 + pipelineは接続だけ” をコード構造で固定

* [ ] workflow層（processing）を “単独実行” できる形にする
* [ ] pipelineは processing関数を順に呼ぶだけ（処理分岐をなくす）
* [ ] train_parent は常に child task 生成 + 集計（standalone/pipeline差を消す）

### P2: PLOTS整理（親は比較、子に詳細）

* [ ] 親タスクから per-model SHAP/散布図/残差等を削除
* [ ] 子タスクへ集約
* [ ] plot title prefix（`01_...`）で順序保証

### P3: データ契約（preprocess bundle / inverse transform）を固定

* [ ] `PreprocessingBundle`（transform/inverse）をjoblibで保存
* [ ] summary.md / recipe.json をカテゴリ別で生成
* [ ] training/inference が bundle を読むだけで整合するよう統一

### P4: ディレクトリ再編 + 不要コード削除

* [ ] `core/ workflow/ integrations/ registry/` を導入
* [ ] 旧パスは段階的に移行（import wrapperで壊さない）
* [ ] vulture + coverage で削除確定 → 物理削除

### P5: README刷新（ユーザー/開発者分離）

* [ ] `README_user.md`: dataset_id→pipeline→model選定→推論の一本道
* [ ] `README_dev.md`: 拡張ポイント、設計ルール、タスクI/O契約

---

## 12. README草案（アウトライン）

### 12.1 README_user.md（非MLユーザー向け）

```md
# AutoML on ClearML（ユーザーガイド）

## 1. できること
- dataset_id を入れて実行 → 複数モデル比較 → おすすめモデルを選択 → 推論

## 2. 最短の使い方（3ステップ）
1) ClearMLのDataset IDを用意
2) pipeline.yaml に dataset_id を入れる
3) run_pipeline を実行

## 3. ClearMLでどこを見れば良いか
- Training Summary の Plots: RecommendedModel / Leaderboard
- Configuration: Trainingだけ（余計な設定は出ません）
- Artifacts: leaderboard.csv から詳細確認
- 推論は model_id をコピペして Inference を実行

## 4. 推論モード
- single / batch / optimize の違いと出力場所

## 5. よくあるトラブル
- pipelineが動かない（queue/agent）
- taskが増えない（task reuse）
```

### 12.2 README_dev.md（開発者向け）

```md
# Developer Guide

## 1. Architecture
- core / workflow / integrations / registry

## 2. Phase I/O Contract
- Dataset contract（manifest, preprocessing bundle）
- Training outputs（leaderboard, model registry）
- Inference outputs（mode別 artifacts）

## 3. Extension Points
- Add model: registry/models.py
- Add metric: registry/metrics.py
- Add preprocessing: core/preprocessing or registry/preprocessors.py
- Add plots: workflow/<phase>/visualization.py

## 4. ClearML Logging Rules
- Hyperparametersは最小
- 参照情報は artifacts / user properties
- 親は比較、子は詳細

## 5. Testing
- smoke tests
- contract tests
```

---

## 13. Doneの定義（受け入れ条件）

* [ ] dataset_id だけで pipeline 実行 → ClearML上で Training Summary が必ず生成される
* [ ] Training Summary の `HyperParameters` に **関係ない項目が出ない**
* [ ] Training Summary の `Plots` が “モデル比較と推奨” だけでスッキリしている
* [ ] 子タスクに SHAP/散布図/残差が集約され、探索可能
* [ ] 推論は model_id 指定で single/batch/optimize が迷わず回り、Artifactsが用途通りに保存される
* [ ] 前処理内容がカテゴリ別に確認でき、学習/推論で同じ情報が参照できる
* [ ] 不要コードが vulture + coverage により削除確定され、リポジトリが簡素化されている

---

## References（ClearML仕様の根拠）

* User Properties は後から編集でき、Configuration内で扱える ([ClearML][1])
* `Task.init(..., auto_connect_arg_parser=...)` で argparse由来の自動ロギングを制御できる ([ClearML][2])
* “main execution Task は1プロセスにつき1つ” の注意 ([ClearML][3])

---


[1]: https://clear.ml/docs/latest/docs/fundamentals/hyperparameters/?utm_source=chatgpt.com "Hyperparameters"
[2]: https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/?utm_source=chatgpt.com "Task"
[3]: https://clear.ml/docs/latest/docs/references/sdk/task/?utm_source=chatgpt.com "Task"



推論タスクに関して以下の点を改良、拡張してください。
推論タスクのConfigurationタブで以下を追加してください
＃単一条件の場合：Hyperparametersの項目に入力条件を各変数ごとに表示
＃Optimizeの場合：inference-summaryタスクのHyperparametersの項目に入力条件を各変数ごとの範囲や設定、最適化のアルゴリズム、目的（最大、最小など）

推論タスクのPlotsタブで有用なグラフをいくつか追加してください。
＃[共通]推論タスクの条件と推論結果をPlotsテーブルにまとめて表示
＃[共通]推論タスク共有で、学習データの内装範囲に対して、推論条件がどのあたりにあるかを散布図グラフでわかるように。
＃Optimizeの場合、inference-summaryタスクに目的（最大、最小など）に応じたTopKを条件と目的変数を表示。Loss historyプロット（縦軸は対数スケール）,contourプロット、並行座標プロット、Feature importanceプロット、contourプロットをグラフ表示
＃Plotsのグラフはナンバリングして優先度ごとにわかりやすいように

単一条件の推論の場合、inference-summaryタスクは生成しないようにしてください。
Optimizeの場合親タスクをinference-summarタスク、子タスクを各条件に対応するタスクとして、Trainの子タスクと同じように"Prediction_runs"というディレクトリを作成し、その配下に子タスクを配置。子タスクの内容は単一条件の推論の内容と一致させるように

Optimizeの場合のyaml例を新たに作成してください
また上記の変更がユーザー視点での内容把握しやすい仕様、開発者視点で保守・拡張しやすいように注意して実装し、実装後残件、改良すべき点を表示してください


＃preprocessed-datasetのデータセットタスクのCONFIGURATIONタブのHYPERPARAMETERSに今の表示情報に加えてyamlの前処理に関わる項目の変数と設定値を保存、ひょうじするようにしてください。以下は対象のyaml例です
preprocessing：
  numeric_imputation: ["mean"]
  categorical_imputation: ["most_frequent"]
  scaling: ["standard"]
  categorical_encoding: ["onehot"]
  polynomial_degree: false
  target_standardize: true

＃training-summaryタスクのCONFIGURATIONタブのHYPERPARAMETERSに今の表示情報に加えてyamlの学習に関わる項目の変数と設定値を保存、ひょうじするようにしてください。以下は対象のyaml例です
models:
  name: LinearRegression
    params: {}
  name: Ridge
    params:
      alpha: [0.1, 1.0]
  name: RandomForest
    params:
      n_estimators: [50]
      max_depth: [null]

ensembles:
  stacking:
    enable: false
    estimators: []
    final_estimator: null
  voting:
    enable: false
    estimators: []
    voting: "hard"

cross_validation:
  n_folds: 5
  shuffle: true
  random_seed: 42

evaluation:
  regression_metrics: ["mae", "rmse", "r2"]
  classification_metrics: ["accuracy", "f1_macro", "roc_auc_ovr"]
  primary_metric: null

optimization:
  method: "grid"
  n_iter: 100