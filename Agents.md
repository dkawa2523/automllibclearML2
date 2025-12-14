# automl_lib_clearML 拡張コード実装計画（Codex / VSCode向け・段階実装）

> 目的：**ユーザー（ML非専門）**が ClearML 上で「結果の把握・参照・業務活用」まで迷わず辿れること。
> 目的：**開発者**が ClearML/各ライブラリ仕様を深掘りしなくても、拡張ポイント（前処理/モデル/評価/可視化/情報付与）を守れば安全に追加できること。
> 注意：**「Phase 9（推論タスクの単一/複数条件/最適化の“見やすさ改善”）」は将来実装**。今回の実装には含めず、**将来実装しやすい設計（データ構造・命名・タグ・I/Oの布石）**に留めます。

---

## 0. 現状アーキテクチャ要約（前提の共有）

このリポジトリは概ね以下の責務分割になっています（README_dev とコードからの整理）：

* `automl_lib/cli/`

  * `run_<phase>.py` が入口（config を読み、フェーズ処理を呼ぶ）
* `automl_lib/phases/<phase>/processing.py`

  * 各フェーズの主処理（前処理、学習、比較…）
* `automl_lib/phases/<phase>/clearml_integration.py`

  * ClearML Task生成、親子リンク、Artifacts/Hyperparameters登録（現状は薄い）
* `automl_lib/clearml/`

  * `utils.py`：`init_task` / `create_child_task` / model登録 / dataset登録など
  * `datasets.py`：dataset local copy / hash tag / CSV探索など
  * `logging.py`：`report_hyperparams()` など（現状は `task.connect(dict)` で **General 1枚**になりがち）
* `automl_lib/pipeline/controller.py`

  * `PipelineController` or `in_process` の実行モード

---

## 1. 今回の実装で達成する「大きな成果物（ユーザーに見える変化）」

### ユーザー視点（ClearML UIで起きる改善）

* タスク/データセットが散らからず、**“このRunに関係するものだけ”**をすぐ追える
  → **run_id（タグ/命名/プロジェクト規約）**で統一
* Configuration → Hyperparameters が **General 1枚から「カテゴリ分割」**され、設定も結果追跡も簡単
  → Data / Preprocessing / Model / CV / Evaluation / ClearML / Output…など
* 前処理タスクに **データ分布や欠損、ターゲット分布等のPlots/テーブル**が出る
  → “何が起きたか”をML非専門でも把握できる
* 学習タスク（親・子）に **比較しやすいScalars（指標・時間）**が揃い、考察が進む
  → Comparison は “まとめ” だが、個別タスクでも理解できる
* 比較タスクで大量組合せを **TopK・モデル別集計・勝率**などで俯瞰できる（Artifacts / Plots / Tables を強化）

### 開発者視点（拡張しやすい変化）

* 「どこを編集すれば、Plots/Scalars/Artifacts/Debug samplesを追加できるか」が明確
  → phaseごとに `meta.py / visualization.py / clearml_integration.py` を標準化
* “RunContext（run_id / project / tags / dataset_key）” を中心に設計
  → 機能追加でも命名・タグがブレない
* （重要）**Clone実行モード**の導入で、テンプレタスクを複製して即実行できる

---

## 2. 実装ロードマップ（Phase 1〜8が今回対象、Phase 9は設計のみ）

### Phase 1：RunContext導入（全拡張の土台）

**狙い**：タスク/データセットの散乱を止める「共通キー」を作る（run_id・タグ・命名の一貫性）。

#### 追加/修正するもの

* ✅ 新規：`automl_lib/run_context.py`（or `automl_lib/clearml/context.py`）
* ✅ 修正：`automl_lib/config/schemas.py`（`run.*` と `clearml.naming.*` を追加）
* ✅ 修正：`automl_lib/pipeline/controller.py`（run_id を pipeline 全体で維持）
* ✅ 修正：`automl_lib/types/phase_io.py`（DatasetInfo/TrainingInfo/ComparisonInfo に `run_id?: str` を追加：**optional**）

#### RunContext（最小仕様）

```python
@dataclass(frozen=True)
class RunContext:
    run_id: str                 # 例: 20251213-153012-a1b2c3
    user: str                   # OS user or config
    project_root: str           # clearml.project_name
    dataset_project: str        # clearml.dataset_project
    dataset_key: str            # dataset_id短縮 or csv stem
    tags_base: list[str]        # ["run:<run_id>", "dataset:<dataset_key>"]
```

#### run_id 生成規約（デフォルト）

* `YYYYMMDD-HHMMSS-<6hex>`（衝突しづらく、UIで時系列追える）
* ただし config に `run.id` が指定されればそれを優先

#### パイプライン/フェーズ間で run_id を維持する方法

* `run_pipeline` の入口で run_id を確定し、環境変数に入れる

  * `AUTO_ML_RUN_ID=<run_id>`
* PipelineController の function step にも `run_id` を `function_kwargs` で明示的に渡せるようにする（将来の堅牢化）

#### Done（受け入れ条件）

* 同じ pipeline 実行で作られたタスク/データセットに `run:<run_id>` が付く（次Phaseで実装）
* `--output-info` の JSON に `run_id` が入る（optional）

---

### Phase 2：命名/タグ規約の実装（散らかり対策の本丸）

**狙い**：ユーザーが ClearML で迷子にならない。プロジェクト階層がユーザーごとにブレても、最低限 run_id で追える。

#### 追加/修正するもの

* ✅ 新規：`automl_lib/clearml/naming.py`
* ✅ 修正：`automl_lib/clearml/utils.py::init_task()`（tagsの付与を標準化できるように）
* ✅ 修正：各 `phases/*/clearml_integration.py`（create_*_task で naming/tags を利用）
* ✅ 修正：`clearml/datasets.py::register_dataset_from_path()` 呼び出し側（dataset name/tags を統一）

#### 命名規約（提案）

* **Task名**（例）

  * preprocessing: `preprocessing [<dataset_key>] [<run_id>]`
  * training summary: `training-summary [<dataset_key>] [<run_id>]`
  * child model: `train [<preproc_id>] [<model_name>] [<run_id>]`
  * comparison: `comparison [<dataset_key>] [<run_id>]`
* **Project階層**（デフォルトは “深くしすぎない”）

  * `AutoML/<user>/<dataset_key>`
  * 子タスクだけは `.../train_models` に逃がす（既存実装を踏襲しつつ改善）
* **Tags**（最低限）

  * `run:<run_id>`
  * `phase:<phase>`
  * `dataset:<dataset_key>`
  * `preprocess:<preproc_id>`（ある場合）
  * `model:<model_name>`（ある場合）

#### “プロジェクトがまちまち”問題への施策（複数）

* **施策A（必須）**：run_idタグで追える（検索・絞り込みの最小保証）
* **施策B（推奨）**：project path builder を導入して default を統一（ただし override 可）
* **施策C（オプション）**：`clearml.tags` に `team:<...>` など追加できる設計

#### Done

* preprocessing/training/comparison で Task名と tags が規約通りになる
* training child task の名前に `preproc_id` を含めて区別できる

---

### Phase 3：Clone実行モード（テンプレ複製で “毎回作るコスト” を削減）

**狙い**：一度作ったタスク（前処理/学習/比較/推論/パイプライン）をテンプレとして複製し、最小変更で再実行できる。

> 重要：ClearML UI でも clone は可能ですが、**CLI/設定から clone → enqueue を一気通貫**にすると「ML非専門」でも迷わず回せます。

#### 追加/修正するもの

* ✅ 新規：`automl_lib/clearml/clone.py`（or `utils.py` に追加）
* ✅ 新規：`automl_lib/cli/run_clone.py`（テンプレtask_id を渡して複製実行）
* ✅ 修正：`config/schemas.py`

  * `clearml.execution_mode: "new" | "clone"`
  * `clearml.template_task_id: str?`（フェーズ共通でまず1個）
  * （将来）`clearml.templates.{preprocessing,training,comparison,inference,pipeline}` に拡張可能な設計にする

#### CLI 仕様（最小）

* `python -m automl_lib.cli.run_clone --task-id <TEMPLATE_TASK_ID> --queue <queue?> --overrides <json/yaml?>`

overrides 例（将来の拡張前提で設計だけ）

```yaml
overrides:
  data.dataset_id: "xxxxxxxx..."
  preprocessing.scaling: ["standard"]
  models[0].params.n_estimators: 500
```

#### 実装方針

* ClearML API の `Task.clone()` を利用（clone → 必要ならパラメータ上書き → enqueue）
* cloneされたタスクには

  * `run:<new_run_id>` を付与
  * `cloned_from:<template_task_id>` タグを付与
* **new_run_idは必ず発行**（テンプレと同一runに混ざらない）

#### Done

* テンプレ task_id を指定すると clone が作成され、queue に投入できる
* clone後の task に run_id/tag が付与される

---

### Phase 4：Hyperparameters をカテゴリ分割（General 1枚問題を解消）

**狙い**：設定が見やすく、結果の追跡も「何を変えたか」が一瞬で分かる。

#### 追加/修正するもの

* ✅ 修正：`automl_lib/clearml/logging.py`

  * `report_hyperparams()` を拡張（互換維持）
  * 新規：`report_hyperparams_sections(task, sections: dict[str, dict])`
* ✅ 修正：各 `phases/*/clearml_integration.py`

  * `report_hyperparams(task, config)` から `report_hyperparams_sections()` へ移行
* ✅ 修正：training child task にもカテゴリ分割を適用（Modelセクションを強調）

#### セクション案（例）

* `Data`
* `Preprocessing`
* `Model`
* `CrossValidation`
* `Evaluation`
* `Optimization`
* `Output`
* `ClearML`

#### 具体の変換ルール（Codex実装ガイド）

* pydantic config の `cfg.model_dump()` をそのまま繋ぐのではなく、部分dictを切り出す
* list/dict が複雑な部分（例：`models[]`）は

  * summary task：`models` を “表示用に整形した dict” にして `ModelCandidates` セクションへ
  * child task：`Model` セクションは “そのタスクが担当する1モデルのparamsのみ”

#### Done

* preprocessing / training-summary / child training / comparison で Hyperparameters が複数カテゴリに分かれる

---

### Phase 5：前処理タスクのPlots/Artifacts拡張（ユーザーが理解できる前処理へ）

**狙い**：「前処理で何が起きたか」をML非専門でも追える。

#### 追加/修正するもの

* ✅ 新規：`automl_lib/phases/preprocessing/visualization.py`
* ✅ 新規：`automl_lib/phases/preprocessing/meta.py`（軽量でもOK）
* ✅ 修正：`automl_lib/phases/preprocessing/processing.py`

  * 前処理後に `visualization` と `meta` を呼び、ClearML に出す

#### Plots案（優先順）

1. **ターゲット分布**（分類：クラス比、回帰：ヒスト/箱ひげ）
2. **欠損サマリ**（列ごとの欠損数/率 上位N）
3. **数値列の分布サマリ**（代表列のみ or サマリ統計）
4. **前処理後特徴量数**（元の特徴量数→変換後の次元）
5. （オプション）相関ヒートマップ（列が多い時は上位Nに制限）

#### Artifacts案（小カテゴリ化）

* `schema_raw.json`（列型推定、カテゴリ列、数値列）
* `preprocess_pipeline.txt` or `preprocess_pipeline.json`（適用した手順）
* `preprocessed_features.csv` は既に出ているが、名前に run/dataset_key を入れるのも検討

#### Debug samples案

* “変換前→変換後のサンプル行”を数行（PIIに注意。必要ならマスク関数を用意）

#### Done

* ClearML の preprocessing task の Plots に最低3種の可視化が出る
* Artifacts に schema/pipeline 情報が出る

---

### Phase 6：学習タスク（親・子）の “比較/考察しやすさ” を強化（Scalars/Plots/Debug）

**狙い**：大量の組合せでも「どれが良いか」「なぜ良いか」を追える。

#### 追加/修正するもの（設計方針）

* ✅ 新規：`automl_lib/training/reporting/`（例）

  * `report_scalars.py`（メトリクス/時間/サイズ）
  * `report_plots.py`（混同行列/ROC/残差など）
  * `report_debug_samples.py`
* ✅ 修正：`automl_lib/training/run_automl`（内部）

  * child task作成時に `run_context / tags / hyperparam sections` を適用
  * 指標は `Logger.report_scalar` に統一タイトル規約で送る
* ✅ 修正：`automl_lib/phases/training/clearml_integration.py`

  * child task name を `train [preproc] [model] [run_id]` にする
  * child task tags に `preprocess:<...>` `model:<...>` を付与

#### Scalars（比較の土台）タイトル規約（例）

* `metric/<metric_name>`（single value：CV mean or holdout score）
* `time/train_seconds`
* `time/predict_seconds`
* `model/num_features`
* `model/num_rows_train`
* `model/size_bytes`（保存している場合）
* `model/params_count`（可能なら）

> こうしておくと、ClearML の “Scalar Compare” で多タスク比較しやすい。

#### Plots（タスク種別に応じて）

* regression：

  * predicted vs actual
  * residual histogram
  * residual scatter
* classification：

  * confusion matrix（正規化版も）
  * ROC / PR curve（可能なら）
* 共通：

  * feature importance（モデルが対応なら）
  * SHAP（optional依存、ある時だけ）

#### Done

* 各 child task に scalars が揃っており、ClearML UI上で比較が成立する
* training-summary に「最良モデル」のサマリ（table/markdown/artifact）が残る

---

### Phase 7：学習比較・評価タスクの拡張（大量組合せを “俯瞰” できる）

**狙い**：比較タスクが「ランキング表を吐くだけ」ではなく、ユーザーが意思決定できる形にする。

#### 追加/修正するもの

* ✅ 修正：`automl_lib/phases/comparison/processing.py`

  * TopKテーブルの report
  * best_by_model / win_summary の report を強化
* ✅ 修正：`automl_lib/phases/comparison/visualization.py`

  * “モデル×前処理” のヒートマップ（指標別）
  * run間比較（既にあるなら見せ方の改善）

#### 追加するClearML出力（例）

* Plots：

  * heatmap（metric）
  * model_summary bar chart（平均/分散）
  * win_summary（勝率）
* Tables：

  * TopKランキング
  * モデル別最良
* Scalars：

  * comparison/best_<metric>
  * comparison/topk_mean_<metric>（任意）

#### Done

* comparison task を開けば「結論（推奨モデル）」と「根拠（ランキング/集計/図）」が揃う

---

### Phase 8：開発者の拡張容易性（ファイル配置/ガイド/テスト）

**狙い**：拡張ポイントが明確で、レビューもしやすい構造にする。

#### 追加/修正するもの

* ✅ 新規：`docs/DEV_GUIDE.md`（README_dev を補完でもOK）
* ✅ 新規：`automl_lib/plugins/`（プロジェクト固有拡張の置き場）
* ✅ 整理：`training/` 内の責務分割（reporting, evaluation, model_factory を明確化）
* ✅ テスト追加：`tests/` に

  * RunContext生成のテスト
  * naming/tags のテスト
  * hyperparam sections 生成のテスト（ClearML無しでも dict の形を検証）
  * preprocessing visualization の “返り値/生成物” テスト（プロットは生成可否だけ）

#### ディレクトリ構成（提案：段階導入でOK）

```text
automl_lib/
  clearml/
    bootstrap.py
    datasets.py
    logging.py
    utils.py
    naming.py        # NEW
    clone.py         # NEW
    context.py       # NEW (RunContext)
  phases/
    preprocessing/
      processing.py
      clearml_integration.py
      visualization.py   # NEW
      meta.py            # NEW
    training/
      processing.py
      clearml_integration.py
    comparison/
      processing.py
      clearml_integration.py
      visualization.py
      meta.py
  training/
    run_automl.py (or run.py)
    reporting/           # NEW
      scalars.py
      plots.py
      debug_samples.py
```

#### Done

* “どこを触れば何が変わるか”が分かる状態（新規開発者が迷わない）
* 拡張のための最小テンプレ（preprocessor/model/metric/plot）が docs にまとまっている

---

## 3. Phase 9（将来）：推論タスク見やすさ改善（今回は実装しない、設計のみ）

**今回やらないこと（明確化）**

* 推論タスク（単一条件/複数条件/最適化）それぞれの UI 改善の実装はしない
* ただし、将来やりやすくするために以下は “今回の設計に織り込む”

### 今回の設計に入れる “布石”

* `InferenceInfo` に `run_id` を持てる（optional）
* 推論タスクでも `RunContext` / naming / tags が自然に適用できる設計
* `automl_lib/phases/inference/` を phase標準構造（processing/meta/visualization/clearml_integration）で追加できる前提にしておく
  ※ directory は作っても良いが、中身は最小スタブに留める（今回の実装対象外）

---

## 4. PR分割（Codexが実装しやすい“粒度”の提案）

### PR1：RunContext + 命名/タグ（Phase1-2）

* [ ] `RunContext` 実装
* [ ] `naming.py` 実装
* [ ] preprocessing/training/comparison の create_task が命名規約を使用
* [ ] datasets 登録タグに run_id を追加

### PR2：Clone実行モード（Phase3）

* [ ] `clearml/clone.py` 追加
* [ ] `cli/run_clone.py` 追加
* [ ] config に execution_mode / template_task_id（最小）を追加

### PR3：Hyperparameterカテゴリ分割（Phase4）

* [ ] `report_hyperparams_sections` 追加
* [ ] preprocessing/training/comparison で sections を適用
* [ ] child task の Model セクション分離

### PR4：前処理Plots/Artifacts（Phase5）

* [ ] preprocessing visualization/meta 追加
* [ ] processing から呼ぶ
* [ ] Debug samples（任意）

### PR5：学習のScalars/Plots強化（Phase6）

* [ ] training/reporting を作成
* [ ] run_automl 側に組み込み
* [ ] “比較できる scalar 命名規約” を統一

### PR6：比較タスクの俯瞰強化（Phase7）

* [ ] TopK/heatmap/win_summary の強化
* [ ] Tables/Scalars を増やす

### PR7：開発者ガイド/構造/テスト（Phase8）

* [ ] docs 追加
* [ ] tests 拡充
* [ ] plugins 置き場追加

---

## 5. 実装上の運用課題と対策（今回の実装で織り込む/織り込まない）

### 想定課題

* ClearMLプロジェクト/データセットが増えすぎる
  → **run_idタグ + 命名規約**が最低限の防波堤
* 同じデータの重複登録
  → registration/editing は hash tag を利用済み。preprocessing も可能なら hash（入力dataset_id + preprocess設定）で再利用設計を検討
* PipelineController 環境差（GPU無しで ResourceMonitor がうるさい）
  → 既存の `disable_resource_monitoring` の運用を継続
* 指標・可視化が増えて重い
  → TopK制限、サンプリング（N行）、列数が多い場合の上位列だけ、を標準化

### 今回 “やらない” 対策（将来）

* 自動アーカイブ（古いタスクを自動で archive）
  → 誤爆リスクがあるため、将来オプションで設計するなら `clearml.cleanup.*` として追加

---

## 6. Codexに渡す「実装順の最短ルート」

1. **RunContext（run_id）**を作る
2. **命名/タグ**を preprocessing/training/comparison に適用
3. **Hyperparameter sections** を導入（General 1枚を潰す）
4. **Preprocessing 可視化**を追加（最もユーザー価値が高い）
5. **Training child scalars** の規約を統一（比較しやすさの核心）
6. **Comparison 俯瞰** を強化
7. **Clone mode**（テンプレ複製）を入れて運用負荷を落とす
8. 最後に docs/tests を固める

> ※ Clone は早く入れたくなりますが、命名/タグ/sections が無い状態で clone を回すと “増殖が加速” します。
> なので **「RunContext→命名/タグ→sections」→ clone** が安全です。

---

必要なら、この計画をそのまま `docs/IMPLEMENTATION_ROADMAP.md` として貼り付けられる体裁（章立て・チェックリスト）にしてあります。
次のステップとしては、PR1（RunContext + 命名/タグ）の **具体的な差分設計（どの関数にどの引数を増やすか、どのタグをいつ付けるか）**まで落とした “Codex実装用のTODOリスト（ファイル別）” も作れますが、まずはこのロードマップを基準に進めるのが最短です。
