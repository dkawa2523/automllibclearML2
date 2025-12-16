# memo（運用のぶれ対策 / 利用促進 / 開発者ガイド）

このメモは「ML非専門ユーザーが迷わず使える運用」と「開発者が安全に拡張できる開発導線」を明文化するためのドラフトです。

---

## 1. ユーザー側：操作・活用シーンの“ぶれ”を抑える施策（迷子/属人化防止）

### 1.1 “黄金導線”を固定（ユーザーに見せる正解ルートを1つにする）
- 入口は原則 `pipeline`（ClearMLモード）に統一し、最後に `reporting` タスクを必ず見る導線にする
- ユーザーが見るタスクは基本この2つだけに寄せる
  - `reporting`（結論と意思決定用の要約）
  - 推奨モデルの `training child task`（詳細：SHAP/特徴量重要度/予測プロット等）

### 1.2 迷子にならないための “検索キー” を統一（ClearML UI運用の型）
- タスク/データセット探索はタグ `run:<run_id>` を最優先（プロジェクト階層が揺れても追える）
- 追加の絞り込みタグ
  - `dataset:<dataset_key>`
  - `phase:<phase>`
- 最後の成果物は `report.md`（ローカル）＋ ClearML の `00_report/report_md`（Task内テキスト）に集約し、Slack/メール共有はここへのリンクで統一

### 1.3 “意思決定”の観点をブレさせない（精度/速度/サイズ）
- `reporting` で必ず以下を提示（見る順を固定）
  1) 推奨（Decision Summary）
  2) TopK（精度/速度/サイズ）
  3) 代替案（最速/最小/pareto候補）
- 追加の現場要件（速度重視/軽量重視）が出ても、`Decision Summary` と `Alternatives` を見れば意思決定できる状態を維持

### 1.4 よくある運用ブレのパターンと抑止
- “training-summaryを見ればいいのか、comparisonを見ればいいのか分からない”
  - 原則：`reporting` を見れば結論に到達できる、詳細は推奨モデルの子タスクへ
- “どのRunの成果物か分からない”
  - `run_id` と `dataset_key` を必ず Task名/タグ/Markdownタイトルに含める（検索の最短）
- “遠隔実行で成果物が欠ける”
  - `report.md` の `Debug: Missing artifacts` と ClearML の `99_debug/missing_artifacts` を一次確認にする

---

## 2. ユーザー側：利用促進のための施策（習慣化/共有/社内展開）

### 2.1 共有物を固定（“報告書として渡せる”形）
- 共有の基本単位を `reporting` タスクに統一（URL + `report.md` artifact）
- 業務担当向けの説明は `report.md` で完結（モデルの詳細は必要なときだけ子タスクへ）

### 2.2 テンプレ化（再実行を楽にして利用回数を増やす）
- 代表的な設定（データセット別/目的別）をテンプレTaskとして残す
- 再実行は clone（`clearml.execution_mode=clone`）を推奨し、再現性と運用負荷を両立

### 2.3 KPI（利用が進んでいるかの簡易指標）
- “意思決定までの時間”が短くなっているか（reportingに到達しているか）
- `run:<run_id>` タグで追えるタスクが揃っているか（散乱していないか）
- `Alternatives` を見て「用途別に選べた」記録が残っているか（議事録/チケットに貼れるか）

---

## 3. 開発者側：開発効率化・改良施策（拡張しやすさ/安全性/レビュー容易性）

### 3.1 設計の原則（守ると壊れにくい）
- **Run単位のキー**：`run_id`（タグ `run:<run_id>`）を全フェーズで一貫利用
- **命名/タグ**：`automl_lib/clearml/naming.py` の規約に寄せ、個別実装でブレさせない
- **拡張は registry / plugins 優先**：プロジェクト固有の追加は `automl_lib/plugins/` に寄せる
- **Pipeline worker の import 安全性**：重い依存は関数内 import（`automl_lib/phases/__init__.py` は lazy import）
- **可観測性**：成果物が欠けたら reporting の debug に出す（運用が詰まらない）

### 3.2 “開発者が見るべきファイル” 早見表（拡張内容別）

#### A) 前処理機能を拡張したい（新しい変換/スケーリング/エンコード等）
- 追加する場所（推奨順）
  - `automl_lib/plugins/`（プロジェクト固有の追加）
  - `automl_lib/registry/preprocessors.py`（共通化して良いもの）
- 前処理パイプライン生成（どう組み立てられるか）
  - `automl_lib/preprocessing/preprocessors.py`
- フェーズ実行・成果物（ClearML/outputs）
  - `automl_lib/phases/preprocessing/processing.py`
  - `automl_lib/phases/preprocessing/visualization.py`（前処理のPlots/テーブル）
  - `automl_lib/phases/preprocessing/meta.py`（前処理のmeta/artifacts）
- 設定・バリデーション（typo検知）
  - `automl_lib/config/schemas.py`（`preprocessing.*` と `preprocessing.plugins`）

#### B) 学習モデルを追加/差し替えたい（新アルゴリズム、optional依存の追加等）
- モデル登録（名前解決/alias/optional import）
  - `automl_lib/registry/models.py`
  - 必要なら `automl_lib/plugins/` で登録モジュールを追加（import時に register）
- モデル生成と学習ループ
  - `automl_lib/training/model_factory.py`
  - `automl_lib/training/run.py`
- 評価・探索・アンサンブル
  - `automl_lib/training/evaluation.py`
  - `automl_lib/training/search.py`
  - `automl_lib/training/ensemble.py`
- ClearML の親子タスク/成果物/スカラー
  - `automl_lib/training/clearml_integration.py`
  - `automl_lib/training/reporting/scalars.py`

#### C) 評価指標（metrics）を追加したい（派生指標含む）
- 指標登録（loss扱い/派生指標）
  - `automl_lib/registry/metrics.py`
- plugin で追加する場合
  - `automl_lib/plugins/example_metrics.py`
  - config: `evaluation.plugins`
- 影響箇所（ランキング/推奨/レポート）
  - `automl_lib/training/run.py`（ランキング用の列/推奨決定）
  - `automl_lib/phases/reporting/processing.py`（Decision/TopK 表示）

#### D) PLOTS/Artifacts 等の情報付与を増やしたい（見やすさ改善）
- 前処理の可視化
  - `automl_lib/phases/preprocessing/visualization.py`
- 学習の可視化（SHAP/特徴量重要度/散布図など）
  - `automl_lib/training/interpretation.py`
  - `automl_lib/training/visualization.py`
  - `automl_lib/training/reporting/plots.py`
- レポート（最終的にユーザーが見るまとめ）
  - `automl_lib/phases/reporting/processing.py`
- ClearML への出力API（共通ヘルパ）
  - `automl_lib/clearml/logging.py`（`report_table/report_plotly/report_image/upload_artifacts`）

#### E) run_id / 命名 / タグ / Clone / Pipeline を変えたい（運用の骨格）
- run_id / dataset_key / run-scoped output
  - `automl_lib/clearml/context.py`
  - `automl_lib/types/phase_io.py`
- 命名/タグ規約
  - `automl_lib/clearml/naming.py`
- Clone 実行
  - `automl_lib/clearml/clone.py`
  - `automl_lib/cli/run_clone.py`
- PipelineController（ステップ連結/キュー/親子リンク）
  - `automl_lib/pipeline/controller.py`
  - `automl_lib/cli/run_pipeline.py`

### 3.3 開発フロー（最小の品質ゲート）
- 変更後に実行
  - `./.venv/bin/python -m compileall -q automl_lib`
  - `./.venv/bin/python -m unittest discover -s tests -q`
- 新しい拡張点を追加したら、dict/CSVの形（列名）だけでもテストで固定（破壊的変更の検知）

---

## 4. 今後の改良候補（優先度高→低の例）

### 優先度 高（運用で詰まりやすい）
- `model_tasks_ranked.csv` が無い場合の reporting フォールバック（`results_summary.csv` から最低限のTopK生成）
- training で成果物生成失敗を reporting の `missing_artifacts` に確実に反映（例外握り潰しを減らす/警告を残す）
- データプレビュー（PIIマスキング付き）を reporting に任意追加（`reporting.preview_rows` 等で制御）

### 優先度 中（拡張しやすさ）
- `models.plugins` の導入（models追加を plugins 経由で公式化）
- レポートの“用途別おすすめ”（精度/速度/サイズの3枚看板）を固定フォーマット化

### 優先度 低（将来：Phase 9）
- 推論タスクの見やすさ改善（単一/複数条件/最適化のUI改善、今回は設計のみ）

