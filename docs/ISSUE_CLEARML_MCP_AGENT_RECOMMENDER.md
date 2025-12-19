# ISSUE: ClearML「MCP Agent」連携で、入力Datasetから処理設定を推奨/自動実行する（Auto-Config / Auto-Run）構想

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / ねらい

- 現状は `dataset_id` と YAML（preprocess/training/inference）を人が用意し、Pipeline を実行する。
- 非MLユーザー/初見データでは「どの前処理/モデル/評価/推論モードが妥当か」を決めるのが難しく、試行錯誤コストが高い。
- ClearML の Task / Dataset / Pipeline と、（想定）MCP Agent 機能を組み合わせて、
  - 入力Datasetの内容を観測（スキーマ、欠損、分布、サイズ、目的変数タイプ等）
  - 設定の推奨（YAML生成、候補提示）
  - 条件により自動実行（Auto-Run）
  を実現したい。

## 想定するゴール体験（ユーザー導線）

### ゴールA: “推奨だけ” で安全に迷いを減らす（Human-in-the-loop）

1. ユーザーは `dataset_id` を入力して **Config Advisor** タスクを実行
2. ClearML 上の `Plots/Artifacts` に以下が出る
   - Dataset summary（型推論、欠損、カテゴリ数、外れ値、サイズ）
   - 推奨パイプライン（preprocess/training/inference）の YAML（差分付き）
   - 推奨理由（ルール/根拠、注意点）
3. ユーザーは推奨 YAML を採用して pipeline を実行

### ゴールB: “推奨 + 自動実行” でワンクリック（Auto-Run）

1. ユーザーは `dataset_id` を入力して **Auto Pipeline**（または pipeline の advisor step）を実行
2. Agent が推奨 config を生成し、**その config を Artifact として保存してから** training/inference を実行
3. training-summary に “推奨config” と “採用理由” が残り、再現性が担保される

## To-Be案（“何を重視するか”別）

### 案A: ルールベース推奨（再現性/透明性最優先）

**重視**

- 推奨の根拠が説明可能（なぜその設定かを明示できる）
- 予期せぬ外部依存が少ない（LLM無しでも成立）

**To-Be像**

- Dataset profiler（統計/型/欠損/カテゴリ/目的変数）を実装し、特徴量（meta-features）を抽出
- ルール（if/else）または軽量なスコアリングで設定を生成
  - 例: 高カテゴリ列が多い→onehotを抑制/target encoding検討
  - 例: 行数が小さい→線形/正則化系も候補に入れる、CV fold を下げる
  - 例: 欠損が多い→imputation を必須化
- 推奨 YAML と根拠（rule hits）を artifacts に出す

**決めるべき方針**

- 推奨対象の範囲（P0は preprocessing/training だけ、推論は後回し等）
- “安全側のデフォルト” の定義（探索を広げすぎない）
- 例外/未知データの扱い（推奨不能時のフォールバック）

**必要な開発項目（例）**

- `core/data/profiling.py`（meta-features抽出）
- `core/recommender/rules.py`（ルール群、根拠の構造化）
- `workflow/advisor/processing.py`（ClearML task化、artifacts/plots出力）
- 推奨 YAML の保存場所（artifact: `recommended_config.yaml`、差分: `recommended_config.diff`）

---

### 案B: LLM（MCP Agent）推奨（柔軟性/拡張性優先、ただし安全ゲート必須）

**重視**

- データ/目的に応じた柔軟な提案（説明文・注意点・推奨理由も生成）
- 将来的に “追加したモデル/前処理” にも追従しやすい

**To-Be像**

- MCP Agent が参照できるコンテキストを整備する
  - Dataset profile（統計 + サンプル + スキーマ）※PII対策必須
  - 利用可能な前処理/モデル/評価の “カタログ”（registry情報）
  - 既存のUI規約（HyperParameters最小、Plotsナンバリング等）
- Agent は推奨 YAML（構造化出力）を生成し、**静的検証**に通ったもののみ採用可能
  - 例: YAML schema validation / allowlist（モデル名・パラメータ範囲）
- Auto-Run は default off。明示フラグ or approve がある場合のみ実行。

**決めるべき方針**

- LLM の実行場所（社内モデル/外部API）とデータ送信範囲（サンプル送信可否）
- PII/機密に対するマスキング方針（列名/値のハッシュ化、統計のみ送る等）
- “推奨” の責任分界（誤推奨の扱い、免責、ログ保存）
- 生成 YAML の検証（許可する設定の上限/範囲）

**必要な開発項目（例）**

- MCP Agent が読むための “dataset/profile artifact” の標準化（JSON schema）
- registry の “機械可読なカタログ” 化（利用可能なモデル/前処理/指標/制約）
- 推奨 YAML の静的検証器（schema + allowlist + safe defaults）
- ClearML 上の “採用/却下” 操作導線（Human-in-the-loopのUI）

---

### 案C: 2段構成（ルールベースで安全側に絞り、LLMは説明/微調整のみ）

**重視**

- 再現性と安全性を確保しつつ、LLMの価値（文章化・注意点・提案）を活かす

**To-Be像**

- ルールベースで候補を狭める（例: モデル候補3つ、前処理は2パターン）
- LLM は “候補の中から選ぶ/理由を書く/追加注意を出す” のみに限定する
- Auto-Run はルールベース部分のみで実行し、LLMは optional

**決めるべき方針**

- “LLMが変更して良い範囲” をどこまで許可するか（候補選択のみ vs パラメータ微調整まで）

**必要な開発項目（例）**

- ルールベースの “候補生成” と “候補評価” の形式化（LLMに渡す構造を固定）

---

### 案D: 実行自動化重視（Auto-Run / Auto-Retry / Auto-Triage）

**重視**

- “ユーザーは dataset_id を入れるだけ” に近づける
- 失敗時も自動で原因整理・再試行・要約を出す

**To-Be像**

- pipeline の最初に advisor step を入れ、推奨 config を生成
- 失敗（メモリ不足/学習失敗/欠損異常等）を分類して:
  - 設定を縮小（モデル候補を減らす、foldを下げる）
  - 前処理を変更（高次元抑制、カテゴリ処理変更）
  - 再実行（回数上限あり）
- 最終的に training-summary は必ず生成し、`error_models.json` 等で残す

**決めるべき方針**

- 自動再試行の上限、何をもって “同一run” とみなすか
- 失敗の分類（例外→カテゴリ）と対応策の安全性（暴走防止）

**必要な開発項目（例）**

- 失敗分類の taxonomy（例外名/ログパターン）
- 再試行ポリシー（縮小ルール、上限、ログの残し方）

## ClearML UI / 出力設計（推奨の見せ方）

### Advisorタスク（推奨のみ）の推奨UI

- Configuration > HyperParameters
  - `Input`: dataset_id, target_column（必要なら）だけ
  - `Policy`: auto_run=false, recommendation_mode=rules|llm|hybrid
- Artifacts
  - `dataset_profile.json`（統計・スキーマ・サイズ）
  - `recommended_config.yaml`
  - `recommendation_report.md`（理由/注意点/想定リスク）
  - `validation_report.json`（allowlistチェック結果）
- Plots（番号付き）
  - `01_Profile/SchemaSummary`（表）
  - `02_Profile/Missingness`（欠損率Top）
  - `03_Recommendation/ConfigDiff`（テーブル or テキスト）
  - `04_Recommendation/Why`（重要な根拠のみ）

### Auto-Run時の再現性担保（必須）

- “実際に使った config” を必ず artifact として保存（`full_config.yaml` / `effective_config.yaml`）
- User Properties に `recommended_config_hash` / `advisor_task_id` を保存し、run追跡可能にする

## 追加検討1: 非データサイエンティスト向けの理解・解釈支援（MCP/LLM Agent）

### ねらい

- 非データサイエンティストが「何が行われたか / 何が良かったか / 何に注意すべきか」を、
  ClearML UI だけで把握できる状態を作る。
- 重要: LLM の “もっともらしい説明” に引っ張られないよう、**根拠は task の metrics / artifacts / properties に限定**する。

### 想定するゴール体験（ユーザー導線）

1. ユーザーは training-summary / inference タスクを開く
2. `Plots` の上部に **Decision Summary（番号付き）**があり、見るべき場所が固定されている
3. “推奨モデル/推論結果” と “根拠” と “注意点（リスク）” を 1画面で理解できる
4. 必要なら **Explain** タスク（run-explainer）を起動し、Markdown/表で要点が出る

### 判断項目（非専門ユーザーが最終判断するためのチェックリスト案）

- 入力データ: 欠損/外れ値/カテゴリ数/データ量が妥当か（品質リスク）
- 評価: primary metric と 2位との差、CVのばらつき、過学習兆候（性能リスク）
- 適用範囲: 推論条件が学習データ範囲内か（外挿/外れ値リスク）
- 実運用: 推論レイテンシ/モデルサイズ/依存関係（運用リスク）
- 再現性: 使った config / dataset lineage / model_id が残っているか（監査リスク）

### To-Be案（理解・解釈支援）

#### 案I-A: “要約テンプレ” の自動生成（ルール/テンプレ中心、透明性優先）

**重視**

- 説明の再現性・透明性（LLM無しでも成立）

**To-Be像**

- training-summary / inference に、固定テンプレの “Decision Summary” を artifacts と plots に出す。
  - 例: primary metric、上位差分、推奨モデルID、推論条件の範囲内判定、注意点（定型）
- 文章は最小限にし、数値と表を中心にする。

**決めるべき方針**

- “要約で出す最小項目” の固定（増やしすぎない）
- リスク判定のルール（例: 外挿判定は分位範囲か、min/maxか）

**必要な開発項目（例）**

- run共通の `decision_summary.json` / `decision_summary.md` のスキーマ定義
- training-summary / inference の “必須scalars/plots/artifacts” を増やさない形で追加

---

#### 案I-B: LLM による “根拠付き説明” 生成（読みやすさ優先、ガード必須）

**重視**

- 非専門向けの読みやすさ（短い文章で理解できる）

**To-Be像**

- MCP/LLM Agent が、task の artifacts/metrics を読み取って説明文を生成する。
- ただし以下を必須にする:
  - 入力コンテキストは **taskが保存した構造化データのみ**（自由テキストや生データは渡さない）
  - 出力は “要約 + 根拠（参照先artifact名/metric名）+ 注意点” の固定フォーマット
  - 生成物は artifacts に保存し、**人がレビュー可能**にする

**決めるべき方針**

- 生成物の扱い（助言であり正ではない、免責/表記）
- PII/機密の扱い（列名/値のマスク、サンプル送信禁止等）
- “説明できない” ときのフォールバック（テンプレ要約に戻す）

**必要な開発項目（例）**

- `explanation_context.json`（LLMへ渡す最小コンテキスト）を各phaseで生成
- `explanation_report.md` のテンプレ（見出し・固定セクション）
- 生成の監査ログ（prompt hash / model / timestamp / input hash）

---

#### 案I-C: Q&A（質問応答）で “気になった点だけ” 聞ける（対話性優先）

**重視**

- 非専門ユーザーが疑問を “自然文” で解消できる

**To-Be像**

- ユーザーは “このrunで推奨モデルを選んだ理由は？” などを質問する。
- Agent は run_id を鍵に tasks を検索し、該当 artifacts/metrics を根拠に回答する。

**決めるべき方針**

- Q&A の回答範囲（断言禁止、必ず根拠を返す、根拠がない場合は “不明” を返す）
- ユーザー権限（閲覧権限のある task/dataset だけ参照できる）

**必要な開発項目（例）**

- run_id から “参照すべきタスク集合” を確定する仕組み（tags/user properties）
- “根拠リンク” を返す仕組み（task URL / artifact 名 / metric 名）

## 追加検討2: 過去タスク/データセットの検索支援（MCP/LLM Agent）

### ねらい

- 非データサイエンティストが「過去に似たデータ/似た設定/うまくいった run」を、
  ClearML 上で迷わず探せる状態を作る。
- 重要: “検索できる” ためには、**tags / user properties / artifacts の標準化**が前提。

### 想定するゴール体験（ユーザー導線）

- ユーザーは自然文で質問:
  - 「xxデータセットで、rmse が最も良かった run を教えて」
  - 「同じ target_column で過去に使った推奨モデル一覧が欲しい」
  - 「この dataset_id と似た分布の dataset を探して」
- Agent は task/dataset を絞り込み、結果を “表（TopK）+リンク” で返す。

### To-Be案（検索支援）

#### 案S-A: まずは “メタデータの整備” で検索性を上げる（LLM無し）

**重視**

- 低コストで効果を出す（LLM導入前に土台を作る）

**To-Be像**

- タスク/データセットに必須の user properties を揃える:
  - 例: `run_id`, `dataset_id`, `dataset_key`, `phase`, `target_column`, `primary_metric`, `recommended_model_id`
- tags を標準化し、ClearML のフィルタで “必ず探せる” 状態にする。

**決めるべき方針**

- “必須メタデータ” の最小セット（増やしすぎない）
- 値の正規化（dataset_key の命名、target_column の表記揺れ）

**必要な開発項目（例）**

- `set_min_user_properties()` を phase 横断で強制（未設定なら警告/失敗）
- README_user に “探し方（検索キー）” を固定して記載

---

#### 案S-B: LLM による “検索クエリ翻訳” （自然文→ClearMLフィルタ）

**重視**

- 非専門ユーザーが “検索の書き方” を覚えなくても良い

**To-Be像**

- ユーザーの自然文を、Agent が
  - dataset_id / run_id / tag / metric threshold などのフィルタに変換して ClearML API を叩く。
- 返す結果は “表 + リンク” の固定フォーマット。

**決めるべき方針**

- どのフィルタまで許可するか（検索範囲が広すぎると重い）
- 監査ログ（誰が何を検索したか）を残すか

**必要な開発項目（例）**

- ClearML API ラッパ（Task/Dataset/Model の検索、ページング、並び替え）
- “検索用の構造化クエリ” のスキーマ（LLM出力はJSONで固定）
- 結果テーブルの artifacts 生成（`search_results.csv` など）

---

#### 案S-C: セマンティック検索（Embedding）で “似たrun/似たdataset” を探す

**重視**

- “似ている” を自然に探せる（キーワードが分からなくても探せる）

**To-Be像**

- run/dataset の profile（統計/スキーマ/要約）を embedding 化し、検索できるようにする。
- 例: “似た欠損パターン/列型/データ量” の dataset を推薦する。

**決めるべき方針**

- embedding の対象（列名を含めるか、統計だけにするか）
- index の保存場所（外部DB/ファイル/別サービス）と更新頻度

**必要な開発項目（例）**

- index 生成ジョブ（新しい run/dataset を取り込む）
- 参照権限の反映（アクセスできないものは返さない）

## 決めるべき方針事項（議論ポイント）

### 1) 推奨の責任範囲

- どこまで自動決定するか（前処理のみ/モデル候補のみ/全設定）
- 誤推奨時の扱い（ログ/免責/ユーザー確認フロー）

### 2) 安全性（ガードレール）

- 許可するモデル/パラメータの allowlist
- データ量上限（サンプリング、メモリ制限）
- Auto-Run の default（off推奨）と明示的トグル

### 3) データ取り扱い（PII/機密）

- LLM に渡す情報の範囲（統計のみ/列名マスク/値サンプル禁止 等）
- ClearML Artifacts に保存する “profile” の粒度（個票を残さない）

### 4) 追加機能時の保守性

- registry（モデル/前処理/指標）追加が推奨ロジックにどう反映されるべきか
- “推奨ロジックのテスト” をどう設計するか（dataset profile -> expected config）

## 必要技術（候補）

- Dataset profiling: pandas + numpy（最小）、必要なら ydata-profiling / sweetviz（重いので要検討）
- ルール/推奨: 自前ルール、または meta-learning（将来）
- LLM/MCP:
  - MCP Agent ランタイム（どの実装/どの権限で動くか要確認）
  - LLM（社内/外部） + プロンプト/構造化出力（JSON schema）
- 検証:
  - YAML schema validation（pydantic / jsonschema）
  - allowlist/denylist（モデル名・パラメータ範囲）
- ClearML:
  - Dataset / Task / PipelineController（advisor step の配置）
  - Artifacts/Plots/User Properties の規約化
- 検索/理解支援（追加検討）:
  - ClearML API（task/dataset/model 検索、metrics/artifacts 取得）
  - ランキング/集計（TopK、しきい値フィルタ）
  - （任意）Embedding/ベクトル検索（FAISS等、外部DB含む）

## 実装コストの目安（概算）

> 前提: 既存の phase 構造を崩さず “advisor” を追加する想定。規模感は要件確定後に再見積もり。

- P0（最小・ルールベース推奨、Human-in-the-loop）: **3〜7人日**
  - dataset profile（軽量）+ 推奨YAML生成 + artifacts/plots 出力 + 簡易バリデーション
- P1（hybrid/LLM導入、PII対策、静的検証強化）: **10〜25人日**
  - MCP/LLM連携 + allowlist検証 + 文章レポート + 運用設計（権限/監査）
- P2（Auto-Run + 自動再試行/原因分類 + 受け入れテスト整備）: **20〜40人日**
  - 失敗taxonomy + retry policy + verify scripts + 安全ゲート + 運用フロー整備
- P3（理解・解釈支援: 要約テンプレ + レポート生成）: **5〜15人日**
  - decision summary スキーマ + artifacts/plots + ルールベースの注意点抽出
- P4（検索支援: 自然文→フィルタ + 結果表）: **7〜20人日**
  - 検索APIラッパ + クエリスキーマ + 監査ログ（必要なら）
- P5（セマンティック検索/類似度推薦）: **15〜35人日**
  - index生成/更新 + 権限制御 + embedding運用（モデル/コスト）

## 判断の単位（受け入れ条件案）

- [ ] dataset_id から “推奨 config” が必ず生成され、ClearML 上で理由/注意点が読める
- [ ] 推奨 config は静的検証に通る（危険な設定が混入しない）
- [ ] Auto-Run は default off（または明示フラグ必須）で、実行した場合は再現性が担保される
- [ ] 個票データが artifacts/LLM へ漏れない（方針に沿ったマスキング/サンプリング）
- [ ] 推奨ロジックの回帰がテストで検知できる（最低限: profile→expected のスナップショット）
- [ ] 非専門ユーザーが “要約を見るだけ” で、推奨/注意点/次アクションを理解できる
- [ ] 自然文検索で過去run/datasetを見つけられ、結果がリンク付きで返る（権限内）

## 次アクション（将来タスク案）

1. “推奨対象範囲” を決める（preprocessのみ/モデル候補まで/全設定）
2. dataset profile の標準スキーマ（JSON）を固定（PII方針も同時に決定）
3. ルールベース P0 を先に作り、運用/UX を固める
4. MCP/LLM を入れる場合は、ガードレール（allowlist/validation/監査）を先に設計してから実装

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/README_user.md`（ある場合）
- `docs/REFORM_PLAN.md`（または `Agents.md` に同内容あり）
