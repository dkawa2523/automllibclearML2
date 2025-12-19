# ISSUE: Copilot / Codex で共同開発しやすくする Prompt Playbook（開発指示テンプレ集）整備

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / 課題（as-is）

- 共同開発で Copilot/Codex 等のコーディング支援を使う場合、指示の書き方・守るべき規約が揃っていないと以下が起きやすい:
  - アーキテクチャ境界（workflow / integrations / core）が崩れて保守が難しくなる
  - ClearML UI の規約（HyperParameters最小、親=要約/子=詳細、Plotsナンバリング）が崩れてユーザーが迷う
  - run_id / tags / 命名の統一が崩れ、後から検索・比較・監査ができなくなる
  - “動くけど契約が壊れる” 変更が入り、再現性や推論互換が壊れる
- 一方で、ワークフロー自体は踏襲しつつも、将来的に処理の抜本的変更/拡張（タスク増、UI増、I/O契約変更）が起きる可能性がある。
- そのため「AIに何を守らせ、何を変更して良いか」を明文化した **Prompt Playbook（ベースプロンプト + 作業別テンプレ）**を用意し、誰でも同じ品質で指示できる状態を作りたい。

## 目的（このISSUEで議論したいこと）

- Copilot/Codex に渡す **ベース指示**と **作業別プロンプト**を整備し、開発の再現性を上げる
- “守るべき規約” と “判断すべき論点” を明確にし、AI利用でも保守性を落とさない
- 新規開発者でも「どのファイルを開き、何を確認し、どう検証するか」を迷わず進められるようにする

## 前提（守りたい原則）

- 既存ワークフロー（dataset_id → preprocess → training-summary → inference）を基本踏襲
- ClearML UI ルール（HyperParameters最小、User Propertiesは要約、Artifactsは再現、Plotsは意思決定）を維持
- `run_id` / tags / 命名規約を維持し、run横断の検索性を落とさない
- “親は要約、子は詳細” を維持して PLOTS 爆発を防ぐ

## To-Be案（“何を重視するか”別）

### 案A: docs に「プロンプト集」を置くだけ（最小コスト / すぐ効果）

**重視**

- すぐ使えること（copy & paste）
- 運用負債を増やさない

**To-Be像**

- `docs/PROMPTS.md` または `docs/prompts/` に以下を整備:
  - ベースプロンプト（このリポジトリで必ず守るルール）
  - 作業別テンプレ（モデル追加/可視化追加/推論モード追加/バグ修正/リファクタ/仕様追加）
  - “提出物” の定義（変更点サマリ、影響範囲、残件）

**決めるべき方針**

- 日本語/英語のどちらで統一するか（混在させるか）
- “最小テンプレ数” と “増やして良い範囲” の境界

**必要な開発項目（例）**

- prompts の目次化（用途→テンプレへのリンク）
- `docs/DEV_GUIDE.md` への導線追加（どのpromptを使うか）

---

### 案B: Copilot/Codex に “repo専用ルール” を直接読ませる（標準化 / 指示漏れ防止）

**重視**

- 開発者ごとの指示漏れを減らす（常に同じガードレール）

**To-Be像（例）**

- リポジトリに AI 向け instructions を配置する（ツール対応に合わせる）:
  - `.github/copilot-instructions.md`（GitHub Copilot）
  - `AGENTS.md`（既存の拡張・強化）
  - （必要なら）`docs/AI_DEV_GUIDE.md`
- 中身は「守るべき規約」「触る場所」「テスト/検証」「禁止事項」を短く固定化する

**決めるべき方針**

- “AIに必ず守らせたいルール” の最小セット（多すぎると読まれない）
- どのツール（Copilot/Codex/ChatGPT等）を主要対象にするか

**必要な開発項目（例）**

- instructions のメンテナ担当（更新ルール）
- 重要規約のチェックリスト化（PRテンプレに入れる等）

---

### 案C: “コンテキストパック” 自動生成（AIに渡す情報をブレさせない）

**重視**

- 参照不足による誤実装を減らす（AIが関係ファイルを見落とさない）

**To-Be像**

- `scripts/build_ai_context.py` 等で、次を 1コマンドで生成:
  - 対象phaseの関連ファイル一覧（自動抽出）
  - 参照すべき docs（DEV_GUIDE / UI規約 / 契約）
  - 既存 verify scripts / tests の実行手順
- 開発者は生成された “貼り付け用テキスト” を Copilot/Codex に渡して作業開始する

**決めるべき方針**

- “どこまでを自動収集するか”（ファイル全文 vs 要点のみ）
- 機密/PII を含む可能性がある artifact/ログは含めない（ルール化）

**必要な開発項目（例）**

- phase別のコンテキスト定義（preprocess/training/inference）
- 生成物のフォーマット（Markdown, JSON）

---

### 案D: “仕様→実装→検証” を強制する開発テンプレ（品質優先）

**重視**

- AI による “勝手な実装” を防ぐ（合意した仕様だけ実装）
- 回帰を減らす（受け入れ条件・検証が先に固定される）

**To-Be像**

- Issue/PR テンプレに以下を必須化:
  - 仕様（ユーザー導線 / UI設計 / I/O契約）
  - 受け入れ条件（ClearMLで確認する場所、必須Artifacts/Plots/Properties）
  - 影響範囲（触る予定のファイル、互換性）
- AI には “このテンプレを埋めてから実装に入る” 指示を与える

**決めるべき方針**

- “受け入れ条件の標準セット” を phase別に固定するか

**必要な開発項目（例）**

- phase別チェックリスト（preprocess/training-summary/inference）
- verify scripts の拡充（UI規約の機械検証）

## Prompt Playbook に入れたい内容（案）

### 0) AIに渡す入力テンプレ（毎回埋める）

> 目的: “指示の抜け” を減らし、AIの推測実装を防ぐためのテンプレ。Copilot/Codex の最初のメッセージに貼る前提。

```text
【目的 / 変更内容】
- やりたいこと:
- 背景（なぜ必要か）:
- 期待するユーザー体験（ClearML上でどこを見るか）:

【スコープ】
- In scope（やる）:
- Out of scope（やらない）:
- 互換性（壊して良い/悪い）:

【ClearML UI 要件（必須）】
- HyperParameters: 追加/変更する section とキー一覧（“入力設定” のみ）
- User Properties: 追加/変更するキー一覧（短い要約のみ）
- Artifacts: 追加/変更するファイル名（再現に必要なもの）
- Plots: 追加/変更するタイトル一覧（必ず `01_...` のように番号付け）
- 親/子タスクの責務（親=要約、子=詳細）:

【I/O契約（必要なら）】
- dataset contract の変更有無（schema/manifest/bundle 等）:
- training outputs（leaderboard/model_id 等）の変更有無:
- inference outputs（mode別 artifacts）の変更有無:

【編集してよい場所 / 触ってはいけない場所】
- 触るファイル候補:
- 触らない/変更禁止のファイル:

【検証（最低限）】
- 実行したいコマンド:
  - `.venv\Scripts\python -m unittest discover -s tests -q`
  - （ClearMLサーバがある場合）`.venv\Scripts\python scripts\verify_clearml_training_summary_plots.py --config config.yaml`
  - （ClearMLサーバがある場合）`.venv\Scripts\python scripts\verify_clearml_inference_pipeline.py --config config.yaml --inference-config inference_config.yaml`

【提出物（作業完了条件）】
- 変更ファイル一覧:
- 仕様上の差分（UI/Artifacts/Plots/Properties）:
- 残件（TODO/リスク/改良点）:
```

### 1) ベースプロンプト（全作業共通）

**目的**: どの作業でも守るべき規約を毎回同じ形で渡す。

**含めたい要点（例）**

- このリポジトリの目的（非MLユーザーが ClearML 上で迷わず判断できる）
- 触る場所の原則（workflow / integrations / core の責務、registry拡張ポイント）
- ClearML UI ルール（HyperParameters最小、Plotsナンバリング、親=要約/子=詳細）
- run_id / tags / 命名の必須化
- “推測で書かない” ルール（必ず該当ファイルを検索/参照してから変更）
- 変更後にやること（最小テスト、verify script、残件の明記）

**ベースプロンプト例（叩き台）**

```text
あなたは automl_lib_clearML の開発者です。
必ず docs/DEV_GUIDE.md とリポジトリの規約（run_id/tags/命名、ClearML UIルール）に従ってください。
目的は「非MLユーザーが ClearML 上で迷わず、推奨モデル/推論結果を理解できる」ことです。

禁止:
- 推測で実装しない（必ず rg で参照箇所を特定し、関係ファイルを読んでから変更）
- 親タスクに重いPlotsを追加しない（親=要約、子=詳細）
- HyperParameters に関係ない設定を大量に出さない（最小の入力だけ）

要件:
- 変更点は最小・局所にする
- 影響範囲/互換性/残件を最後に箇条書きで出す
```

**ベースプロンプト例（詳細版・叩き台）**

```text
あなたは automl_lib_clearML の開発者です。以下の制約を絶対に守って変更してください。

【最重要ゴール】
- 非MLユーザーが ClearML 上で「見るべき場所が固定」され、推奨モデル/推論結果を迷わず理解できること。

【アーキテクチャ境界（守る）】
- core: ClearML非依存のロジック
- workflow: phaseの組み立て（I/O、親子タスク、成果物の生成）
- integrations/clearml: ClearMLへのログ出力・命名・tags・properties
- registry: 追加点（モデル/指標/前処理）

【ClearML UI ルール（守る）】
- HyperParameters: “入力設定” のみ、section分割して最小化（大量dump禁止）
- User Properties: 短い要約と検索キーのみ（長文禁止）
- Artifacts: 再現・一覧・表（YAML/CSV/JSON）を保存
- Plots: 親タスクは要約（少数の番号付きPlotsのみ）、詳細は子タスクへ集約

【運用ルール（守る）】
- run_id / tags / 命名の統一を崩さない（検索できなくなる変更は禁止）
- 推測で実装しない（必ず `rg` で参照箇所を特定し、関係ファイルを読んでから変更）
- 変更は最小・局所（不必要なリファクタ禁止）

【実装の進め方】
1) 既存実装の入口/責務/契約を特定（workflow/registry/integrations）
2) 追加/変更する UI（HyperParameters/Artifacts/Plots/Properties）を先に設計し、最小化
3) 実装は契約を壊さない範囲で行い、必要なら tests / verify scripts を更新
4) 最後に “影響範囲/互換性/残件” を箇条書きで報告する
```

### 2) 作業別プロンプト（テンプレ例）

#### 2.1 新しいモデルを追加（training 子タスク）

**狙い**: registry追加 + training子タスクの I/O 契約/metrics/plots を崩さない。

```text
目的: 新しいモデル <MODEL_NAME> を追加する。
制約: registry 追加場所を一意にし、training-summary の UI（Plots 01..固定）を壊さない。

背景:
- 既存ワークフロー（dataset_id → preprocess → training-summary → inference）を踏襲する。
- モデルの追加は “子タスク側の比較用 metrics/scalars を揃える” ことが重要。

やること:
1) 既存の追加点を確認:
   - `automl_lib/registry/models.py`（登録）
   - `automl_lib/workflow/training/model_tasks.py`（子タスク生成/学習）
   - `automl_lib/workflow/training/runner.py`（summary集計）
2) `<MODEL_NAME>` の学習/予測APIを既存モデルに合わせて実装（例外時のエラーハンドリングも）
3) 子タスクが最低限出すべき出力を揃える:
   - 比較用 scalar/metric（例: `metric/<name>`）
   - model registry 登録/または model artifact
   - 失敗時のエラー要約（親が集計できる形）
4) training-summary の leaderboard に載ること（モデル名・主要指標・model_id）を確認
5) config 例（yaml）を追加（既存の `config_training.yaml` の形式に合わせる）
6) テスト/検証:
   - `.venv\Scripts\python -m unittest discover -s tests -q`
   - （ClearMLサーバがある場合）`.venv\Scripts\python scripts\verify_clearml_training_summary_plots.py --config config.yaml`

提出物:
- 変更ファイル一覧
- 追加したモデルの設定例（yaml）
- 互換性/残件
```

#### 2.2 inference の Plots/Artifacts を追加（親子の責務を維持）

```text
目的: inference の可視化/Artifacts を追加する（親は要約、子は詳細）。
制約:
- Plots タイトルは必ず `01_...` の番号付き（表示順を固定）
- HyperParameters は入力設定のみ（推論条件や範囲は “入力として” 見せる）
- Artifacts に再現情報（input/output/config）を残す

参照:
- `automl_lib/workflow/inference/processing.py`（入口）
- `automl_lib/workflow/inference/runner.py`（親子/実行）
- `automl_lib/workflow/inference/visualization.py`（plots）

確認:
- single/batch は親タスクを作らない仕様を維持
- optimize は inference-summary + Prediction_runs/* の子タスク構造を維持
```

#### 2.3 ClearML の HyperParameters 汚染を防ぐ変更

```text
目的: 関係ないHyperParametersが出ないようにする。
制約: Task.init の auto_connect_arg_parser を無効化し、connectする辞書は最小にする。
実装後: ClearML UI 上で増えた/減った section を列挙し、意図通りか確認する。
```

#### 2.4 前処理（preprocess）を追加/変更（dataset contract を壊さない）

```text
目的: 前処理 <CHANGE_DESC> を追加/変更する。
制約:
- preprocess は “再現可能” であること（bundle/recipe/summary が崩れない）
- 変更は preprocessed dataset contract（schema/manifest/preprocessing/*）に反映し、train/infer と整合させる
- ClearML 上で、前処理の要点がカテゴリ別に確認できる（長文1行dump禁止）

参照:
- `automl_lib/workflow/preprocessing/processing.py`（入口）
- `automl_lib/workflow/preprocessing/meta.py`（contract生成）
- `automl_lib/workflow/preprocessing/clearml_integration.py`（ClearML出力）
- `automl_lib/registry/preprocessors.py`（追加点）

やること:
1) 現状の contract（どのファイルを生成しているか）を確認
2) `<CHANGE_DESC>` をどこに入れるべきか決める（registry追加か、既存処理の拡張か）
3) bundle/joblib と recipe/summary の整合を保つ（必要なら contract version を上げる方針を提案）
4) ClearML HyperParameters に “ユーザーが編集する前処理項目だけ” を追加（Inputではなく Preprocessing section）
5) テスト/検証:
   - `.venv\Scripts\python -m unittest discover -s tests -q`
   - （契約にテストがある場合）対象テストを更新

提出物:
- contract 変更の有無（ファイル/キー一覧）
- 互換性（過去datasetで動くか）
- 追加した YAML 設定例
```

#### 2.5 評価指標（metrics）を追加（leaderboard と primary metric の整合を守る）

```text
目的: 新しい評価指標 <METRIC_NAME> を追加し、training-summary の集計に反映する。
制約:
- 指標は “比較可能な scalar 命名規約” に揃える（例: `metric/<name>`）
- training-summary の Plots セット（01..固定）を崩さない（必要なら “子タスク側に詳細”）
- primary metric の扱い（最小/最大、符号）を明示する

参照:
- `automl_lib/registry/metrics.py`（追加点）
- `automl_lib/training/*`（評価計算）
- `automl_lib/workflow/training/runner.py`（summary集計）

やること:
1) 既存 metrics の実装パターンを確認し `<METRIC_NAME>` を追加
2) 子タスクが scalar として必ず出すようにする
3) training-summary の leaderboard に表示されることを確認（列名、並び替え）
4) config（yaml）で有効化できるようにする（evaluation.*）
5) テスト/検証（最小）:
   - `.venv\Scripts\python -m unittest discover -s tests -q`

提出物:
- metric の定義/意味（大きいほど良い or 小さいほど良い）
- config 例（yaml）
```

#### 2.6 training-summary のダッシュボード要件を変更（“親=要約” を崩さない）

```text
目的: training-summary の “要約表示” を <CHANGE_DESC> だけ変更する。
制約:
- training-summary の Plots は “番号付きで少数のみ” を維持（増やす場合は理由と代替案を提示）
- 重い可視化（SHAP/残差/散布図など）は子タスクへ（親に追加しない）
- Artifacts（leaderboard.csv 等）は必ず残す（失敗時も finally で保存されるべき）

参照:
- `automl_lib/workflow/training/summary_plots.py`（ダッシュボード）
- `automl_lib/workflow/training/artifacts.py`（summary artifacts）
- `automl_lib/workflow/training/clearml_summary.py`（失敗時/Properties等）

やること:
1) 現状の plots title（01_..）一覧と、変更後の差分を先に提案
2) 変更は最小差分で実装（表示/集計のみ）
3) verify script がある場合、期待に合わせて更新

検証:
- `.venv\Scripts\python -m unittest discover -s tests -q`
- （ClearMLサーバがある場合）`.venv\Scripts\python scripts\verify_clearml_training_summary_plots.py --config config.yaml`
```

#### 2.7 pipeline の step を追加/順序変更（後から編集しやすい形を壊さない）

```text
目的: pipeline に新しい step <STEP_NAME> を追加（または順序/依存を変更）する。
制約:
- PipelineController の step は `auto_connect_arg_parser=False` を維持
- queue/agent 設定が不足している場合は分かりやすくエラーにする（silent failure禁止）
- step間の受け渡し（result dict のキー）を壊さない

参照:
- `automl_lib/pipeline/controller.py`（DAG構築）
- `automl_lib/workflow/<phase>/processing.py`（stepの入口）

やること:
1) 既存の step 追加パターン（add_function_step）を確認
2) `<STEP_NAME>` の入力（config_path/config_data/input_info/run_id）契約を揃える
3) queue チェック（agents.*）に `<STEP_NAME>` を追加する必要があるか検討
4) pipeline controller task の HyperParameters に “ユーザーが編集する入力” を増やしすぎない

提出物:
- step の I/O（入出力dictの例）
- 追加した config の例（yaml）
```

#### 2.8 YAML / config を拡張（新しい項目を追加しても UI汚染しない）

```text
目的: config に新しい項目 <CFG_KEY> を追加し、該当phaseで利用できるようにする。
制約:
- 追加項目は “入力設定” として HyperParameters に見せるべきか、参照用（artifact）にすべきかを分ける
- 既存 config.yaml の互換性を壊さない（デフォルト値、optional を適切に）

やること:
1) 既存の config モデル/ローダを確認（pydantic/hydra等の方式に合わせる）
2) `<CFG_KEY>` を追加し、該当phaseが参照する経路を通す
3) ClearML へは最小の section のみ connect（full config は artifact に保存）
4) 既存のサンプル YAML（`config.yaml`, `config_training.yaml`, `inference_config*.yaml`）を更新

提出物:
- 追加した設定の意味とデフォルト
- UI（HyperParameters/Artifacts）の出し分け方針
```

#### 2.9 バグ修正（再現→原因特定→最小修正→回帰防止）

```text
目的: バグ <BUG_DESC> を修正する。
制約:
- “直したつもり” を避けるため、再現手順/原因/修正/検証を明文化する
- ついでリファクタはしない（関連箇所のみ）

やること:
1) まず再現手順を作る（CLI / pipeline / unit test のどれで再現できるか）
2) `rg` で該当箇所を特定し、最小の原因を説明できる状態にする
3) 最小修正で直す（副作用がある場合は代替案も提示）
4) 回帰防止としてテスト or verify script の期待値を追加

検証:
- `.venv\Scripts\python -m unittest discover -s tests -q`
```

#### 2.10 未使用コードを削除（Deprecated → Remove の2段階を守る）

```text
目的: 未使用コードを安全に削除する（“主観” ではなく機械的に確定）。
制約:
- いきなり物理削除しない（まず deprecated 化して入口を断つ）
- vulture/coverage 等で参照がないことを確認し、docs/DEPRECATIONS.md に残す

参照:
- `docs/DEPRECATIONS.md`
- `scripts/unused_code_audit.py`

やること:
1) 入口（CLI/pipeline）から切り離す
2) 静的解析/実行トレースで未使用を確認
3) 次リリースで削除、の段階計画を docs に追記
```

#### 2.11 verify script を追加/更新（ClearML UI規約を機械検証する）

```text
目的: 仕様 <SPEC_DESC> を機械検証する verify script を追加/更新する（ClearML UI規約/Artifactsの有無など）。
制約:
- 既存の verify scripts の設計を踏襲し、失敗理由が分かるエラーを出す
- ClearMLサーバがない場合は実行できないので、README/DEV_GUIDE に “前提” を明記する

参照:
- `scripts/verify_clearml_training_summary_plots.py`
- `scripts/verify_clearml_inference_pipeline.py`
- `docs/DEV_GUIDE.md`

やること:
1) 何を “受け入れ条件” として固定するかを箇条書きで定義（plots/artifacts/properties）
2) verify script にチェックを追加（必要なら待機/検索条件を run_id/tag で安定化）
3) 失敗時に “どの条件がNGだったか” を出す（一覧/差分）

提出物:
- 追加したチェック項目一覧
- 実行コマンド例（必要な config も含む）
```

#### 2.12 ドキュメント更新（ユーザー/開発者の導線を壊さない）

```text
目的: 仕様変更 <CHANGE_DESC> を docs に反映し、ユーザー/開発者が迷わないようにする。
制約:
- README_user は “非MLユーザーの一本道” を維持（見る場所固定、余計な詳細は入れない）
- README_dev / DEV_GUIDE は “拡張ポイント” と “規約” を一意に示す（触る場所が分かる）

参照:
- `README_user.md`
- `README_dev.md`
- `docs/DEV_GUIDE.md`

やること:
1) 変更点が “ユーザーの操作” に影響するか整理（config追加、見る場所、出力物）
2) UI（HyperParameters/Plots/Artifacts/Properties）の “見る場所ルール” を更新
3) 開発者向けに “触る場所” と “守る規約” を追記（最小）

提出物:
- 変更した docs 一覧
- ユーザーがやる手順の差分（3ステップで書けるか）
```

#### 2.13 リファクタ（責務分離を保ち、差分を小さくする）

```text
目的: リファクタ <REFactor_DESC> を行い、保守性/拡張性を上げる。
制約:
- 大規模改修になりそうなら “段階案（P0/P1）” を先に提案し、いきなり全置換しない
- core/workflow/integrations の境界を壊さない（ClearML依存が core に入らない）
- 既存挙動（I/O契約、UI規約）を壊さない（必要なら互換ラッパを残す）

やること:
1) 変更したい “痛み” を具体化（どこが重複/複雑/壊れやすいか）
2) 影響範囲を列挙し、段階移行の計画を出す（deprecated 方針が必要なら併記）
3) 最小の差分で第一段階（P0）を実装し、テスト/verify を通す

提出物:
- リファクタの目的/非目的（やらないこと）
- 影響範囲と移行計画（P0/P1）
```

### 3) レビュー用プロンプト（AIに差分レビューさせる）

**目的**: レビュアーの観点を固定し、AIレビューでも “壊れやすい規約” の見落としを減らす。

```text
あなたは automl_lib_clearML のコードレビュアーです。以下の観点で差分レビューしてください。

【ClearML UI 規約】
- 親タスクに重いPlots（SHAP/残差/散布図など）を追加していないか（親=要約、子=詳細）
- Plots のタイトルが `01_...` の番号付きで、順序が安定するか
- HyperParameters に “入力以外” の巨大configが connect されていないか（汚染していないか）
- User Properties に長文を入れていないか（検索キー/要約のみか）
- Artifacts に再現情報（full_config.yaml/leaderboard.csv 等）が残っているか

【run_id / tags / 命名】
- tags が run_id/phase/dataset を含み、検索で辿れるか
- 命名が規約に沿っていて、同名衝突や regex 破壊文字が入っていないか

【I/O契約】
- preprocess dataset contract / training outputs / inference outputs を壊していないか
- 互換性（既存 config/dataset/model）に影響がある変更か

【テスト/検証】
- 影響範囲に対して最小のテスト/verify の更新があるか

出力:
- 指摘（必須/推奨）を分けて箇条書き
- 具体的な改善案（どのファイル/どの関数を直すべきか）
```

## 決めるべき方針事項（議論ポイント）

### 1) どこまでを “プロンプトで固定” し、どこからを “ドキュメント参照” にするか

- プロンプトは短いほど使われるが、短いほど抜けが出る
- “必ず守るルール” はプロンプトへ、詳細は docs 参照へ、の境界を決める

### 2) Prompt の版管理と更新フロー

- 変更したら誰がレビューするか（UI規約に近いので責任者が必要）
- どの粒度で更新するか（プロンプト単体にバージョンを振るか）

### 3) 成果物の評価方法

- “新人でも迷わず追加できる” をどう測るか
- どの verify/tests を最低限必須にするか

## 判断の単位（受け入れ条件案）

- [ ] 新規開発者が Prompt Playbook だけで、モデル追加/可視化追加/推論拡張の指示を出せる
- [ ] AIによる変更でも、ClearML UI 規約（親=要約、Plots番号、HyperParameters最小）が崩れない
- [ ] run_id/tags/命名規約の逸脱が減り、検索性が落ちない
- [ ] 変更後の検証手順（tests/verify）が Playbook に含まれている

## 必要な開発項目（将来タスク案）

1. “プロンプトの置き場所” を決める（docs/ / .github/ / AGENTS.md）
2. ベースプロンプト + 作業別テンプレの最小セットを作る（P0）
3. phase別のチェックリスト（UI/I-O契約）をテンプレに埋め込む
4. verify scripts を Playbook から呼べる形に整理（任意）
5. 運用（更新・レビュー担当）を決める

## 期待効果（まとめ）

- AIを使っても “触るべき場所・守るべき規約” がブレにくくなる
- 非MLユーザー向けの ClearML UI が崩れにくくなる（保守性/運用性の向上）
- 共同開発でのレビュー観点が揃い、回帰を早期に検知しやすくなる

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/DEPRECATIONS.md`
- `docs/ISSUE_DEV_COLLABORATION_WORKFLOW_TOBE.md`
- `docs/ISSUE_PIPELINE_EXTENSIBILITY_AND_STEP_GROWTH.md`
