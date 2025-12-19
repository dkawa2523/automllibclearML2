# ISSUE: ClearMLのタスク/パイプラインを clone して Agent 実行する運用（Remote-first）への拡張案 + 複数Agent管理

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / 課題（as-is）

- 現状のプロジェクトは「ローカルから CLI を叩いて実行」も成立する設計を優先している。
- 一方で実運用では、計算資源（CPU/GPU/メモリ）や依存環境の差分、同時実行、監査の観点から、ClearML Agent による remote execution が有利になりやすい。
- 本プロジェクトは既に “clone 実行” のための部品（`clearml.execution_mode=clone` / `template_task_id` / overrides 等）を持つが、以下は今後の設計/運用として整理が必要:
  - どの単位で template を用意し、何を HyperParameters として編集可能にするか
  - PipelineController を remote で回す場合の controller queue / step queue の設計
  - 複数Agent（複数マシン/複数GPU/複数環境）をどう分け、どう管理するか
  - 失敗時の切り分け（queue問題/依存問題/権限/データアクセス）をどう楽にするか

## 目的（このISSUEで議論したいこと）

- “ローカル実行中心” から “ClearML clone + Agent 実行中心” に移行する場合の To-Be 運用像を複数提示する
- それぞれの案について
  - 決めるべき方針事項
  - 必要な開発項目
  - 期待効果 / デメリット
  を明確にし、チームで議論しやすくする
- 複数Agent/複数queue の設計・管理方法（CPU/GPU/用途別）を検討する

## 前提（守りたい原則）

- 既存ワークフロー（dataset_id → preprocess → training-summary → inference）を基本踏襲
- ClearML UI ルール（HyperParameters最小、User Propertiesは要約、Artifactsは再現、Plotsは意思決定）
- run_id / tags / 命名規約を維持し、検索性を落とさない
- “親は要約、子は詳細” を維持して PLOTS 爆発を防ぐ

## To-Be案（“何を重視するか”別）

### 案A: Phaseテンプレ（clone）中心（最小の運用変更で remote 化）

**重視**

- 既存の CLI/Phase を維持しつつ、実行場所だけを Agent に移す
- 非Pythonユーザーでも ClearML UI から実行できる方向に寄せる

**To-Be像**

- phaseごとに “テンプレタスク” を用意し、ユーザー/運用者はそれを Clone → Queue に enqueue して実行する。
  - template: `preprocessing`, `training-summary`, `inference(single/batch/optimize)`（必要なら `data_registration/data_editing` も）
- “編集可能な項目” は HyperParameters（最小）に限定し、完全configは artifacts/config objects に保存する。
- 実行環境は Agent 側で固定（依存・データアクセス・GPU等）。

**決めるべき方針**

- テンプレを phase 単位で持つか、pipeline 単位で持つか
- テンプレ更新時の運用（誰が、どの頻度で、どう検証して更新するか）
- “ユーザーが編集して良い” HyperParameters の最小セット（汚染防止）

**必要な開発項目（例）**

- テンプレ作成/更新手順の docs（task_id の保管場所、更新フロー、ロールバック）
- clone 実行で上書きする overrides の整理（allowlist化、誤爆防止）
- queue が未指定/agent不在の場合に分かりやすく失敗する（事前チェック/ガイド）

**メリット**

- 最小改修で remote 実行を始められる
- ClearML UI が “入力フォーム” として機能しやすい

**デメリット/リスク**

- phase 単位だと “run の入口” が分散しやすい（どれを先に回すかが人依存になる可能性）
- テンプレ管理が運用負債になりやすい（参照切れ/更新漏れ）

---

### 案B: Pipelineテンプレ（clone）中心（ユーザー導線を “1つの入口” に固定）

**重視**

- 非データサイエンティストが迷わない（入口を1つに固定）
- run_id を軸に “pipeline → summary → inference” が自動で繋がる

**To-Be像**

- “pipeline controller タスク” をテンプレ化し、ユーザーは pipeline を Clone → enqueue するだけ。
- step は `clearml.agents.*` の queue マッピングで実行される。
- 失敗時も training-summary が残る（部分成功/失敗モデル記録の方針が必要）。

**決めるべき方針**

- PipelineController の controller queue（services queue）をどう管理するか
- step queue の設計（phase別/CPU-GPU別/優先度別）
- pipeline の “編集可能項目” をどこまで許すか（dataset_id だけにする、など）

**必要な開発項目（例）**

- pipeline controller template の作成/更新手順
- step増加時の queue/agent チェックの更新手順（規約化）
- 受け入れ確認（verify script）を “remote pipeline” 前提に拡張

**メリット**

- “ユーザーがやること” が最小（dataset_id を入れて実行）
- run 全体の追跡が容易（DAG/親子/Artifacts）

**デメリット/リスク**

- PipelineController の運用（services queue, controllerの安定性）がボトルネックになり得る
- stepが増えた時に DAG が見づらい/重い（設計で抑制が必要）

---

### 案C: ハイブリッド（ローカルは “enqueue専用” / 実処理は常にAgent）

**重視**

- ローカルPCに依存を入れたくない（ユーザー環境を汚さない）
- 実行の再現性/監査性を Agent 側に寄せる

**To-Be像**

- ローカルは “enqueue client” としてのみ利用:
  - dataset_id と最小config を入力して、clone/enqueue して終了
- 実処理はすべて Agent が担当し、ローカルは計算しない（ネットワーク/依存/リソース問題を回避）。

**決めるべき方針**

- ローカルに必要な最小要件（ClearML SDKのみで良いか）
- enqueue の認可/監査（誰が何を実行できるか）

**必要な開発項目（例）**

- enqueue専用CLI（入力を最小化し、誤実行を防ぐ）
- ユーザー/運用者向け docs（“実行はClearMLで見る/止める” を固定）

**メリット**

- 依存・GPU・データアクセスを中央集約できる

**デメリット**

- ClearML/Agent 運用が前提（落ちると何もできない）

## 複数Agent/複数Queue 設計案（管理方法の検討）

### 1) Queue の分け方（例）

- `services`（PipelineController / lightweight）
- `cpu-preprocess`（前処理専用）
- `cpu-train`（軽量モデル/CPU）
- `gpu-train`（NN/SHAP等重い処理）
- `cpu-infer`（推論）
- `gpu-infer`（GPU推論/最適化）

### 2) Agent の配置（例）

- CPUノード群: `cpu-preprocess`, `cpu-train`, `cpu-infer`
- GPUノード群: `gpu-train`, `gpu-infer`
- Controller専用: `services`（リソース小、安定性重視）

### 3) “どのqueueに投げるか” を誰が決めるか

- 運用固定（ユーザーは触れない）: 最も事故が少ない
- phase別固定（ユーザーは phase だけ選ぶ）: 柔軟性と安全のバランス
- 完全自由（ユーザーが queue を指定）: 柔軟だが誤実行が増えやすい

### 4) Agent環境の固定方法（論点）

- Dockerモード（推奨）:
  - base image を固定し、依存を揃える
  - GPUノードは CUDA 対応 image を揃える
- requirements 固定:
  - `requirements.txt` / `requirements-optional.txt` の運用（version pin 方針）
  - 依存の差分が結果差分を生みやすい点への対策

### 5) 監視/運用（論点）

- queue ごとの稼働監視（agentがいない/詰まっているを検知）
- 失敗時の一次切り分け（queue/依存/データアクセス/権限）
- template更新の検証（最小データで smoke run を回してから更新）

## 決めるべき方針事項（議論ポイント）

### A. 入口をどこに固定するか

- pipeline template を入口にするか（推奨）
- phase template を入口にするか（柔軟だが迷いやすい）

### B. queue/agent の権限と運用

- ユーザーに queue 選択を許可するか
- services queue を誰が管理するか（SRE/運用担当）

### C. 再現性（実行環境の固定）

- Docker運用を必須にするか
- requirements の pin 方針（厳格/緩い）

### D. 失敗時の扱い（ユーザー体験）

- 1モデル失敗でも training-summary を残す（部分成功）を標準にするか
- 失敗の分類/再試行（将来）を入れるか

## 判断の単位（受け入れ条件案）

- [ ] ユーザーは ClearML 上で “テンプレ→編集→実行” ができる（Python不要）
- [ ] run_id を軸にタスクが追跡でき、要約（training-summary / inference-summary）に迷わず辿れる
- [ ] queue/agent 不備は事前または早期に明示エラーになる（silent failure がない）
- [ ] 実行環境（依存/GPU/データアクセス）の差分で結果がブレない（少なくとも方針が決まっている）
- [ ] 複数agent構成でも、どの処理がどのqueueで動くかが説明できる（docs/設定で固定）

## 必要な開発項目（将来タスク案）

1. テンプレ運用の方針決定（案A/B/Cどれを入口にするか）
2. queue設計（phase別/CPU-GPU別）と、`clearml.agents.*` の標準設定を用意
3. テンプレ作成・更新・検証（smoke run）フローを docs 化
4. enqueue時の事前チェック（queue/agent）を強化（必要なら）
5. 運用監視（queue詰まり/agent不在/失敗分類）を整備（必要なら）

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/ISSUE_NONPYTHON_USER_EXECUTION_OPTIONS.md`
- `docs/ISSUE_PIPELINE_EXTENSIBILITY_AND_STEP_GROWTH.md`
- `scripts/verify_clearml_training_summary_plots.py`
- `scripts/verify_clearml_inference_pipeline.py`

