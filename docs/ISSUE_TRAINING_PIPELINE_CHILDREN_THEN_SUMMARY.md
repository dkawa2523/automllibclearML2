# ISSUE: Pipelineで「学習子タスク（各モデル）」を直列/並列に実行し、Training-summaryを“後段の評価・比較タスク”にする案

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / 現状（as-is）

- 現在の ClearML Pipeline は概ね `preprocessing -> training -> (optional) inference` の 1 step 構成で、`training` step の実体が **Training-summary（親）**になり、その中で **各モデルの学習子タスク**を生成・実行して結果を集計しています。
- そのため Pipeline 上は「training が 1 step」に見え、モデルごとの進捗/失敗は Training-summary 側の子タスクを見る必要があります。

関連: `docs/DEV_GUIDE.md`（現状の Pipeline / training-summary / 学習子タスク構造の説明）

## 検討したい案（to-be）

### 案B（提案）

- Pipeline が `preprocessing` の後に **モデル数ぶんの学習子タスク**（例: `train_model::<model_name>`）を step として生成し、直列/並列で実行する
- 全学習子タスクが完了した後に、**Training-summary 相当のタスク**を「評価・比較（集計）専用」として実行する
  - Training-summary タスクは **学習はしない**
  - 既存の Training-summary と同じ `Plots` / `Artifacts`（leaderboard / recommended model 等）を出す

### 目的（何を改善したいか）

- Pipeline の見通しを上げる（どのモデルが今動いている/落ちたかが Controller の DAG で分かる）
- “1プロセスにつき main task は 1つ” をより素直に守る（学習子タスクを step task = main task として扱う）
- Training-summary を「集計」へ責務限定し、拡張ポイントを分離しやすくする（学習ロジックと集計ロジックの分離）

## 期待できる効果（メリット）

### 1) ユーザー体験（非ML）

- Pipeline の DAG から「学習がどこまで進んだか」「どれが失敗したか」を把握しやすい
- Training-summary は最終ステップとして固定され、ユーザーが見るべきタスクがより明確になる

### 2) 運用（ClearML / Agent）

- モデルごとに step を分けることで、queue/agent の割り当て・スケールが分かりやすい（重いモデルだけ別キュー等の設計が可能）
- 学習子タスクが “独立した Task” として実行されるため、再実行・再利用の粒度が揃う（1モデルだけ rerun など）

### 3) 保守性（開発者）

- 「学習子タスクの契約（Artifacts/Scalars）」と「集計タスクの契約（leaderboard 等）」を明確に分けられる
- Training-summary が子タスク生成を担わないため、Task 生成/リンク/例外処理が単純化する可能性がある

## 想定デメリット / リスク（デメリットも含めて）

### 1) 失敗時の扱いが難しくなる可能性

- Pipeline の step は通常「親が失敗したら子が動かない」依存関係になるため、
  - 1モデル失敗 → Training-summary（集計）が実行されず、結果が見えない
  - という状況が起きうる（現状は Training-summary が “部分成功” を許容して集計を継続できる設計が可能）
- 対策案はあるが要検討（例: 失敗モデルは成功扱いで終了させ、エラー情報を artifact に保存する、など）

### 2) step数が増えて UI/運用が重くなる

- モデル候補が多い/ハイパラが多い場合、Pipeline の step が増えすぎて見づらくなる
- ClearML サーバ側のタスク数・ログ量・メトリクス量も増加する（課金/ストレージ/検索性の影響）

### 3) 集計タスクが ClearML API 依存を強める

- Training-summary（集計専用）は、各学習タスクの成果物を集めるために
  - task-id の受け渡し（Pipeline outputs で渡すか、run_id tag 検索するか）
  - artifacts のダウンロード/metrics の取得
  - など ClearML API 依存が増える可能性がある

### 4) “HyperParameters汚染” が別形で増える可能性

- PipelineController の step task は内部メタ（`kwargs` 等）が HyperParameters に出る仕様があり、学習子タスクが step 化するとその影響範囲が広がる
  - ただし本プロジェクト方針（見せたい section を別途安定して出す）で吸収できる可能性は高い

## 決めるべき方針（議論ポイント）

### A. 集計対象の特定方法（“どの学習タスクを集計するか”）

- A1: Pipeline から **task_id のリスト**を Training-summary step に渡す（最も混線しにくい）
- A2: `run:<run_id>` / `phase:training` など **tags 検索**で集める（実装は楽だが誤混入リスク）
- A3: “モデル登録（Model Registry）の一覧” を run_id で引く（Registry を正とする）

### B. 失敗時の集計方針（部分成功を許容するか）

- B1: 1つでも失敗したら run 失敗（厳格、運用は単純）
- B2: 失敗モデルは `error_models.json` 等に落として **集計は継続**（ユーザー価値は高いが実装・設計が必要）

### C. Pipeline step としての学習子タスク実装方式

- C1: `add_function_step` で `run_train_model_processing(model_name=...)` を直接呼ぶ（テンプレ不要）
- C2: `add_step(base_task_id=...)` で “学習子タスクテンプレ” を clone して `model_name` を上書き（運用は ClearML 的に自然）

### D. 表示上の“格納場所”の規約

- D1: 現状通り project suffix（例: `.../train_models`）で分ける
- D2: inference optimize と同様に `Training_runs/<model_name>` のような階層に揃える（UI統一）

## 判断の単位（検証観点 / 受け入れ条件案）

- [ ] Pipeline UI だけで “何が失敗したか” が一次判断できる（非MLが迷わない）
- [ ] Training-summary が **必ず**生成され、`Plots`/`Artifacts` が現状と同等（leaderboard / recommended model など）
- [ ] 失敗モデルがあっても、方針に応じた挙動（B1/B2）が再現できる
- [ ] 学習子タスクの再実行が容易で、run_id/tag 追跡が崩れない
- [ ] step数増加時の UI / サーバ負荷が許容範囲（上限目安を決める）

## 実装のたたき台（将来タスク分解案）

1. `run_train_model_processing`（1モデル学習）を **単独タスクで成立**する形で切り出す
2. PipelineController で `training` step を “モデル数ぶんの step + 集計 step” へ置換
3. 集計 step が読むための “学習子タスク成果物の最小契約（artifact/metric）” を固定
4. 失敗時方針（B1/B2）に応じた挙動・ログ（`error_models.json` 等）を定義
5. 既存 `scripts/verify_clearml_training_summary_plots.py` 等に検証を追加

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/REFORM_PLAN.md`（または `Agents.md` に同内容あり）

