# ISSUE: ClearML上で“タスク内容を把握しやすく / 迷子にならない / 増えても見つけやすい” 情報設計・階層設計（To-Be options）

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / 課題（as-is）

- 現状でも `preprocessing` / `training-summary`（+学習子タスク）/ `inference`（モードによって親子）など複数タスクが生成される。
- タスクが増えるほど、ClearML UI 上で以下が起きやすい:
  - どのタスクを見れば良いか分からない（ユーザー導線が散る）
  - 同じrunのタスクを横断で見つけにくい（検索・フィルタの手掛かり不足）
  - 親子タスクの関係が辿れない / “どれが要約でどれが詳細か” が曖昧
  - タスクが増えた時に project/tag/命名が統一されていないと、後から整理不能
- 将来的に pipeline のステップやタスクが増えることが前提であり、**情報設計（IA）を先に決めないと運用負債になりやすい**。

## 目的（このISSUEで議論したいこと）

### ユーザー視点（非ML）

- `run_id` を軸に、ClearML 上で **「見るべきタスク」**が一意に分かる
- タスクの Configuration / Plots / Artifacts の “見る場所” が固定され、判断に迷わない
- タスクが増えても、検索・一覧で **すぐ見つかる**（フィルタが効く）

### 開発者視点

- “タスクが増える/階層が深くなる” ほど、命名規約・タグ規約・リンク規約が重要になる
- 追加・変更時に **触る場所が一意**で、設計の一貫性が崩れない
- “後から編集/整理” できる（既存runの検索性が落ちない）

## まず守る原則（前提）

- **親は要約、子は詳細**（PLOTS 爆発を防ぐ）
- **HyperParameters は入力設定の最小**、参照用情報は Artifacts / User Properties に寄せる
- `run_id` / tags / 命名はすべてのタスクで統一し、run横断で追跡可能にする
- タスク再利用（task reuse）による混線を避ける（必要なら `reuse_last_task_id=False` を徹底）

## To-Be案（“何を重視するか”別の構成案）

### 案A: 検索性最優先（tags / user properties / 命名を強制し、UIの探し方を固定）

**重視**

- 既存構造を崩さず、短期で効果を出す
- ClearML の検索・フィルタだけで迷子を減らす

**To-Be像**

- 全タスク共通:
  - tags: `run:<run_id>`, `phase:<phase>`, `dataset:<dataset_key>`（+必要なら `model:<name>` / `mode:<inference_mode>`）
  - User Properties: `run_id`, `dataset_id`, `dataset_key`, `phase`, `parent_task_id` など “検索キー” を必須化
  - task name: `"<phase> ... ds:<dataset_key> run:<run_id>"` を必須化（regex安全なトークンで）
- ユーザーガイドで「ClearMLでの見つけ方」を固定する（例: `tags run:<run_id>` で検索 → phase別に見る）。

**決めるべき方針**

- tags のキー一覧（最小セット）と、増やして良い拡張タグの境界
- user properties に入れる “必須キー” と “長文禁止” のルール

**必要な開発項目（例）**

- tags/user properties を強制する共通ヘルパ（漏れたら警告）
- “run_id の入口” を統一し、全phaseが必ず付与する仕組みを強化

**メリット**

- UI構成（project階層）を変えなくても、検索だけで運用が回る

**デメリット**

- “タスク一覧が膨大” なケースでは、検索しても目的タスクを特定しづらい（一覧の見せ方は改善しない）

---

### 案B: 階層の見やすさ最優先（Project構造 + ディレクトリ風命名で整列）

**重視**

- タスク数が増えたときに “一覧” で探しやすい
- どこに何があるか（親/子、要約/詳細）が視覚的に分かる

**To-Be像**

- Project の階層（例）:
  - `AutoML/<dataset_key>/run:<run_id>/Summaries`（要約タスクだけ）
  - `AutoML/<dataset_key>/run:<run_id>/Training_runs`（学習子タスク）
  - `AutoML/<dataset_key>/run:<run_id>/Prediction_runs`（推論子タスク）
  - `AutoML/<dataset_key>/run:<run_id>/Data`（dataset/preprocess関連）
- タスク名も “ディレクトリ風 prefix” を持つ:
  - `Training_runs/<model_name> ...`
  - `Prediction_runs/trial:<k> ...`
  - `Summaries/training-summary ...`
- “ユーザーが見るのは Summaries だけ” を徹底し、他は詳細に隔離する。

**決めるべき方針**

- Project 階層をどこまで細かくするか（深すぎると逆に探しにくい）
- 既存の project_mode / suffix（例: `train_models_suffix`）との整合
- 学習/推論の子タスクの置き場所を統一するか（`Training_runs` / `Prediction_runs` を固定するか）

**必要な開発項目（例）**

- `build_project_path()` / naming を “階層版” に拡張（互換性を保ちつつ移行）
- 既存タスクの move_to_project / rename の統一
- docs（ユーザー/開発者）に “見る場所ルール” と “階層ルール” を追記

**メリット**

- タスクが増えても “置き場所” が固定され、一覧で見つけやすい

**デメリット**

- project分割は運用コスト（権限/検索範囲/テンプレ管理）が増える場合がある

---

### 案C: ナビゲーション最優先（Run Index / Links / Dashboard を“入口”にする）

**重視**

- 非MLが “どこから入るか” を固定し、リンクを辿れば迷わない
- タスクが増えても入口（Index）が変わらない

**To-Be像**

- runごとに **入口タスク**（例: `run-summary` または pipeline controller）を必ず1つ作り、
  - `Overview` の Plots に “Run Index”（リンク集/表）を置く
  - Artifacts に `run_index.json` / `run_index.md` を置く
  - User Properties に “重要ID”（dataset_id, recommended_model_id 等）を置く
- 各タスクは、親/関連タスクへのリンク（URL, task_id）を artifact/user properties に保存する。
  - 例: preprocessing -> training-summary -> inference への導線を固定

**決めるべき方針**

- 入口タスクを何にするか:
  - pipeline controller task を入口にするか
  - training-summary を入口にするか
  - run-summary を新設するか（＝新しいタスク種）
- Link の表現:
  - ClearML の親子関係（parent task）だけに頼るか
  - 明示的に “リンク集artifact” を正とするか

**必要な開発項目（例）**

- `integrations/clearml/links.py` 相当の共通機構（関連タスクの登録・参照）
- run単位の index 生成（最小: JSON/CSV、将来: HTMLレポート）
- scripts で “indexからすべてのタスクが辿れる” を検証する仕組み

**メリット**

- タスクが増えてもユーザーは “入口→リンク” で迷わない

**デメリット**

- Index を作るタスクが失敗すると “入口が壊れる” リスクがある（堅牢化が必要）

---

### 案D: 表示の一貫性最優先（タスクテンプレ/レイアウトを固定し、UI差分を減らす）

**重視**

- “どのタスクを開いても同じ見た目/同じ場所に同じ情報” を徹底し、理解コストを下げる

**To-Be像**

- Phaseごとに “UIスキーマ” を固定:
  - Configuration: 必須セクション名（Input/Training/Inference…）
  - Plots: タイトルprefix（`01_...`）の必須セット（要約は最小）
  - Artifacts: 必須ファイル名（`leaderboard.csv`, `trials.csv` 等）
  - User Properties: 必須キー
- 追加可視化は子タスク側のみ、親タスクは固定セットから増やさない。

**決めるべき方針**

- 各phaseの “必須” をどこまで固定するか（柔軟性とのトレードオフ）
- 新機能追加時の “UI変更の審査基準” をどう決めるか（CIで検証するか）

**必要な開発項目（例）**

- verify scripts（既にある `verify_clearml_*` を拡張）で UI規約を機械検証
- UIスキーマ（必須plots/artifacts/properties）の定義ファイル化

**メリット**

- タスク数が増えても “読む場所が同じ” なのでユーザーが迷いにくい

**デメリット**

- 柔軟な実験・探索のための自由度が下がる（子タスクに閉じ込める設計が前提）

## 共通で決めるべき方針事項（議論ポイント）

### 1) 入口の定義（ユーザーが最初に開くタスク）

- pipeline controller / training-summary / run-summary（新設）のどれを “入口” にするか
- 入口から “推論に使う model_id をコピーできる” までの最短導線をどう保証するか

### 2) タスクの分類（要約 vs 詳細）

- “要約タスク” の集合を固定するか（例: preprocessing-summary, training-summary, inference-summary）
- “詳細タスク” の置き場所（Training_runs/Prediction_runs）を統一するか

### 3) 発見性（Search / Filter / 一覧）

- tags の最小セットと、一覧表示（列）に出す user properties の最小セット
- runが大量にあるときの “探し方” をどこまでガイドするか（README_user に書くか）

### 4) 編集容易性（後から変える時の負債を減らす）

- 命名/タグ/プロジェクト階層の変更が、過去runの検索性に与える影響（移行戦略）
- “タスクが増えた時に追加するべき最小情報” をテンプレ化する（チェックリスト運用）

## 判断の単位（受け入れ条件案）

- [ ] run_id を1つ指定すれば、関連タスクを漏れなく見つけられる（検索/リンクのどちらでも）
- [ ] 非MLが “見るべきタスク” を 1〜2個に絞れる（それ以外は詳細として隔離）
- [ ] Plots は番号付きで並び、要約タスクは固定セットのみ（増えない）
- [ ] タスクが増えても、プロジェクト階層 or tags のどちらかで確実に整理できる
- [ ] 開発者が新しいタスクを追加するとき、命名/タグ/リンク/UIのチェック項目が明確（漏れない）

## 次アクション（将来タスク案）

1. “ユーザーが迷うパターン” を具体例で収集（どの画面で詰まるか）
2. 案A〜Dから、短期に採用する優先度を決める（例: 短期A+D、将来B or C）
3. 命名/タグ/プロジェクト階層/リンクの **最小規約**をドキュメント化（チェックリスト化）
4. verify script を追加し、UI規約（必須plots/properties/セクション名）を機械検証できるようにする

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/README_user.md`（存在する場合）
- `automl_lib/integrations/clearml/naming.py`
- `automl_lib/integrations/clearml/properties.py`
- `automl_lib/integrations/clearml/links.py`（存在/追加候補）

