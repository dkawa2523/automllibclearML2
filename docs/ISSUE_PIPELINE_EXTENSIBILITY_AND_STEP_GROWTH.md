# ISSUE: Pipelineの編集容易性（後から変更しやすい）と、処理タスク増加に耐えるコード設計案（To-Be options）

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / 現状（as-is）

- Pipeline は主に `automl_lib/pipeline/controller.py` が責務を持ち、ClearML PipelineController に対して
  - `data_registration`（任意）
  - `data_editing`（任意）
  - `preprocessing`
  - `training`
  - `inference`（任意）
  を `add_function_step` で追加する形になっている。
- “増えたステップをどう並べ替える/入れ替える/条件付きにするか” の編集単位がコード寄りになりやすく、
  タスクが増えるほど `controller.py` が肥大化しやすい。

## 目的（このISSUEで議論したいこと）

- Pipeline の “項目（ステップ）” を後から編集しやすくする（順序/依存/有効無効/queue/入力/出力の変更）。
- タスク（ステップ）が増えた場合でも、既存設計（workflow層/clearml integration/run_id/命名規約）を壊さず拡張できる形を検討する。
- ユーザー視点（非ML）で「迷わないUI」を維持しつつ、開発者視点で「触る場所が一意」である状態を維持する。

## 前提（守りたい原則）

- **タスクは独立実行できる**（Pipeline で繋ぐために “別物” にならない）
- ClearML UI の原則（HyperParameters最小、User Propertiesは要約、Artifactsは再現、Plotsは意思決定）
- `run_id` / tags / 命名規約がブレない（検索で追える）
- “親は要約、子は詳細” を維持して PLOTS 爆発を防ぐ

## To-Be案（“何を重視するか”別）

### 案A: 現状踏襲（最小改修で、controllerの責務を“組み立て”に限定）

**重視すること**

- 変更コスト最小（既存コードの延長）
- コードの追跡容易性（宣言DSLに寄せすぎない）

**To-Be像**

- `controller.py` は “step定義の列挙” をやめて、`PipelineSteps` のような小さな構造体（StepSpec）配列を返す関数に分割する。
- 各ステップの詳細は `automl_lib/pipeline/steps/<step>.py` に分離し、controller はそれを順に呼ぶだけにする。
  - 例: `build_preprocessing_step(pipe, cfg, ctx, ...) -> (step_name, output_expr)`

**決めるべき方針**

- “stepを増やす単位” をどこまで細かくするか（phase単位のままか、sub-step（例: feature_selection）も独立させるか）
- 依存/順序の表現方法（親stepを直前固定にするか、名前参照にするか）

**必要な開発項目（例）**

- `StepSpec`（name/parents/queue/function/config/input/output）定義
- step builder の配置規約（新規step追加で触る場所を一意化）
- controller 側の “共通処理（queueチェック、config埋め込み、run_id env 等）” をヘルパ化

**メリット**

- 既存構造を壊さず拡張しやすい
- “コードを見れば動きが分かる” を維持できる

**デメリット**

- 順序/依存/条件分岐の編集は依然としてコード修正が必要

---

### 案B: 宣言的Pipeline（YAML/JSONでDAGを定義し、コードは“registry”で実体を解決）

**重視すること**

- “後から編集しやすい” を最大化（順序/依存/有効無効/queue/再実行の粒度を設定ファイルで扱える）
- ステップ数が増えても controller が肥大化しない

**To-Be像**

- Pipeline の DAG を `pipeline.yaml`（または `config.yaml` 内の `pipeline:` セクション）で定義する。
  - 例: `steps: [{name, kind, enabled, parents, queue, kwargs_ref, outputs}]`
- `kind` をキーにして `registry` から実体（callable）を引く:
  - `kind: preprocessing` -> `automl_lib.workflow.preprocessing.processing:run_preprocessing_processing`
- “各stepが connect する HyperParameters の最小辞書” も step側で定義し、UI規約を守る。

**決めるべき方針**

- 設定の責務分割:
  - DAG定義（pipeline）と、phase設定（preprocessing/training/inference）の境界をどう切るか
- step間のデータ受け渡し:
  - PipelineController の `function_return`（dict）で渡すか
  - ClearML Artifact / Dataset / Model Registry を正とするか
- 互換性:
  - 既存 `config.yaml` のどこまでを宣言化するか（段階的移行の必要）

**必要な開発項目（例）**

- `pipeline/spec.py`（PipelineSpec / StepSpec の型定義 + validation）
- `pipeline/registry.py`（kind -> callable の解決、追加方法を明確化）
- `pipeline/builder.py`（spec -> PipelineController への変換）
- “ユーザーが編集すべき項目” と “参照用の完全config” の分離（Artifactsへの保存規約）

**メリット**

- タスクが増えても “追加=specに1行 + registryに1箇所” に寄せられる
- DAG の編集・レビューがしやすい（コードを読まなくても変更意図が見える）

**デメリット/リスク**

- 宣言DSLの設計を誤ると、逆に複雑で保守しにくくなる（抽象化しすぎ問題）
- “動的にstep数が変わる” ケース（モデル数ぶんのstep生成等）の表現が難しい
  - 対策: “テンプレ + 展開” の仕組みが必要（または案Cへ）

---

### 案C: テンプレTask中心（cloneベースでstepを増やし、Pipelineは“テンプレの接続”だけに寄せる）

**重視すること**

- ClearML 流儀（テンプレ→clone→enqueue）に寄せ、運用の安定性を上げる
- “後から編集” を ClearML UI（テンプレ編集）でも可能にする

**To-Be像**

- PipelineController は、function step ではなく `base_task_id` を利用した clone/run（または `Task.create` を利用）へ寄せる。
- 各phase/stepに対応する “テンプレタスク” を用意し、
  - pipeline.yaml は `template_task_id` と “上書きする最小HyperParameters” を持つだけにする。

**決めるべき方針**

- テンプレ管理:
  - テンプレtask_idの保管場所（yaml/secret/環境変数）と更新フロー
- 再現性:
  - テンプレが更新された時に、過去runの再現性をどう担保するか（versioning）

**必要な開発項目（例）**

- `clone` 実行の統一（既存 `automl_lib/integrations/clearml/clone.py` の拡張）
- テンプレtaskの “最小HyperParameters” の抽出・接続規約
- テンプレ運用ドキュメント（誰が/いつ/どう更新するか）

**メリット**

- ClearML UI 上で “テンプレ編集→実行” の運用がしやすい
- 実行環境（agent/requirements）の差分をテンプレ側に寄せやすい

**デメリット/リスク**

- template_task_id の管理が運用負債になりやすい（参照切れ/誤参照）
- “コードだけ見て全体が分かる” から遠ざかる（UI依存が強くなる）

---

### 案D: ハイブリッド（宣言的DAG + 動的step展開 + summaryは別枠）

**重視すること**

- タスク増加に耐えつつ、モデル数/推論trial数などの “動的なstep増加” を自然に扱う

**To-Be像（例）**

- pipeline spec に “展開子” を導入:
  - 例: `kind=train_model` を `models:` のリストで展開し、`train_model::<name>` を自動生成
  - その後に `kind=training_summary` を実行（集計専用）
- step展開の責務を `pipeline/expander.py` に閉じ込める

**決めるべき方針**

- “どこまでを動的展開の対象にするか”（trainingのモデル、inference optimize の trial など）
- 失敗時の挙動（部分成功でsummaryを出すか、失敗したら止めるか）

**必要な開発項目（例）**

- specのテンプレ/展開機構
- 展開後の UI 見通しを守る命名・ディレクトリ規約（`Training_runs/` / `Prediction_runs/` の統一）

**メリット**

- “編集容易性” と “動的step” を両立できる可能性がある

**デメリット/リスク**

- 設計範囲が広く、実装コストが大きい（段階導入が前提）

## 共通で決めるべき方針（議論ポイント）

### 1) 編集単位（ユーザーが変えたい粒度）

- pipelineで編集する対象は何か:
  - stepの有効無効
  - stepの順序/依存
  - stepごとの queue
  - stepごとの config（パス or dict）
- “ユーザーが編集して良い項目” と “参照用” をどう分けるか（HyperParameters汚染を避ける）

### 2) step間の受け渡し（Phase I/O contract）

- 返すべき最小情報は何か（dataset_id / preprocessed_dataset_id / model_id / task_id 等）
- “ファイル/Artifact/Dataset/Registry” のどれを正とするか

### 3) タスク増加とUI維持（探索性 vs シンプルさ）

- step数が増えても非MLが迷子にならない導線（見るべき親タスクを固定）
- “子タスクの置き場所” の規約（project suffix / ディレクトリ風の命名）

### 4) 互換性・移行戦略

- 既存の `config.yaml` / `config_training.yaml` 等とどう共存させるか
- 既存CLI（`automl_lib/cli/run_pipeline.py`）の引数や挙動をどこまで変えるか

## 判断の単位（受け入れ条件案）

- [ ] ステップ追加/削除/順序変更が “最小の差分” でできる（レビューしやすい）
- [ ] ステップが増えても controller/builder が肥大化しない（責務分離が保たれる）
- [ ] 既存の命名・tags・run_id 追跡が崩れない
- [ ] ClearML UI 上の HyperParameters/Plots/Artifacts の規約が守られる
- [ ] 失敗時でも “ユーザーが見るべき要約タスク” が残る（方針に沿って）

## 次アクション（将来タスク案）

1. まず “最初に増えそうなタスク” を列挙して、必要な編集（順序/条件/動的展開）の要件を確定
2. 案A〜Dのうち、短期はA、将来はB/D…など段階導入のロードマップを決める
3. Phase I/O contract（最低限の outputs）を固定し、step間の受け渡しを安定化
4. pipelineの “編集可能項目” をUI設計（HyperParameters最小化）とセットで定義

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/REFORM_PLAN.md`（または `Agents.md` に同内容あり）
- `automl_lib/pipeline/controller.py`

