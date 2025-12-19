# ISSUE: 複数人で共同開発するための “開発ワークフロー To-Be” 案（品質・拡張性・運用の両立）

> 位置づけ: **将来検討（今の改良スコープ外）**

## 背景 / 現状（as-is）

- 本プロジェクトは ClearML 上で「run_id を軸に迷わない」ことを狙い、Phase（preprocess/train/infer）と ClearML 出力（HyperParameters/Plots/Artifacts/Properties）の責務分離を進めている。
- 一方、開発者が増えたり、追加フェーズ/モデル/可視化/推論モードが増えるほど、以下がリスクになる:
  - 追加方法が属人化（触る場所が複数、規約が曖昧）
  - ClearML UI の規約（HyperParameters最小、親=要約/子=詳細）が崩れ、ユーザーが迷う
  - run_id/tags/命名が揃わず、後から検索不能になる
  - “動くけど再現できない”/“テストがないため壊れたことに気付けない” が増える

## 目的（このISSUEで議論したいこと）

- 複数人で継続的に開発・改良・修正できるように、**共通の開発ワークフロー**と**守るべき規約**を To-Be として整理する
- 「何を重視するか」別に運用/開発の選択肢を用意し、プロジェクトの成熟度に合わせて段階導入できるようにする
- 新規追加（モデル/前処理/可視化/推論）時に **どこを触れば良いか一意**である状態を維持する

## 守りたい原則（前提）

- タスクI/O契約（dataset contract / model registry / inference artifacts）が崩れない
- ClearML UI ルール（親=要約、子=詳細、Plotsナンバリング、HyperParameters最小）が崩れない
- run_id / tags / 命名規約が崩れない（検索性が落ちない）
- 破壊的変更は段階的に（deprecated -> remove の2段階）

## To-Be案（“何を重視するか”別）

### 案A: 速度重視（軽量運用・小チーム向け）

**重視**

- 実装スピード、短いフィードバックループ
- ルールは最小限、しかし “ClearML UI 規約” と “I/O契約” は絶対に守る

**To-Be像**

- ブランチ運用: trunk-based（短命feature branch → PR）
- CI: 最小（unit test + ruff のみ）
- レビュー: “UI規約チェックリスト” を必須化（HyperParameters/Plots/Artifacts/Properties）

**決めるべき方針**

- 最小チェック項目（必ず守る規約）の定義
- “実験的機能” をどこまで許容するか（mainに入れる条件）

**必要な開発項目（例）**

- PRテンプレに ClearML 規約チェック（Plots番号、Artifacts必須、User Properties必須）を追加
- `docs/DEV_GUIDE.md` に “追加手順（モデル/前処理/指標/可視化）” の最短ルートを追記

---

### 案B: 品質重視（壊さない・再現性重視）

**重視**

- 回帰を防ぐ（プロダクトとしての安定）
- “ユーザー導線が壊れない” を機械的に保証する

**To-Be像**

- CI: ruff + unit tests + 型チェック（必要なら） + ClearML UI 検証スクリプト（サーバ接続が必要なら optional）
- 変更は “契約” に基づいて行い、契約変更は docs とテストを伴う

**決めるべき方針**

- “契約テスト” の範囲:
  - preprocess dataset contract の必須ファイル（schema/manifest/bundle 等）
  - training-summary の必須plots/artifacts/properties
  - inference の mode別必須出力
- 外部依存（ClearML server）が必要な検証を CI でどう扱うか（optional job / nightly / 手動）

**必要な開発項目（例）**

- verify scripts（`scripts/verify_clearml_*`）を “仕様の受け入れテスト” として拡張
- “UIスキーマ” を定義（必須plots/artifacts/properties）し、検証で参照する
- 破壊的変更の運用（`docs/DEPRECATIONS.md` のプロセス）を強化

---

### 案C: 拡張性重視（プラグイン/レジストリ中心で責務分離）

**重視**

- 新しいモデル/前処理/評価/可視化を安全に追加できる
- 追加時の差分が小さく、触る場所が一意

**To-Be像**

- 追加ポイントを明確化し、基本は `registry/*` + `workflow/<phase>/*` への追加だけで完結する
- “core（ClearML非依存）/workflow（I/O組み立て）/integrations（ClearML出力）” の境界を明示し、相互依存を制限する

**決めるべき方針**

- “拡張ポイントのAPI” をどこまで固定するか（引数/返り値/型）
- 新しいモデルが必要とする依存（xgboost/catboost等）をどう管理するか（extras、optional）

**必要な開発項目（例）**

- registry API を文書化（追加時の最小手順、禁止事項、UI規約）
- 型（pydantic / dataclass）で Phase I/O を固定し、拡張は optional に寄せる
- “追加したら必ず増えるべき可視化/Artifacts” のテンプレを用意

---

### 案D: 運用重視（ClearMLテンプレ/clone中心、実行の再現性を最大化）

**重視**

- ClearML の運用（テンプレ、queue、agent）をプロジェクトの中心に置き、実行のブレを減らす

**To-Be像**

- “テンプレタスク” を正とし、configは最小差分だけを上書きして clone 実行する
- テンプレ更新の versioning と、過去runの再現性を運用で担保する

**決めるべき方針**

- テンプレの管理方法（task_id の保管場所、更新手順、ロールバック）
- “テンプレ更新” と “コード更新” の同期ルール（どちらが先か）

**必要な開発項目（例）**

- テンプレ一覧（task_id/用途/更新者/更新日）を docs に管理
- テンプレの smoke 実行（最小データ）を自動化

## 共同開発で決めるべき方針事項（議論ポイント）

### 1) 開発フロー（Git / PR / レビュー）

- ブランチ戦略（trunk-based / release branch）
- PRの最小要件（テスト・ドキュメント・UI規約チェック）
- コードオーナー（CODEOWNERS）やレビュー担当の決め方

### 2) 仕様の置き場所（Docs / Tests / Scripts）

- 仕様: docs（ユーザー/開発者/ISSUE）
- 受け入れ条件: verify scripts / contract tests
- 変更履歴: DEPRECATIONS / CHANGELOG（必要なら）

### 3) “タスクI/O契約” の変更管理

- dataset contract / inference outputs / training-summary outputs のバージョニング方針
- 互換性の壊し方（deprecated期間、移行手順）

### 4) ClearML UI規約の強制方法

- 人手レビュー（チェックリスト）で足りるか
- 機械検証（scripts）をどこまで必須にするか

### 5) 依存関係・実行環境の管理

- optional依存の扱い（extras/requirements-optional）
- GPU/大規模データを含む検証の扱い（手動/夜間/別環境）

## 判断の単位（受け入れ条件案）

- [ ] 新しいモデル/前処理/指標/可視化の追加が “一意の場所” で完結する（最小差分でレビュー可能）
- [ ] ClearML 上で run_id を検索すれば、必ず要約タスクに辿れる（ユーザー迷子にならない）
- [ ] HyperParameters/Plots/Artifacts/Properties の規約が機械的に検証できる（少なくとも主要phase）
- [ ] 破壊的変更は deprecated を経由し、移行ガイドがある
- [ ] CI（または手順）で回帰が検知できる（少なくとも unit/contract）

## 次アクション（将来タスク案）

1. “重視する案” を決める（短期=案A/B、将来=案C/D など段階導入でも可）
2. 最小のチェックリスト（UI規約 + I/O契約）を定義して PR テンプレ化
3. 契約テスト/verify scripts の対象範囲を決め、最小から導入
4. 拡張ポイントの “追加手順テンプレ” を整備（docs + サンプル）

## 関連リンク（プロジェクト内）

- `docs/DEV_GUIDE.md`
- `docs/DEPRECATIONS.md`
- `docs/CLEANUP.md`

