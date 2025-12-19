# Deprecations

本リポジトリでは「削除」は2段階で行います（安全策）：

1. Deprecated（public API から外す／互換の猶予を設ける）
2. 次リリース以降に削除（vulture + coverage + rg で参照ゼロを確認）

## 追加ルール

- Deprecated しただけでは削除しない（必ずこのファイルに記録する）
- CLI / 設定キー / 出力ファイル名などの public API は、Deprecated 期間を設ける
- “未使用っぽい” は削除理由にならない（機械的確定：`scripts/unused_code_audit.py`）

---

## Deprecation List

### Template

```md
### <ID> <title>
- Status: deprecated | scheduled-removal | removed
- Added: YYYY-MM-DD
- Target removal: vX.Y.Z (or YYYY-MM-DD)
- Scope: module | cli | config | artifact | plot | other
- Affected: <paths / commands / keys>
- Replacement: <what to use instead>
- Rationale: <why>
- Notes: <migration steps / compatibility notes>
```

---

### D001 comparison phase removal
- Status: removed
- Added: 2025-12-18
- Target removal: v0.1.0
- Scope: module/cli
- Affected: `automl_lib/phases/comparison/*`, `automl_lib/workflow/comparison/*`, `automl_lib/cli/run_comparison.py`, `config_comparison.yaml`
- Replacement: none
- Rationale: comparison フェーズは不要（設計方針により廃止）
- Notes: 旧ファイルは削除済み

### D002 legacy phases package removal
- Status: removed
- Added: 2025-12-18
- Target removal: v0.1.0
- Scope: module
- Affected: `automl_lib/phases/*`
- Replacement: `automl_lib/workflow/*`
- Rationale: 旧アーキテクチャ（phases）を廃止し、workflow へ統一して責務分離・保守性を改善
- Notes: phase 実装は `automl_lib.workflow.<phase>` を参照

### D003 plugin examples removal
- Status: removed
- Added: 2025-12-18
- Target removal: v0.1.0
- Scope: module
- Affected: `automl_lib/plugins/*`
- Replacement: none（必要ならプロジェクト外部のモジュールを config の plugins で読み込む）
- Rationale: 未使用のサンプル実装を削除し、拡張ポイントを整理
- Notes: plugin 仕組み自体は残す（例は同梱しない）

### D004 legacy clearml wrapper package
- Status: deprecated
- Added: 2025-12-18
- Target removal: v0.2.0
- Scope: module
- Affected: `automl_lib/clearml/*`
- Replacement: `automl_lib/integrations/clearml/*`
- Rationale: ClearML 共通基盤を `automl_lib/integrations/clearml/` に集約し、二重構造による混乱を防ぐ
- Notes: 当面は互換ラッパとして残す（段階的に内部 import を置換）
