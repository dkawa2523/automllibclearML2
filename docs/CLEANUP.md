# Cleanup / Unused code removal

本リポジトリは度重なる追加実装で不要コードが混入しやすいため、削除は「主観」ではなく機械的に確定してから行います。

## 0) ワンコマンド監査（推奨）

vulture + coverage をまとめて実行し、レポートを生成します。

```bash
./.venv/bin/python scripts/unused_code_audit.py --out-dir outputs/cleanup
```

出力：
- `outputs/cleanup/unused_code_report.md`
- `outputs/cleanup/unused_code_report.json`
- `outputs/cleanup/vulture.txt`
- `outputs/cleanup/coverage_report.txt`
- `outputs/cleanup/coverage.json`

### coverage を「CLI/pipeline の代表経路」まで広げる（推奨）

unit test だけだと通らない経路（CLI 実行・PipelineController の組み立て等）を追加でトレースしたい場合は、
`--coverage-cmd` を追加します。

同梱の `scripts/coverage_entrypoints_smoke.py` は以下を offline-safe で実行します:

- training CLI（ローカルCSV・ClearML無効）→ `best_model.joblib` を生成
- inference CLI（`model_path` で推論）
- pipeline controller の組み立て（PipelineController は stub で置換し、実タスク実行はしない）

```bash
./.venv/bin/python scripts/unused_code_audit.py \
  --out-dir outputs/cleanup \
  --coverage-cmd "scripts/coverage_entrypoints_smoke.py"
```

## 1) ツール導入（dev）

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
```

ネットワーク制限などで `pip install` ができない場合は、まず静的 import グラフから「到達しないモジュール」を機械的に抽出できます（精度は vulture/coverage より落ちますが、最初の足がかりになります）。

```bash
./.venv/bin/python scripts/import_reachability.py --json outputs/import_reachability.json
```

## 2) 静的解析（未使用候補の抽出）

```bash
./.venv/bin/vulture automl_lib --min-confidence 80
./.venv/bin/ruff check automl_lib
```

## 3) 動的解析（実行トレースで到達しない箇所を確定）

```bash
./.venv/bin/python -m coverage run -m unittest discover -s tests -v
./.venv/bin/python -m coverage report -m
```

## 4) 削除の進め方（安全策 / 2段階）

- 参照箇所（`rg -n "<name>"`）がゼロのものから着手する
- いきなり大量削除せず、削除→ユニットテスト→次へ の反復で進める
- public API（CLI/設定キー/出力ファイル名）の互換が必要なら、deprecated期間を設ける
- Deprecated の記録は `docs/DEPRECATIONS.md` に残す（削除までのログを残す）

### 推奨フロー

1. `scripts/unused_code_audit.py` で候補抽出（vulture/coverage）
2. `docs/DEPRECATIONS.md` に追加（Stage 1: deprecated）
3. public entrypoint（CLI/export）から外す（必要なら warning 付きの互換層）
4. 次リリースで再度監査し、参照ゼロ・到達ゼロを確認して削除（Stage 2）
