# Plugins

このディレクトリは「プロジェクト固有の拡張」を置くためのスペースです。

## 例（追加の考え方）

- 独自の評価指標：`automl_lib/registry/metrics.py` に登録関数を追加するか、ここに登録用モジュールを作って `evaluation.plugins` から import する
- 独自のモデル：`automl_lib/registry/models.py` と同様に登録する
- 独自の前処理：`automl_lib/registry/preprocessors.py` と同様に登録する

## 使い方（最小）

1. `automl_lib/plugins/<your_plugin>.py` を作る（import時に registry 登録が走る形が簡単）
2. config の `evaluation.plugins`（または各種 plugin list）にモジュール名を追加する
3. `TrainingConfig` の validation で import/typo を早期検知できます

