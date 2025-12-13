"""
automl_lib
-----------
新しい階層で各フェーズの処理・設定・ClearML連携を統一的に管理するためのパッケージ。
当面は既存の `auto_ml` 実装をラップしつつ、段階的にこちらへ機能を移行する。
"""

__all__ = ["config", "clearml", "phases", "pipeline", "registry"]
