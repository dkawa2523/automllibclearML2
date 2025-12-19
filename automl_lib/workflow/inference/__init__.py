"""
inference workflow package
--------------------------
処理の入口（processing）を提供する。
"""

from .processing import run_inference_processing

__all__ = [
    "run_inference_processing",
]
