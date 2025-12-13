from __future__ import annotations

import os
import logging
import warnings
from pathlib import Path


def _suppress_noisy_warnings() -> None:
    """
    Best-effort suppression for noisy warnings in local ClearML workflows.

    Opt-out:
      AUTO_ML_SUPPRESS_WARNINGS=0
    """

    flag = str(os.environ.get("AUTO_ML_SUPPRESS_WARNINGS", "1")).strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return

    # ClearML logs this as a warning (not Python warnings).
    try:
        logger = logging.getLogger("clearml")
        if not getattr(logger, "_automl_noise_filter_installed", False):
            class _AutoMLClearMLNoiseFilter(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
                    try:
                        msg = record.getMessage()
                    except Exception:
                        return True
                    if msg.startswith(
                        "InsecureRequestWarning: Certificate verification is disabled! Adding "
                        "certificate verification is strongly advised."
                    ):
                        return False
                    return True

            logger.addFilter(_AutoMLClearMLNoiseFilter())
            setattr(logger, "_automl_noise_filter_installed", True)
    except Exception:
        pass

    # urllib3 InsecureRequestWarning (verify_certificate=false, local dev)
    try:
        from urllib3.exceptions import InsecureRequestWarning  # type: ignore

        warnings.filterwarnings("ignore", category=InsecureRequestWarning)
    except Exception:
        pass

    # matplotlib legend warning emitted through ClearML's plotlympl renderer
    try:
        warnings.filterwarnings(
            "ignore",
            message=r"No artists with labels found to put in legend.*",
            category=UserWarning,
        )
    except Exception:
        pass


def ensure_clearml_config_file() -> None:
    """
    Ensure ClearML can find a config file even when running under subprocesses
    (e.g., ClearML PipelineController local steps).

    If the user already set CLEARML_CONFIG_FILE we do nothing. Otherwise we look
    for `clearml.conf` at the repository root and point ClearML to it.
    """

    if os.environ.get("CLEARML_CONFIG_FILE"):
        return
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "clearml.conf"
    if candidate.exists():
        os.environ["CLEARML_CONFIG_FILE"] = str(candidate)


_suppress_noisy_warnings()
