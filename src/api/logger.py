"""
src/api/logger.py

SOC-grade logging configuration for the AI-SOC Adversarial Defense System.

Design decisions
----------------
* Two handlers: rotating file handler (persistent audit trail) + coloured
  stream handler (operator terminal).
* A custom Formatter enriches every record with a severity tag so that
  SIEM/grep pipelines can filter on e.g. ``[CRITICAL]`` without parsing
  Python log-level integers.
* A dedicated ``soc_alert()`` helper emits CRITICAL records with a
  structured prefix that downstream tools can parse unambiguously.
* Module-level ``get_logger()`` factory ensures every sub-module gets a
  child of the same root logger – no duplicate handlers, unified config.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "soc_alerts.log"
_ROOT_LOGGER_NAME = "ai_soc"
_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per file
_BACKUP_COUNT = 5                # keep 5 rotated archives
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"

# ANSI colour codes – used only on the stream handler
_RESET = "\033[0m"
_COLOURS: dict[int, str] = {
    logging.DEBUG:    "\033[36m",   # Cyan
    logging.INFO:     "\033[32m",   # Green
    logging.WARNING:  "\033[33m",   # Yellow
    logging.ERROR:    "\033[31m",   # Red
    logging.CRITICAL: "\033[1;31m", # Bold Red
}


# ---------------------------------------------------------------------------
# Custom Formatters
# ---------------------------------------------------------------------------

class SOCFormatter(logging.Formatter):
    """
    Structured log formatter that prepends a bracketed severity tag to
    every message, enabling trivial grep / SIEM ingestion.

    Example output (file):
        2024-07-15T10:23:41 | [CRITICAL] | ai_soc.api.main | 187 |
        ATTACK DROPPED – anomaly_score=0.92 request_id=req-001
    """

    _SEVERITY_TAG: dict[int, str] = {
        logging.DEBUG:    "[DEBUG]   ",
        logging.INFO:     "[INFO]    ",
        logging.WARNING:  "[WARNING] ",
        logging.ERROR:    "[ERROR]   ",
        logging.CRITICAL: "[CRITICAL]",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        tag = self._SEVERITY_TAG.get(record.levelno, "[UNKNOWN] ")
        record.soc_tag = tag
        return super().format(record)


class ColouredSOCFormatter(SOCFormatter):
    """
    Extends SOCFormatter with ANSI colour codes for terminal output.
    Colours are stripped automatically when stdout is not a TTY so that
    piped output (e.g. ``| tee``) stays clean.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        formatted = super().format(record)
        if sys.stderr.isatty() or (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
            colour = _COLOURS.get(record.levelno, _RESET)
            return f"{colour}{formatted}{_RESET}"
        return formatted


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------

def _build_root_logger() -> logging.Logger:
    """
    Construct and configure the root ``ai_soc`` logger exactly once.
    Subsequent calls to ``get_logger()`` return children of this logger,
    inheriting all handlers via propagation.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    # Guard: if handlers already attached, the logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ------------------------------------------------------------------ #
    # File handler – rotating, plain-text, machine-parseable              #
    # ------------------------------------------------------------------ #
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(_LOG_FILE),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_fmt = SOCFormatter(
        fmt="%(asctime)s | %(soc_tag)s | %(name)s | %(lineno)d | %(message)s",
        datefmt=_DATE_FMT,
    )
    file_handler.setFormatter(file_fmt)

    # ------------------------------------------------------------------ #
    # Stream handler – coloured, human-readable, DEBUG and above          #
    # ------------------------------------------------------------------ #
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_fmt = ColouredSOCFormatter(
        fmt="%(asctime)s | %(soc_tag)s | %(name)-30s | %(message)s",
        datefmt=_DATE_FMT,
    )
    stream_handler.setFormatter(stream_fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Prevent log records from bubbling up to the root Python logger
    logger.propagate = False

    return logger


# Eagerly build the root logger when this module is imported so that all
# child loggers created via get_logger() are guaranteed to find it.
_root_logger: logging.Logger = _build_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger scoped to *name*.

    Usage::

        from src.api.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Inference complete.")

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.  The final logger
        name will be ``ai_soc.<name>`` so that log lines are traceable
        back to their source file without ambiguity.
    """
    # Strip the package prefix if already present to avoid double-nesting
    if name.startswith(_ROOT_LOGGER_NAME + "."):
        child_name = name
    else:
        # Normalise absolute module paths (e.g. src.api.main → src.api.main)
        child_name = f"{_ROOT_LOGGER_NAME}.{name}"

    return logging.getLogger(child_name)


# ---------------------------------------------------------------------------
# SOC Alert helper
# ---------------------------------------------------------------------------

def soc_alert(
    message: str,
    *,
    request_id: Optional[str] = None,
    anomaly_score: Optional[float] = None,
    extra: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Emit a ``CRITICAL`` log record using the SOC structured format.

    This helper should be called whenever an adversarial attack is **dropped**
    (i.e. the request is rejected entirely rather than sanitized).  The
    structured prefix allows downstream SIEM rules to fire on
    ``ATTACK DROPPED`` without brittle regex on free-form messages.

    Parameters
    ----------
    message:
        Human-readable description of the threat event.
    request_id:
        Caller-supplied correlation ID, echoed into the alert.
    anomaly_score:
        Normalised [0, 1] score from the Threat Detection module.
    extra:
        Arbitrary key-value pairs appended to the alert as ``key=value``.
    logger:
        Logger instance to use; defaults to the root ``ai_soc`` logger.
    """
    _logger = logger or _root_logger

    parts: list[str] = ["ATTACK DROPPED"]
    if request_id is not None:
        parts.append(f"request_id={request_id}")
    if anomaly_score is not None:
        parts.append(f"anomaly_score={anomaly_score:.4f}")
    if extra:
        parts.extend(f"{k}={v}" for k, v in extra.items())
    parts.append(f"– {message}")

    _logger.critical(" | ".join(parts))