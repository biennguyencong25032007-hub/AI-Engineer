"""
Structured Logging với structlog (fallback to standard logging).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False
    structlog = None

try:
    from src.config import config
except ImportError:
    class config:
        log_dir = Path("./logs")
        verbose = False


class SimpleLogger:
    """Simple logger fallback when structlog not available."""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._logger.addHandler(handler)

    def info(self, msg, **kwargs): self._logger.info(msg)
    def debug(self, msg, **kwargs): self._logger.debug(msg)
    def warning(self, msg, **kwargs): self._logger.warning(msg)
    def error(self, msg, **kwargs): self._logger.error(msg)
    def __getattr__(self, name):
        return getattr(self._logger, name)


def setup_logger(name: str | None = None):
    """
    Setup logger với:
    - Console output với màu sắc (nếu structlog)
    - File logging
    """
    # Ensure log directory
    config.log_dir.mkdir(parents=True, exist_ok=True)

    if HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if not sys.stdout.isatty() else
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.DEBUG if config.verbose else logging.INFO
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(name)
    else:
        return SimpleLogger(name)


def get_logger(name: str | None = None):
    """Get or create a logger."""
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return SimpleLogger(name)
