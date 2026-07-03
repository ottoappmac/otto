"""Logger utility module."""

import os
import sys
import logging
import inspect


LOG_LEVEL_MAPPING = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.CRITICAL,
}


def _get_log_level_from_env() -> int:
    """Get logging level from environment variable with fallback to INFO."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    return LOG_LEVEL_MAPPING.get(log_level_str, logging.INFO)


def init_logger():
    """Initialize the logger with console output."""
    log_level = _get_log_level_from_env()

    logging_handlers = [logging.StreamHandler(sys.stdout)]

    logformat = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=logformat, handlers=logging_handlers)


def get_logger(custom_logger_name: str | None = None) -> logging.Logger:
    """Get a logger instance for the calling module."""
    log_level = _get_log_level_from_env()

    # Get the name of the calling module
    caller_frame = inspect.stack()[1]
    caller_module = inspect.getmodule(caller_frame[0])
    logger_name = caller_module.__name__ if caller_module else ""

    if custom_logger_name:
        logger_name = custom_logger_name

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    return logger
