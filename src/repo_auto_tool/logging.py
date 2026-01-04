"""Structured logging configuration for repo-auto-tool.

This module provides:
- JSON-formatted logging for structured log output
- Console and file output handlers
- Configurable log levels via CLI
"""

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

# Valid log level names
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
VALID_LOG_LEVELS: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging.

    Each log entry is a single-line JSON object with standardized fields:
    - timestamp: ISO 8601 format with timezone
    - level: Log level name (DEBUG, INFO, etc.)
    - logger: Logger name (typically module path)
    - message: The log message
    - Additional fields from extra data if provided
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A single-line JSON string representing the log entry.
        """
        log_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add any extra fields attached to the record
        # Standard LogRecord attributes to exclude
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                try:
                    # Ensure value is JSON serializable
                    json.dumps(value)
                    log_data[key] = value
                except (TypeError, ValueError):
                    log_data[key] = str(value)

        return json.dumps(log_data, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Format log records for readable console output.

    Uses a human-readable format with optional colors for different levels.
    """

    # ANSI color codes for different log levels
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True) -> None:
        """Initialize the console formatter.

        Args:
            use_colors: Whether to use ANSI colors in output.
        """
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record for console output.

        Args:
            record: The log record to format.

        Returns:
            A formatted string for console display.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = record.levelname

        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        message = record.getMessage()
        formatted = f"{timestamp} {level_str} {record.name}: {message}"

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    console_output: bool = True,
    json_format: bool = False,
    use_colors: bool = True,
) -> logging.Logger:
    """Configure logging for the application.

    Sets up the root logger with appropriate handlers for console and/or file
    output. Supports both human-readable and JSON-formatted logs.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to. If provided, logs are
            written to this file in JSON format.
        console_output: Whether to output logs to console (stderr).
        json_format: Whether to use JSON format for console output.
            File output is always JSON-formatted.
        use_colors: Whether to use ANSI colors in console output.
            Ignored if json_format is True.

    Returns:
        The configured root logger.

    Raises:
        ValueError: If an invalid log level is provided.

    Example:
        >>> setup_logging("DEBUG", log_file=Path("app.log"))
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Validate log level
    level_upper = level.upper()
    if level_upper not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: {level!r}. "
            f"Must be one of: {', '.join(sorted(VALID_LOG_LEVELS))}"
        )

    # Get numeric level
    numeric_level = getattr(logging, level_upper)

    # Get root logger for our package
    root_logger = logging.getLogger("repo_auto_tool")
    root_logger.setLevel(numeric_level)

    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(numeric_level)

        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter(use_colors=use_colors))

        root_logger.addHandler(console_handler)

    # File handler (always JSON format for structured log analysis)
    if log_file is not None:
        # Ensure parent directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(JSONFormatter())

        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    This is a convenience function that ensures loggers are properly
    namespaced under the repo_auto_tool package.

    Args:
        name: The logger name, typically __name__ from the calling module.

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)
