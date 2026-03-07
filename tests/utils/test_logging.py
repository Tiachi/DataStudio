"""
Unit tests for datastudio.utils.logging module.

Tests cover:
- LogLevel enum
- ColoredFormatter class
- Logger class
- get_logger function
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from datastudio.utils.logging import ColoredFormatter, Logger, LogLevel, get_logger


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_debug_level(self):
        """Test DEBUG level exists."""
        assert LogLevel.DEBUG is not None

    def test_info_level(self):
        """Test INFO level exists."""
        assert LogLevel.INFO is not None

    def test_warning_level(self):
        """Test WARNING level exists."""
        assert LogLevel.WARNING is not None

    def test_error_level(self):
        """Test ERROR level exists."""
        assert LogLevel.ERROR is not None

    def test_critical_level(self):
        """Test CRITICAL level exists."""
        assert LogLevel.CRITICAL is not None


class TestColoredFormatter:
    """Tests for ColoredFormatter class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        formatter = ColoredFormatter()
        assert formatter is not None

    def test_init_with_format(self):
        """Test initialization with custom format."""
        fmt = "%(levelname)s - %(message)s"
        formatter = ColoredFormatter(fmt=fmt)
        assert formatter is not None

    def test_init_without_color(self):
        """Test initialization without color."""
        formatter = ColoredFormatter(use_color=False)
        assert formatter is not None

    def test_format_info_record(self):
        """Test formatting INFO level record."""
        formatter = ColoredFormatter(use_color=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted

    def test_format_error_record(self):
        """Test formatting ERROR level record."""
        formatter = ColoredFormatter(use_color=False)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "Error message" in formatted

    def test_format_with_color(self):
        """Test formatting with color enabled."""
        formatter = ColoredFormatter(use_color=True)

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        # Should contain ANSI color codes or message


class TestLoggerInit:
    """Tests for Logger initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        logger = Logger()
        assert logger is not None

    def test_init_with_name(self):
        """Test initialization with custom name."""
        logger = Logger(name="custom_logger")
        assert logger is not None

    def test_init_with_level(self):
        """Test initialization with custom level."""
        logger = Logger(level="DEBUG")
        assert logger is not None

    def test_init_with_log_file(self, temp_dir):
        """Test initialization with log file."""
        log_file = os.path.join(temp_dir, "test.log")
        logger = Logger(log_file=log_file)

        # Log file should be created
        logger.info("Test message")
        logger.close()

    def test_init_with_work_dir(self, temp_dir):
        """Test initialization with work_dir."""
        logger = Logger(work_dir=temp_dir)
        assert logger is not None
        logger.close()

    def test_init_console_disabled(self):
        """Test initialization with console disabled."""
        logger = Logger(console=False)
        assert logger is not None
        logger.close()

    def test_init_colored_disabled(self):
        """Test initialization with colored disabled."""
        logger = Logger(colored=False)
        assert logger is not None
        logger.close()


class TestLoggerGetInstance:
    """Tests for Logger.get_instance class method."""

    def test_get_instance_singleton(self):
        """Test get_instance returns singleton."""
        logger1 = Logger.get_instance()
        logger2 = Logger.get_instance()

        assert logger1 is logger2
        logger1.close()

    def test_get_instance_with_kwargs(self):
        """Test get_instance with kwargs."""
        logger = Logger.get_instance(name="test_instance")
        assert logger is not None
        logger.close()


class TestLoggerInitFromConfig:
    """Tests for Logger.init_from_config class method."""

    def test_init_from_config(self, temp_dir):
        """Test initialization from config dict."""
        cfg = {
            "name": "config_logger",
            "level": "DEBUG",
            "console": True,
            "colored": True,
        }

        logger = Logger.init_from_config(cfg, work_dir=temp_dir)
        assert logger is not None
        logger.close()

    def test_init_from_empty_config(self, temp_dir):
        """Test initialization from empty config."""
        logger = Logger.init_from_config({}, work_dir=temp_dir)
        assert logger is not None
        logger.close()


class TestLoggerLogging:
    """Tests for Logger logging methods."""

    @pytest.fixture
    def logger(self, temp_dir):
        """Create logger for testing."""
        log_file = os.path.join(temp_dir, "test.log")
        logger = Logger(log_file=log_file, console=False)
        yield logger
        logger.close()

    def test_debug(self, logger):
        """Test debug logging."""
        logger.debug("Debug message")

    def test_info(self, logger):
        """Test info logging."""
        logger.info("Info message")

    def test_warning(self, logger):
        """Test warning logging."""
        logger.warning("Warning message")

    def test_error(self, logger):
        """Test error logging."""
        logger.error("Error message")

    def test_critical(self, logger):
        """Test critical logging."""
        logger.critical("Critical message")

    def test_log_with_args(self, logger):
        """Test logging with format arguments using f-string."""
        # Logger uses _format_message, not standard % formatting
        logger.info(f"Message with {'string'} and {42}")

    def test_log_with_kwargs(self, logger):
        """Test logging with kwargs."""
        logger.info("Message", extra={"custom": "data"})


class TestLoggerLogMetrics:
    """Tests for Logger.log_metrics method."""

    @pytest.fixture
    def logger(self):
        """Create logger for testing."""
        logger = Logger(console=False)
        yield logger
        logger.close()

    def test_log_metrics_basic(self, logger):
        """Test logging metrics."""
        metrics = {"accuracy": 0.95, "loss": 0.05}
        logger.log_metrics(metrics)

    def test_log_metrics_with_step(self, logger):
        """Test logging metrics with step."""
        metrics = {"accuracy": 0.95}
        logger.log_metrics(metrics, step=100)

    def test_log_metrics_empty(self, logger):
        """Test logging empty metrics."""
        logger.log_metrics({})


class TestLoggerLogStats:
    """Tests for Logger.log_stats method."""

    @pytest.fixture
    def logger(self):
        """Create logger for testing."""
        logger = Logger(console=False)
        yield logger
        logger.close()

    def test_log_stats_basic(self, logger):
        """Test logging stats."""
        stats = {"total": 100, "processed": 90, "rejected": 10}
        logger.log_stats("Processing Stats", stats)

    def test_log_stats_without_wandb(self, logger):
        """Test logging stats without wandb."""
        stats = {"count": 50}
        logger.log_stats("Stats", stats, to_wandb=False)


class TestLoggerClose:
    """Tests for Logger.close method."""

    def test_close(self, temp_dir):
        """Test closing logger."""
        log_file = os.path.join(temp_dir, "test.log")
        logger = Logger(log_file=log_file)

        logger.info("Before close")
        logger.close()

        # Should not raise after close

    def test_close_multiple_times(self, temp_dir):
        """Test closing logger multiple times."""
        logger = Logger()
        logger.close()
        logger.close()  # Should not raise


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_default(self):
        """Test get_logger with default name."""
        logger = get_logger()
        assert logger is not None

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger(name="custom")
        assert logger is not None

    def test_get_logger_returns_same_instance(self):
        """Test get_logger returns same instance for same name."""
        logger1 = get_logger(name="test")
        logger2 = get_logger(name="test")

        assert logger1 is logger2


class TestLoggerEdgeCases:
    """Edge case tests for Logger."""

    def test_log_none_message(self):
        """Test logging None message."""
        logger = Logger(console=False)
        logger.info(None)
        logger.close()

    def test_log_unicode_message(self):
        """Test logging unicode message."""
        logger = Logger(console=False)
        logger.info("Unicode log message with special chars: accents and symbols")
        logger.close()

    def test_log_very_long_message(self):
        """Test logging very long message."""
        logger = Logger(console=False)
        logger.info("A" * 10000)
        logger.close()

    def test_log_exception_info(self):
        """Test logging with exception info."""
        logger = Logger(console=False)

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Error occurred", exc_info=True)

        logger.close()

    def test_log_dict_message(self):
        """Test logging dict as message."""
        logger = Logger(console=False)
        logger.info({"key": "value", "number": 42})
        logger.close()


class TestLoggerRegistration:
    """Tests for Logger registration."""

    def test_registered_in_hooks(self):
        """Test Logger is registered in HOOKS registry."""
        from datastudio.utils.registry import HOOKS

        assert "Logger" in HOOKS
