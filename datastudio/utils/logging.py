"""Unified logging system.

Provides a configurable :class:`Logger` with file, console, and
optional Weights & Biases backends.
"""

import logging
import os
import sys
import time
from enum import Enum
from typing import Any, Dict, Optional, Union

from datastudio.utils.registry import HOOKS

# Optional wandb import
try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class LogLevel(Enum):
    """Log levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ANSI color codes for terminal output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[34m",  # Blue
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
}


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    def __init__(self, fmt: str = None, datefmt: str = None, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}[{levelname}]{COLORS['RESET']}"
        return super().format(record)


@HOOKS.register_module()
class Logger:
    """Configurable logger with file, console, and WandB backends.

    Example config::

        logger = dict(
            type='Logger',
            level='INFO',
            log_file='logs/run.log',
            console=True,
            wandb=dict(enabled=True, project='datastudio', name='exp_1'),
        )
    """

    _instance: Optional["Logger"] = None

    def __init__(
        self,
        name: str = "datastudio",
        level: Union[str, int] = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
        colored: bool = True,
        work_dir: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        timestamp: bool = True,
        **kwargs,
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(
            getattr(logging, level.upper()) if isinstance(level, str) else level
        )
        self._logger.handlers.clear()

        self.work_dir = work_dir or os.getcwd()
        self._wandb_enabled = False

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self._logger.level)
            fmt = "%(asctime)s %(levelname)s %(message)s"
            datefmt = "%Y-%m-%d %H:%M:%S"
            formatter = ColoredFormatter(fmt, datefmt, use_color=colored)
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        # File handler
        if log_file:
            # Add timestamp to log filename if enabled
            if timestamp:
                log_file = self._add_timestamp_to_filename(log_file)

            log_path = (
                log_file
                if os.path.isabs(log_file)
                else os.path.join(self.work_dir, log_file)
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(self._logger.level)
            fmt = "%(asctime)s [%(levelname)s] %(message)s"
            file_handler.setFormatter(logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S"))
            self._logger.addHandler(file_handler)

        # WandB handler
        if wandb_config and wandb_config.get("enabled", False):
            self._init_wandb(wandb_config)

    def _add_timestamp_to_filename(self, filename: str) -> str:
        """Add timestamp to filename before extension.

        Example: logs/run.log -> logs/run_20251003123456.log
        """
        timestamp_str = time.strftime("%Y%m%d%H%M%S")
        base, ext = os.path.splitext(filename)
        return f"{base}_{timestamp_str}{ext}"

    def _init_wandb(self, config: Dict) -> None:
        """Initialize wandb logging."""
        if not HAS_WANDB:
            self._logger.warning("wandb not installed, skipping wandb logging")
            return

        try:
            wandb.init(
                project=config.get("project", "datastudio"),
                name=config.get("name"),
                config=config.get("config"),
                dir=self.work_dir,
                entity=config.get("entity"),
                reinit=True,
            )
            self._wandb_enabled = True
            self._logger.info(
                f"WandB initialized: project={config.get('project')}, name={config.get('name')}"
            )
        except Exception as e:
            self._logger.warning(f"Failed to initialize wandb: {e}")

    @classmethod
    def get_instance(cls, **kwargs) -> "Logger":
        """Get or create singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def init_from_config(cls, cfg: Dict, work_dir: Optional[str] = None) -> "Logger":
        """
        Initialize logger from config dict.

        Args:
            cfg: Logger configuration dict
            work_dir: Working directory for log files
        """
        config = cfg.copy() if cfg else {}
        config.pop("type", None)  # Remove type key if present
        if work_dir:
            config["work_dir"] = work_dir
        cls._instance = cls(**config)
        return cls._instance

    def _format_message(self, msg: Any, **kwargs) -> str:
        """Format message with additional key-value pairs."""
        if isinstance(msg, dict):
            parts = []
            for k, v in msg.items():
                if k:
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.4f}")
                    else:
                        parts.append(f"{k}={v}")
                else:
                    parts.append(str(v))
            msg_str = ", ".join(parts)
        else:
            msg_str = str(msg)

        if kwargs:
            extra = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in kwargs.items()
            )
            msg_str = f"{msg_str}, {extra}" if msg_str else extra

        return msg_str

    def _log_to_wandb(self, data: Dict, step: Optional[int] = None) -> None:
        """Log metrics to wandb."""
        if self._wandb_enabled and HAS_WANDB:
            try:
                wandb.log(data, step=step)
            except Exception:
                pass

    def _log_text_to_wandb(
        self, key: str, text: str, step: Optional[int] = None
    ) -> None:
        """Log text message to wandb."""
        if self._wandb_enabled and HAS_WANDB:
            try:
                # Use wandb.log with text - shows in run logs
                wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            except Exception:
                pass

    def debug(self, msg: Any, to_wandb: bool = False, **kwargs) -> None:
        """Log debug message."""
        formatted = self._format_message(msg, **kwargs)
        self._logger.debug(formatted)
        if to_wandb and self._wandb_enabled:
            self._log_to_wandb({"debug": formatted})

    def info(self, msg: Any, to_wandb: bool = False, **kwargs) -> None:
        """
        Log info message.

        Args:
            msg: Message (str or dict). Dict auto-logged to wandb.
            to_wandb: If True, also log string message to wandb.
        """
        formatted = self._format_message(msg, **kwargs)
        self._logger.info(formatted)
        if self._wandb_enabled:
            if isinstance(msg, dict):
                self._log_to_wandb(msg)
            elif to_wandb:
                self._log_to_wandb({"log": formatted})

    def warning(self, msg: Any, to_wandb: bool = False, **kwargs) -> None:
        """Log warning message."""
        formatted = self._format_message(msg, **kwargs)
        self._logger.warning(formatted)
        if to_wandb and self._wandb_enabled:
            self._log_to_wandb({"warning": formatted})

    def error(self, msg: Any, to_wandb: bool = True, **kwargs) -> None:
        """Log error message. Errors are logged to wandb by default."""
        formatted = self._format_message(msg, **kwargs)
        self._logger.error(formatted)
        if to_wandb and self._wandb_enabled:
            self._log_to_wandb({"error": formatted})

    def critical(self, msg: Any, to_wandb: bool = True, **kwargs) -> None:
        """Log critical message. Critical errors are logged to wandb by default."""
        formatted = self._format_message(msg, **kwargs)
        self._logger.critical(formatted)
        if to_wandb and self._wandb_enabled:
            self._log_to_wandb({"critical": formatted})

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics for tracking (to file and wandb).

        Args:
            metrics: Dictionary of metric name -> value
            step: Optional step number
        """
        # Format for file/console
        parts = []
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6f}")
            else:
                parts.append(f"{k}={v}")

        step_str = f"[step={step}] " if step is not None else ""
        self._logger.info(f"{step_str}{', '.join(parts)}")

        # Log to wandb
        self._log_to_wandb(metrics, step)

    def log_stats(
        self, title: str, stats: Dict[str, Any], to_wandb: bool = True
    ) -> None:
        """
        Log statistics summary.

        Args:
            title: Title for the stats section
            stats: Dictionary of stat name -> value
            to_wandb: Whether to log to wandb (default True)
        """
        self._logger.info(f"{'='*20} {title} {'='*20}")
        for k, v in stats.items():
            if isinstance(v, float):
                self._logger.info(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                self._logger.info(f"  {k}:")
                for kk, vv in v.items():
                    self._logger.info(f"    {kk}: {vv}")
            else:
                self._logger.info(f"  {k}: {v}")
        self._logger.info("=" * (42 + len(title)))

        # Log numeric stats to wandb
        if to_wandb and self._wandb_enabled:
            wandb_stats = {
                f"{title}/{k}": v
                for k, v in stats.items()
                if isinstance(v, (int, float))
            }
            if wandb_stats:
                self._log_to_wandb(wandb_stats)

    def close(self) -> None:
        """Close logger and cleanup."""
        if self._wandb_enabled and HAS_WANDB:
            try:
                wandb.finish()
            except Exception:
                pass

        for handler in self._logger.handlers[:]:
            handler.close()
            self._logger.removeHandler(handler)


def get_logger(name: str = "datastudio") -> Logger:
    """
    Get the global DataStudio logger instance.

    Args:
        name: Logger name (ignored if instance exists)

    Returns:
        Logger instance
    """
    if Logger._instance is None:
        Logger._instance = Logger(name=name)
    return Logger._instance
