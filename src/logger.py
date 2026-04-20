"""Structured logging setup with GPU monitoring for FinQA chatbot."""

import logging
import sys
import time
from typing import Any, Optional

import structlog
from structlog.processors import JSONRenderer, TimeStamper
from structlog.stdlib import add_log_level, filter_by_level

from src.config import config

# Try to import GPU monitoring
try:
    import pynvml
    GPU_AVAILABLE = True
    pynvml.nvmlInit()
except Exception:
    GPU_AVAILABLE = False


def get_gpu_utilization() -> Optional[dict[str, Any]]:
    """Get current GPU utilization metrics."""
    if not GPU_AVAILABLE or not config.performance.gpu_monitoring_enabled:
        return None

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return {
            "gpu_util_percent": util.gpu,
            "gpu_memory_used_mb": memory.used // (1024 * 1024),
            "gpu_memory_total_mb": memory.total // (1024 * 1024),
            "gpu_memory_percent": (memory.used / memory.total) * 100,
        }
    except Exception as e:
        return {"gpu_error": str(e)}


def add_gpu_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add GPU utilization to log context."""
    gpu_stats = get_gpu_utilization()
    if gpu_stats:
        event_dict["gpu"] = gpu_stats
    return event_dict


def add_app_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    """Add application context to logs."""
    event_dict["app"] = "finqa-chatbot"
    return event_dict


def setup_logging() -> None:
    """Configure structured logging with GPU monitoring."""

    # Determine if we want JSON or console output
    if config.logging.format == "json":
        renderer = JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog
    structlog.configure(
        processors=[
            filter_by_level,
            add_log_level,
            add_app_context,
            add_gpu_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.logging.level),
    )


class LoggerContext:
    """Context manager for logging with timing and automatic cleanup."""

    def __init__(
        self,
        logger: structlog.BoundLogger,
        operation: str,
        **kwargs: Any,
    ):
        self.logger = logger
        self.operation = operation
        self.context = kwargs
        self.start_time: Optional[float] = None

    def __enter__(self) -> structlog.BoundLogger:
        """Start timing and log entry."""
        self.start_time = time.time()
        self.logger.info(
            f"{self.operation}_started",
            operation=self.operation,
            **self.context,
        )
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log completion with duration."""
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0

        if exc_type is None:
            self.logger.info(
                f"{self.operation}_completed",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                **self.context,
            )
        else:
            self.logger.error(
                f"{self.operation}_failed",
                operation=self.operation,
                duration_ms=round(duration_ms, 2),
                error_type=exc_type.__name__,
                error_msg=str(exc_val),
                **self.context,
            )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance with the given name."""
    return structlog.get_logger(name)


# Initialize logging on module import
setup_logging()
