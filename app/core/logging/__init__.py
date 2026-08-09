"""Minimal structured logging for the rank microservice.

Subset of the main API's logging system: provides the same public API
used by the copied worker/service modules (``setup_logging``,
``get_logger``, ``bind_context``) without the Sentry/interceptor/handler
machinery.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator


@dataclass
class LogContext:
    request_id: str = ""
    user_id: str = ""
    method: str = ""
    path: str = ""
    pipeline_stage: str = ""
    job_id: str = ""
    provider: str = ""
    model: str = ""
    worker_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "pipeline_stage": self.pipeline_stage,
            "job_id": self.job_id,
            "provider": self.provider,
            "model": self.model,
            "worker_id": self.worker_id,
        }
        d.update(self.extra)
        return {k: v for k, v in d.items() if v}


_log_context: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "log_context", default=LogContext()
)


def get_context() -> LogContext:
    return _log_context.get()


@contextmanager
def bind_context(**kwargs: Any) -> Generator[None, None, None]:
    """Context manager to enrich logs temporarily.

    Usage:
        async with bind_context(pipeline_stage="rank", job_id="..."):
            ...
    """
    current = _log_context.get()
    new_ctx = LogContext(
        request_id=current.request_id,
        user_id=current.user_id,
        method=current.method,
        path=current.path,
        pipeline_stage=kwargs.get("pipeline_stage", current.pipeline_stage),
        job_id=kwargs.get("job_id", current.job_id),
        provider=kwargs.get("provider", current.provider),
        model=kwargs.get("model", current.model),
        worker_id=kwargs.get("worker_id", current.worker_id),
        extra={
            **current.extra,
            **{k: v for k, v in kwargs.items()
               if k not in ("pipeline_stage", "job_id",
                            "provider", "model", "worker_id")},
        },
    )
    token = _log_context.set(new_ctx)
    try:
        yield
    finally:
        _log_context.reset(token)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def setup_logging(config: LoggingConfig | None = None) -> None:
    """Initialize logging ONCE at startup (call from create_app() and worker.py)."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"
        )
    )
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a hierarchical logger.

    Convention: ``get_logger(__name__)`` → ``"app.services.rank"``
    """
    return logging.getLogger(name)


class LoggingConfig:
    """Placeholder for API parity — the simplified setup needs no config."""

    level: str = "INFO"
