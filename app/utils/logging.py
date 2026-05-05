"""Structured JSON logging."""

import logging
import sys
import uuid
from contextvars import ContextVar

from pythonjsonlogger import jsonlogger

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname.upper()
        log_record["logger"] = record.name
        log_record["timestamp"] = self.formatTime(record)
        rid = request_id_ctx.get("")
        if rid:
            log_record["request_id"] = rid


def setup_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger("learning_companion")
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    log.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        CustomJsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    log.addHandler(handler)
    log.propagate = False
    return log


def generate_request_id() -> str:
    return str(uuid.uuid4())[:8]


logger = setup_logging()
