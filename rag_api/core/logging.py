"""Structured logging: key=value text or JSON lines, selected by LOG_FORMAT."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes every LogRecord has; anything else was passed through `extra=`.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in record.__dict__.items() if k not in _RESERVED}


class StructuredFormatter(logging.Formatter):

    def __init__(self, json_output: bool = False) -> None:
        super().__init__()
        self._json = json_output

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        extras = _extras(record)

        if self._json:
            entry: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                **extras,
            }
            if record.exc_info:
                entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(entry, default=str, ensure_ascii=False)

        line = (
            f"{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')} "
            f"{record.levelname:<7} {record.name}: {message}"
        )
        if extras:
            line += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(json_output=log_format.lower() == "json"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Library chatter (one line per HTTP request / HF hub call) is noise at INFO.
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
