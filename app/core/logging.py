import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_step(logger: logging.Logger, agent: str, run_id: str, step: str, status: str, duration_ms: int | None = None) -> None:
    """Structured step log. Only ids, step names, status and timing — never headers, tokens or PII."""
    duration = f" duration={duration_ms}ms" if duration_ms is not None else ""
    logger.info("[%s] run=%s step=%s status=%s%s", agent, run_id, step, status, duration)
