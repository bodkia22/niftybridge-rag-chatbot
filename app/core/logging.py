import logging


def setup_logging(level: str) -> None:
    """Configures the root logger for the application.

    Called once at application startup (or at the top of standalone
    scripts) so all modules can use `logging.getLogger(__name__)` and
    have their messages formatted and filtered consistently.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Third-party libraries are noisy at INFO (httpx logs every HTTP
    # request, huggingface_hub logs cache/auth warnings) — silence them
    # to WARNING so application logs stay readable, without lowering the
    # app's own log level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)