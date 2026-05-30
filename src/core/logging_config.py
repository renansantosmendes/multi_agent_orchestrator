import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger with a stdout handler and standard formatter.

    Args:
        name: Logger name, typically __name__ of the calling module.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
