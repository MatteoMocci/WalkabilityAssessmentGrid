"""Project logging setup with ASCII-safe console and rotating file handlers."""

import logging
from logging import handlers

class AsciiSanitizer(logging.Filter):
    """Ensure messages contain only ASCII; replace non-ASCII with escapes."""
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Sanitize log message to ASCII-safe text.

        Steps:
        1) Encode with backslash replacement.
        2) Decode back to ASCII string.
        """
        try:
            record.msg = str(record.msg).encode("ascii", "backslashreplace").decode("ascii")
        except Exception:
            pass
        return True

def get_logger(name="WalkCNN", level=logging.INFO, log_file="walkcnn.log"):
    """
    Build a console + rotating-file logger with ASCII sanitization.

    Steps:
    1) Create/get logger and set base level.
    2) Configure console handler.
    3) Configure rotating file handler.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S"))
    ch.addFilter(AsciiSanitizer())
    logger.addHandler(ch)

    fh = handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.addFilter(AsciiSanitizer())
    logger.addHandler(fh)

    return logger
