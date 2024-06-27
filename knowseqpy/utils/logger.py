import logging
from datetime import datetime

# Global internal flag to know if the logger has already been configured
_logger_configured = False


def get_logger() -> logging.Logger:
    """
    Retrieves the configured logger instance. If the logger hasn't been configured yet, it sets it up first.
    """
    global _logger_configured
    if not _logger_configured:
        set_logger()
        _logger_configured = True
    return logging.getLogger()


def set_logger(enable_file_logging: bool = False, log_level: int = logging.INFO, propagate: bool = True):
    """
    Configures the logging system. This function sets up a logger with a stream handler,
    and optionally a file handler if enabled.

    Args:
        enable_file_logging: If True, logs will also be saved to a file with a timestamped filename.
        log_level: The logging level (default: logging.INFO).
        propagate: If False, prevents logs from propagating to the root logger to avoid duplicate logs.
    """
    global _logger_configured
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # Ensure handlers are not duplicated
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if enable_file_logging:
            file_handler = logging.FileHandler(f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_knowseq_logs.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = propagate

    _logger_configured = True
