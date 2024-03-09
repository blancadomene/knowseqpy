import logging
from datetime import datetime

# Global internal flag to know if the logger has already been configured
_logger_configured = False


def get_logger():
    global _logger_configured
    if not _logger_configured:
        set_logger()
        _logger_configured = True
    return logging.getLogger()


def set_logger(enable_file_logging=False, log_level=logging.INFO, propagate=True):
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

    # If set to False, prevent logs from propagating to the root logger to avoid duplicate logs
    logger.propagate = propagate

    _logger_configured = True
