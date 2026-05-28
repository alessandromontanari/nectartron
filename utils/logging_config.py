import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
        log_file_name,        
        logger_name="nectartron", 
        log_level=logging.INFO,
    ):
    """Set up a rotating file logger for the main scripts

    Parameters
    ----------
    log_file_name : str
        Base name for the log file
    logger_name : str, optional
        Name of the logger, by default 'nectartron'
    log_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG), by default logging.INFO
        
    Returns
    -------
    logging.Logger
        Configured logger
    """

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{logger_name}-{log_file_name}.log")

    logger = logging.getLogger(f"{logger_name}-{log_file_name}")
    logger.setLevel(log_level)

    # Avoid duplicate handlers if the function is called multiple times
    if logger.handlers:
        return logger

    # Rotating file handler for when the app will be running continously
    handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per log file
        backupCount=3,  # Keep up to 3 backup log files
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(funcName)s (line %(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


def log_and_print(logger, message, level="info"):
    """Log a message and print it to the console."""
    if level not in ["info", "warning", "error"]:
        raise ValueError(
            "Invalid log level. Use 'info', 'warning', or 'error'."
        )
    if level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    print(message)