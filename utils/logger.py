# logger.py
import logging
import sys
from pathlib import Path


def get_logger(name, level=logging.INFO, log_file=None):
    """Get a logger with the specified name and configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_root_logger(level=logging.INFO, log_file=None):
    """Setup the root logger configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
