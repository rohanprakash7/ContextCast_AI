"""
Logging Configuration for Conversational RAG Platform
Centralized logging setup for the entire application
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (creates directory if needed)
    """
    # Use settings if not provided
    log_level = log_level or settings.LOG_LEVEL
    log_file = log_file or settings.LOG_FILE
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create handlers with UTF-8 encoding
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    # Set UTF-8 encoding for console (Windows fix)
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level")
    if log_file:
        logger.info(f"Logging to file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module
    
    Args:
        name: Name of the module (usually __name__)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("This is an info message")
    """
    return logging.getLogger(name)


# Initialize logging on import
setup_logging()