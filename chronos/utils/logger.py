import sys
from loguru import logger
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "local")

def setup_logger():
    logger.remove()

    if ENVIRONMENT == "production":
        logger.add(sys.stderr, serialize=True)
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

    return logger

log = setup_logger()