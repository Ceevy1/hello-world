"""Project logger setup."""

from __future__ import annotations

import logging

import colorlog


def setup_logger(config: dict, debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else getattr(logging, config["project"].get("log_level", "INFO"))
    logger = logging.getLogger("transfer_learning")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
