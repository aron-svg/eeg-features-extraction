#!/usr/bin/env python
import logging
import logging.config
import os
import sys

import yaml


class ColoredFormatter(logging.Formatter):
    """Wraps the whole formatted line in an ANSI color keyed by level, so
    INFO/WARNING/ERROR stand out from each other in a terminal."""

    COLORS = {
        logging.DEBUG: "\033[36m",  # cyan
        logging.INFO: "\033[32m",  # green
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def format(self, record):
        message = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        return f"{color}{message}{self.RESET}" if color else message


default_formatter = ColoredFormatter(
    "%(asctime)s | %(levelname)8s |  %(funcName)s | %(message)s",
    "%Y-%m-%d %H:%M:%S",
)

__logger = logging.getLogger(__name__)

__ch = logging.StreamHandler(sys.stdout)
__ch.setFormatter(default_formatter)

# default behavior
__logger.addHandler(__ch)

# create logs/ folder
if not os.path.exists("logs"):
    os.makedirs("logs")

# Parse config file - resolved relative to this file, not the cwd the
# interpreter happens to be launched from.
__logger_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logger_config.yaml")
if os.path.exists(__logger_config_file) and os.path.isfile(
    __logger_config_file
):
    try:
        log_cfg = yaml.safe_load(open(__logger_config_file))
        logging.config.dictConfig(log_cfg)   # NOSONAR

    except Exception as ex:
        __logger.error(
            f"Unexpected error whilst reading logger configuration file {__logger_config_file}"
        )
        __logger.error(f"Exception: {ex}")
        raise
else:
    __logger.warning(
        f"no configuration file {__logger_config_file} found, using default setting"
    )
