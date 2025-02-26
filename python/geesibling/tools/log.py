"""
setup log and reimport log functions
"""

import logging
import os

TRACE_LEVEL_NUM = logging.DEBUG - 5
TRACE_LEVEL_NAME = "TRACE"
logging.addLevelName(TRACE_LEVEL_NUM, TRACE_LEVEL_NAME)


def _trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(TRACE_LEVEL_NUM, message, args, **kws)  # pylint: disable=protected-access


logging.Logger.trace = _trace


class ColorFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    trace = "\x1b[38;5;182m"
    debug = "\x1b[38;5;111m"
    info = "\x1b[38;5;253m"
    warning = "\x1b[33;1m"
    error = "\x1b[38;5;9m"
    critical = "\x1b[38;5;141m"
    reset = "\x1b[0m"
    format_spec = "{}%(asctime)s - %(name)s (%(filename)s:%(lineno)d) - %(levelname)s]{} %(message)s"

    FORMATS = {
        TRACE_LEVEL_NUM: format_spec.format(trace, reset),
        logging.DEBUG: format_spec.format(debug, reset),
        logging.INFO: format_spec.format(info, reset),
        logging.WARNING: format_spec.format(warning, reset),
        logging.ERROR: format_spec.format(error, reset),
        logging.CRITICAL: format_spec.format(critical, reset),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


LOG_LEVEL = os.getenv("GEESIBLING_LOG_LEVEL")
ch = logging.StreamHandler()
ch.setFormatter(ColorFormatter())
logger = logging.getLogger("geesibling")
try:
    level = LOG_LEVEL.upper() if LOG_LEVEL else logging.INFO
    logger.setLevel(level)
except ValueError:
    logger.setLevel(logging.INFO)

logger.handlers = (ch,)
logger.propagate = False

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical
trace = logger.trace
