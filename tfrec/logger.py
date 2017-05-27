from __future__ import division, absolute_import, print_function
import logging


__all__ = ['easy_setup']


def _getRootLogger():
    """Return the top-most, root python logger object."""
    log = logging.getLogger()
    return log


def _getFormatter():
    """Return a formatter that does a good job showing helpful information."""
    fmt = '%(asctime)s: %(name)-18s: %(levelname)-10s: %(message)s'
    formatter = logging.Formatter(fmt)
    return formatter


def _init():
    """Set up the root logger.

    You should call this method first before you call _addFileHandler()
    and _addStreamHandler().
    """
    log = _getRootLogger()
    log.setLevel(logging.DEBUG)


def _addFileHandler(filename, log_level, log):
    """Add a file to the list of logging destinations of the given logger.

    This is helpful for debugging crashes that happen out in the wild
    while you're not looking.
    """
    handler = logging.FileHandler(filename)
    handler.setLevel(log_level)
    handler.setFormatter(_getFormatter())
    log.addHandler(handler)


def _addStreamHandler(log_level, log):
    """Add the console (aka, terminal) as a logging destination of the given logger.

    This is helpful for debugging while you're sitting at your console.
    """
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(_getFormatter())
    log.addHandler(handler)


def easy_setup(logger_name=None, console_output=False, filename=None):
    """Build a logger with the given name, optionally outputting to
    the console and/or to a file.

    A common use-case is to build a logger for each module by including
    a line like this at the top of each module file:
        log = logger.easy_setup(__name__, "{}_log.log".format(__name__))
    """
    _init()
    log = logging.getLogger(logger_name)
    if filename is not None:
        _addFileHandler(filename, logging.INFO, log)  # <-- log to a file
    if console_output:
        _addStreamHandler(logging.INFO, log)          # <-- log to the terminal
    return log

