"""
Logging helpers for pyArgos.

Provides convenience functions for obtaining loggers by class or instance,
a custom ``EXECUTION`` log level (15, between DEBUG and INFO), and
configuration initialization from ``~/.pyargos/log/argosLogging.config``.
"""

import json
import logging
import logging.config
import os.path
import pathlib
from importlib.resources import files
from typing import List

#: Custom log level between DEBUG (10) and INFO (20), used for
#: step-by-step execution progress messages.
EXECUTION = 15

#: Default directory for pyArgos log configuration and log files.
ARGOS_DEFAULT_LOG_DIR = pathlib.Path.home() / ".pyargos" / "log"

UNIFY_ALL_LOGS_DEBUG = False

# This function is named to match the style of the stdlib logging module
# noinspection PyPep8Naming
def getClassLogger(cls: type):
    """
    Get a logger named after a class using its fully qualified name.

    The logger name is built from ``cls.__module__`` and ``cls.__qualname__``
    (e.g., ``"argos.manager.experimentManager"``).

    Parameters
    ----------
    cls : type
        The class to get the logger for.

    Returns
    -------
    logging.Logger
        A logger named ``"<module>.<qualname>"``.
    """
    name = cls.__module__ + "." + cls.__qualname__
    return logging.getLogger(name)


def get_logger(instance, name=None):
    """
    Get a logger for a class instance.

    If ``UNIFY_ALL_LOGS_DEBUG`` is True, returns the root logger
    (useful for debugging when you want all logs merged).

    Parameters
    ----------
    instance : object
        The class instance to get the logger for.
    name : str, optional
        If provided, returns a logger with this name instead of
        deriving it from the instance's class.

    Returns
    -------
    logging.Logger
        A logger for the instance's class, or a named logger if
        ``name`` is specified.
    """
    global UNIFY_ALL_LOGS_DEBUG
    if UNIFY_ALL_LOGS_DEBUG:
        return logging.getLogger("")
    return getClassLogger(instance.__class__) if name is None else logging.getLogger(name)

def get_classMethod_logger(instance, name=None):
    """
    Get a logger for a specific method of a class instance.

    The logger name includes the class's fully qualified name plus
    the method name (e.g., ``"argos.manager.experimentManager.loadDevices"``).

    Parameters
    ----------
    instance : object
        The class instance.
    name : str, optional
        The method name to append to the logger name. If None, uses
        the class name only.

    Returns
    -------
    logging.Logger
        A logger named ``"<module>.<qualname>.<name>"``.
    """
    lgname = instance.__class__ if name is None else instance.__class__.__module__ + "." + instance.__class__.__qualname__ + "." + name
    return logging.getLogger(lgname)

def unify_all_logs_debug():
    """
    Merge all loggers to the root logger for debugging.

    When enabled, all calls to :func:`get_logger` return the root logger,
    so all log messages go to a single output. This is useful for debugging
    but should not be used in production.
    """
    global UNIFY_ALL_LOGS_DEBUG
    UNIFY_ALL_LOGS_DEBUG = True



def get_default_logging_config(*, disable_existing_loggers: bool = False) -> dict:
    """
    Load or create the default pyArgos logging configuration.

    Reads the config from ``~/.pyargos/log/argosLogging.config``. If the
    file doesn't exist, copies the default config from the package resources.

    The ``{argos_log}`` placeholder in the config is replaced with the
    actual log directory path.

    Parameters
    ----------
    disable_existing_loggers : bool, optional
        Whether to disable loggers that were created before this config
        is applied. Defaults to False.

    Returns
    -------
    dict
        A logging configuration dictionary suitable for
        ``logging.config.dictConfig()``.
    """
    defaultLocalConfig = os.path.join(ARGOS_DEFAULT_LOG_DIR, 'argosLogging.config')
    if not os.path.isfile(defaultLocalConfig):
        defaultConfig = files('argos.utils.logging').joinpath('argosLogging.config').read_text()
        with open(defaultLocalConfig,'w') as localConfig:
            localConfig.write(defaultConfig)

    with open(defaultLocalConfig,'r') as localConfig:
        config_text = "\n".join(localConfig.readlines())

    config_text = config_text.replace("{argos_log}", str(ARGOS_DEFAULT_LOG_DIR))
    config = json.loads(config_text)
    assert isinstance(config, dict)
    config['disable_existing_loggers'] = disable_existing_loggers
    return config


def _define_logger_execution():
    """Register the custom EXECUTION log level and add the execution() method to Logger."""
    def execution(self, message, *args, **kws):
        self.log(EXECUTION, message, *args, **kws)

    logging.Logger.execution = execution
    logging.addLevelName(EXECUTION, 'EXECUTION')
    logging.EXECUTION = EXECUTION


def initialize_logging(*logger_overrides: (str, dict), disable_existing_loggers: bool = True) -> None:
    """
    Initialize logging for the Hera library

    Calling this function configures loggers for the hera library, according to
    the default configuration, with modifications you can add using :py:func:`with_logger` calls.

    The function is called automatically when hera is imported, so you only need
    to call it explicitly if you really want to add such modifications.

    To override e.g. the ``hera.bin`` logger's level to ``DEBUG``, call like so::

        initialize_logging(
            with_logger("hera.bin", level="DEBUG"),
        )

    This will use the rest of the definitions for this logger (handlers, formatters,
    whatever other parameters the logger class takes) from the default configuration,
    if such definitions exist.

    As the ``initialize`` in the name implies, you're expected to call this
    function, if at all, before you start getting logger objects. If you call it
    after some loggers were created, consider passing ``disable_existing_loggers=False``.
    """
    if not os.path.isdir(ARGOS_DEFAULT_LOG_DIR):
        os.makedirs(ARGOS_DEFAULT_LOG_DIR, exist_ok=False)
    _define_logger_execution()
    config = get_default_logging_config(disable_existing_loggers=disable_existing_loggers)
    for logger_name, logger_dict in logger_overrides:
        # This says: Use whatever was configured, if any, and update with what was provided
        config['loggers'].setdefault(logger_name, logger_dict).update(logger_dict)
    logging.config.dictConfig(config)


def with_logger(logger_name, level=None, handlers=None, propagate=None) -> (str, dict):
    """
    Build a logger override tuple for use with :func:`initialize_logging`.

    Creates a ``(name, config_dict)`` tuple that can be passed as a
    positional argument to ``initialize_logging()``. Only non-None
    parameters are included in the config dict, so existing settings
    for omitted parameters are preserved.

    Parameters
    ----------
    logger_name : str
        The logger name (e.g., ``"argos.kafka"``).
    level : str, optional
        The log level (e.g., ``"DEBUG"``, ``"INFO"``).
    handlers : list[str], optional
        List of handler names to attach to the logger.
    propagate : bool, optional
        Whether to propagate messages to parent loggers.

    Returns
    -------
    tuple[str, dict]
        A ``(logger_name, config_dict)`` tuple.

    Examples
    --------
    >>> initialize_logging(
    ...     with_logger("argos.kafka", level="DEBUG"),
    ...     with_logger("argos.manager", level="INFO", handlers=["console"]),
    ... )
    """
    logger_dict = dict(level=level, handlers=handlers, propagate=propagate)
    # Remove from dict parameters not supplied; this allows the use here
    # to just override specific settings on existing loggers
    empty = [key for key, value in logger_dict.items() if value is None]
    for key in empty:
        del logger_dict[key]

    return logger_name, logger_dict
