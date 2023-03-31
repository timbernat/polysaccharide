import logging
from pathlib import Path
from typing import Iterable

# Logging config
LOG_FORMATTER = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)-7s:%(module)16s:line %(lineno)-3d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class MultiLogFileHandler(logging.FileHandler):
    '''Subclass to cut down boilerplate of logfile I/O for loggers with multiple origins'''
    def add_to_loggers(self, *loggers : list[logging.Logger]) -> None:
        for logger in loggers:
            logger.addHandler(self)

    def remove_from_loggers(self, *loggers : list[logging.Logger]) -> None:
        for logger in loggers:
            logger.removeHandler(self)

def config_mlf_handler(log_path : Path, loggers : Iterable[logging.Logger], writemode : str='a', formatter : logging.Formatter=LOG_FORMATTER) -> MultiLogFileHandler:
    '''Further boilerplate reduction for configuring logging to file'''
    handler = MultiLogFileHandler(log_path, mode=writemode)
    handler.setFormatter(formatter)
    handler.add_to_loggers(*loggers)

    return handler