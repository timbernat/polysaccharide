import logging
from logging import Logger

from pathlib import Path
from typing import Iterable, Union
from datetime import datetime


LOGGER_REGISTRY = logging.root.manager.loggerDict # dict of all extant Loggers through all loaded modules, keyed by logger name
LOG_FORMATTER   = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)-7s:%(module)16s:line %(lineno)-3d] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

class MultiStreamFileHandler(logging.FileHandler):
    '''Subclass to cut down boilerplate of logfile I/O for loggers with multiple origins'''
    def __init__(self, *args, loggers : Union[Logger, list[Logger]]=None, formatter : logging.Formatter=LOG_FORMATTER, proc_name : str='Process', **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.proc_name = proc_name
        self.setFormatter(formatter)
        self.personal_logger = logging.getLogger(str(self.__hash__())) # create unique logger for internal error logging
        self.personal_logger.addHandler(self)

        self._loggers = []
        if loggers is None:
            return
    
        if isinstance(loggers, Logger): # only reachable if loggers is excplicitly passed
            self.register_logger(loggers) # handle the singleton logger case
        else:
            self.register_loggers(*loggers) # handle a list of loggers

    def __enter__(self):
        self._start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exception, traceback):
        if exc_type is None:
            self.personal_logger.info(f'{self.proc_name} completed in {datetime.now() - self._start_time}\n')
        else:
            self.personal_logger.error(f'{exc_type.__name__} : {exception}\n')
        self.unregister_loggers() # prevents multiple redundant writes within the same Python session

        return True # TOSELF : return Falsy value only if errors are unhandled

    def register_logger(self, logger : Logger) -> None:
        '''Add an individual Logger to the File Stream'''
        logger.addHandler(self)
        self._loggers.append(logger)

    def unregister_logger(self, logger : Logger) -> None:
        '''Remove an individual Logger from the collection of linked Loggers'''
        logger.removeHandler(self)
        logger_idx = self._loggers.index(logger)
        self._loggers.pop(logger_idx)
    
    def register_loggers(self, *loggers : list[Logger]) -> None:
        '''Record a new Logger and add the File handler to it - enables support for multiple Logger streams to a single file'''
        for logger in loggers:
            self.register_logger(logger)

    def unregister_loggers(self) -> None:
        '''Purge all currently registered Loggers'''
        for logger in self._loggers: # TOSELF : can't just call individual unregister_logger() method, since pop will change list size while iterating (causes some Loggers to be "skipped")
            logger.removeHandler(self)
        self._loggers.clear()

# legacy versions, kept for backwards compatibility reasons        
class MultiLogFileHandler(logging.FileHandler):
    '''Subclass to cut down boilerplate of logfile I/O for loggers with multiple origins'''
    def __init__(self, *args, **kwargs) -> None:
        self._start_time = datetime.now()
        super().__init__(*args, **kwargs)

    @property
    def runtime(self):
        return datetime.now() - self._start_time

    def add_to_loggers(self, *loggers : list[Logger]) -> None:
        for logger in loggers:
            logger.addHandler(self)

    def remove_from_loggers(self, *loggers : list[Logger]) -> None:
        for logger in loggers:
            logger.removeHandler(self)

def config_mlf_handler(log_path : Path, loggers : Iterable[Logger], writemode : str='a', formatter : logging.Formatter=LOG_FORMATTER) -> MultiLogFileHandler:
    '''Further boilerplate reduction for configuring logging to file'''
    handler = MultiLogFileHandler(log_path, mode=writemode)
    handler.setFormatter(formatter)
    handler.add_to_loggers(*loggers)

    return handler