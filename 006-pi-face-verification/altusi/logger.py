"""
Logger class
============

Wrapper of built-in `logging` Python module
that supports logging task to both console and files
"""

"""
Revision
--------
    2019, Sep 24: first version
"""


import os
import logging

# format for logging message
LOG_FILE_FORMAT = '%(asctime)s  %(name)12s  [%(levelname)s]  %(message)s'
LOG_CONSOLE_FORMAT = '%(asctime)s  [%(levelname)s]  %(message)s'

class Logger():
    """Logger class for logging task"""

    def __init__(self, name, console=True):
        """Initialize logger

        Parameters
        ----------
        name : str
            name of logger and file to log
        console : bool
            whether logging messages are emitted to console or not
        """
        self.logger = logging.getLogger(name)
        
        self.__configLogger(name, console)

    
    def __configLogger(self, name, console):
        """Configure logger object

        Parameters
        ----------
        name : str
            name of logging file
        console : bool
            whether logging messages are emitted to console or not
        """
        self.logger.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(LOG_FILE_FORMAT)
        console_formatter = logging.Formatter(LOG_CONSOLE_FORMAT)

        # if `console` is set, setup Handler to process 
        if console:
            console_log = logging.StreamHandler()
            console_log.setLevel(logging.DEBUG)
            console_log.setFormatter(console_formatter)

        file_log = logging.FileHandler(name + '.log',
                                      mode='w')
        file_log.setLevel(logging.DEBUG)
        file_log.setFormatter(file_formatter)

        if console:
            self.logger.addHandler(console_log)
        self.logger.addHandler(file_log)


    def critical(self, msg):
        """Emit CRITICAL message"""
        self.logger.critical(msg)


    def error(self, msg):
        """Emit ERROR message"""
        self.logger.error(msg)


    def warning(self, msg):
        """Emit WARNING message"""
        self.logger.warning(msg)


    def info(self, msg):
        """Emit INFO message"""
        self.logger.info(msg)


    def debug(self, msg):
        """Emit DEBUG message"""
        self.logger.debug(msg)
