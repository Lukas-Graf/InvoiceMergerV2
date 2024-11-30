"""
Module holds all the template loggers and
writes messages into ./logs.log

File is written in pylint standard
"""

import sys
import logging


def get_logger(name: str = "LOG", level: str = "INFO", file: str = "logs") -> logging.Logger:
    """
    Creates and configures a logger for the script, logging to both a file and the console.

    Parameters
    ----------
    name : str, optional (default="LOG")
        Name of the logger
    level : str, optional (default="INFO")
        Logging level (e.g., "DEBUG", "INFO", "WARNING")
    file : str, optional (default="logs")
        File name (without extension) where the logs will be written

    Returns
    -------
    logging.Logger
        The configured logger instance
    """ 

    logger = logging.getLogger(name=name)
    if not logger.hasHandlers():
        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - |%(levelname)s| ->%(message)s<- (%(filename)s, line.%(lineno)d)",
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(f"./{file}.log")
        stream_handler = logging.StreamHandler(sys.stdout)

        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Prevent propagation to the root logger to avoid duplicate logs
        logger.propagate = False

    return logger

if __name__ == "__main__":
    logger_test = get_logger()
    logger_test.info("Works fine!")
    