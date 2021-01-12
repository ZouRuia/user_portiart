import os
import sys
from loguru import logger

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(APP_DIR)


def get_log_file(name):
    return os.path.join(APP_DIR, "logs", "{}.log".format(name))


# logger.add(get_log_file("debug.{time:YYYY-MM-DD}"), level="DEBUG", rotation="00:00",
#            compression="zip", filter=lambda record: record["level"].name == "DEBUG", backtrace=True, diagnose=False)
logger.add(get_log_file("info.{time:YYYY-MM-DD}"), level="INFO", rotation="00:00",
           compression="zip", filter=lambda record: record["level"].name == "INFO", backtrace=True, diagnose=False)
logger.add(get_log_file("error.{time:YYYY-MM-DD}"), level="ERROR", rotation="00:00",
           compression="zip", filter=lambda record: record["level"].name == "ERROR", backtrace=True, diagnose=False)


def func(a, b):
    return a / b


def nested(c):
    try:
        func(5, c)
    except ZeroDivisionError as e:
        logger.debug(e)
        logger.exception(e)


if __name__ == "__main__":
    nested(0)
