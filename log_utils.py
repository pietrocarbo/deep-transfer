import logging


def get_logger():
    console_logs_lvl = logging.DEBUG # CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET
    console_logs_format = '%(asctime)s - [%(levelname)s] - %(name)s:%(pathname)s:%(lineno)d - %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(console_logs_lvl)
    logging.basicConfig(format=console_logs_format)

    file_logs_lvl = logging.INFO
    file_logs_format = '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
    handler = logging.FileHandler('logs.txt', mode='w')
    handler.setLevel(file_logs_lvl)
    handler.setFormatter(logging.Formatter(file_logs_format))
    logger.addHandler(handler)

    return logger