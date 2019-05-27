from datetime import datetime
import functools
import logging


class Monitor(object):
    def __init__(self, task_name):
        self.__task_name = task_name

    @property
    def task_name(self):
        return self.__task_name

    def __enter__(self):
        self.__start_time = datetime.now()

        logging.info('try to {0}...'.format(self.__task_name))

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # https://docs.python.org/3/reference/datamodel.html?highlight=__exit__#object.__exit__
        self.__end_time = datetime.now()

        elapsed = self.__end_time.timestamp() - self.__start_time.timestamp()

        if not exc_type:
            logging.info('{0} successfully, elapsed: {1:.3f} s'.format(
                self.__task_name, elapsed))
        else:
            logging.error('{0} failed, elapsed:{1:.3f} s'.format(
                self.__task_name, elapsed))


def monitor(task_name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            start_time = datetime.now()

            name = func.__name__
            if task_name:
                name = task_name

            logging.info('try to {0}...'.format(name))

            try:
                result = func(*args, **kw)

                end_time = datetime.now()
                elapsed = end_time.timestamp() - start_time.timestamp()

                logging.info('{0} successfully, elapsed: {1:.3f} s'.format(
                    name, elapsed))

                return result
            except BaseException:
                end_time = datetime.now()
                elapsed = end_time.timestamp() - start_time.timestamp()

                logging.error('{0} failed, elapsed:{1:.3f} s'.format(
                    name, elapsed))

                raise

        return wrapper

    return decorator
