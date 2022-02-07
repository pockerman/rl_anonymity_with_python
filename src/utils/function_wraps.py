from functools import wraps
from typing import Callable
import time

from src.utils import INFO


def time_func(fn: Callable):
    """
    Execute the given callable and time the time
    it tool to execute
    :param fn:
    :return:
    """
    @wraps(fn)
    def measure(*args, **kwargs):
        time_start = time.perf_counter()
        result = fn(*args, **kwargs)
        time_end = time.perf_counter()
        print("{0} Done. Execution time"
              " {1} secs".format(INFO, time_end - time_start))
        return result
    return measure
