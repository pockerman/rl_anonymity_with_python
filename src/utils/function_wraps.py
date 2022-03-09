from functools import wraps
from typing import Callable
import time

from src.utils import INFO


def time_func(fn: Callable):
    """Execute the given callable and time the time
     it took to execute

    Parameters
    ----------
    fn: Callable to execute

    Returns
    -------

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


def time_func_wrapper(show_time: bool):
    def _time_func(fn: Callable):
        @wraps(fn)
        def _measure(*args, **kwargs):
            time_start = time.perf_counter()
            result = fn(*args, **kwargs)
            time_end = time.perf_counter()
            if show_time:
                print("{0} Done. Execution time {1} secs".format(INFO, time_end - time_start))
            return result, time_end - time_start
        return _measure
    return _time_func

