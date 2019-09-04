from collections import defaultdict
import time
import functools

timing = defaultdict(float)


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        timing[func.__name__] += run_time
        # print(f"{func.__name__!r}: {run_time:.4f}")
        return value

    return wrapper_timer
