"""
Self-Supervised Learning for Recognition of Sports Poses in Image - Master's Thesis Project
Module for timing of a function via decorator.
Organisation: Brno University of Technology - Faculty of Information Technology
Author: Daniel Konecny (xkonec75)
Date: 22. 11. 2021
"""

import functools
import time


def timer(func):
    """
    Print the runtime of the decorated function
    Source: https://realpython.com/primer-on-python-decorators/#timing-functions
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs.")
        return value

    return wrapper_timer
