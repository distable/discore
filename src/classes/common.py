import os
import signal
from pathlib import Path
from typing import Callable

def first_non_null(*args):
    for value in args:
        if value is not None:
            return value
    return None

def extract_dict(obj, *names):
    d = {}
    for x in names:
        v = getattr(obj, x)
        if isinstance(v, Callable):
            d[x] = v()
        else:
            d[x] = v
    return d


def setup_ctrl_c(func=None):
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        if func:
            func()
        else:
            os._exit(0)

    # CTRL-C handler
    signal.signal(signal.SIGINT, sigint_handler)
