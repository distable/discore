import os
import signal
from pathlib import Path
from typing import Callable
from jsonic import deserialize, serialize, jsonic_serializer, jsonic_deserializer

def first_non_null(*args):
    for value in args:
        if value is not None:
            return value
    return None

def wserialize(obj):
    """
    Weird serialization
    Clears the classlib prefix from the class name.
    """
    ret = serialize(obj, string_output=True)
    ret = ret.replace('src.classes', '__CLASSLIB__')

    return ret


def wdeserialize(s, classlib: str):
    """
    Weird serialization
    jsonic serializes class names, but the full path won't be the same
    across every installation. This function replaces the class name with
    the proper prefix.
    """
    s = s.replace('__CLASSLIB__', classlib)
    ret = deserialize(s)

    return ret

@jsonic_serializer(serialized_type=Path)
def serialize_path(path: Path):
    return {'path': path.as_posix()}

@jsonic_deserializer(Path)
def serialize_path(serialized_path):
    return Path(serialized_path['path'])

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
